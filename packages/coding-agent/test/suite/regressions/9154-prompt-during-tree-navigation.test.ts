import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { assistantMsg, userMsg } from "../../utilities.ts";
import { createHarness, getMessageText, getUserTexts, type Harness } from "../harness.ts";

function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve!: () => void;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

function seedHistory(harness: Harness): string {
	harness.sessionManager.appendMessage(userMsg("first prompt"));
	const targetId = harness.sessionManager.appendMessage(assistantMsg("first response"));
	harness.sessionManager.appendMessage(userMsg("second prompt"));
	harness.sessionManager.appendMessage(assistantMsg("second response"));
	harness.session.refreshContext();
	return targetId;
}

describe("issue #9154: prompt during tree navigation", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("rejects a prompt while branch summarization is in progress", async () => {
		let markSummaryStarted = () => {};
		const summaryStarted = new Promise<void>((resolve) => {
			markSummaryStarted = resolve;
		});
		let releaseSummary = () => {};
		const summaryReleased = new Promise<void>((resolve) => {
			releaseSummary = resolve;
		});

		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.on("session_before_tree", async () => {
						markSummaryStarted();
						await summaryReleased;
						return { summary: { summary: "abandoned branch summary" } };
					});
				},
			],
		});
		harnesses.push(harness);

		const timestamp = Date.now();
		const targetId = harness.sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "first prompt" }],
			timestamp: timestamp - 1500,
		});
		harness.sessionManager.appendMessage(fauxAssistantMessage("first response", { timestamp: timestamp - 1000 }));
		harness.sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "abandoned prompt" }],
			timestamp: timestamp - 500,
		});
		harness.sessionManager.appendMessage(fauxAssistantMessage("abandoned response", { timestamp }));
		harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;
		harness.setResponses([fauxAssistantMessage("unexpected response")]);

		const navigationPromise = harness.session.navigateTree(targetId, { summarize: true });
		await summaryStarted;

		const preflightResult = vi.fn();
		let promptError: unknown;
		try {
			await harness.session.prompt("prompt during summary", {
				source: "rpc",
				preflightResult,
			});
		} catch (error) {
			promptError = error;
		} finally {
			releaseSummary();
			await navigationPromise;
		}

		const persistedUserTexts = harness.sessionManager
			.getEntries()
			.flatMap((entry) =>
				entry.type === "message" && entry.message.role === "user" ? [getMessageText(entry.message)] : [],
			);

		expect(preflightResult).not.toHaveBeenCalled();
		expect(promptError).toEqual(
			expect.objectContaining({ message: expect.stringContaining("compaction is in progress") }),
		);
		expect(getUserTexts(harness)).not.toContain("prompt during summary");
		expect(persistedUserTexts).not.toContain("prompt during summary");
		expect(harness.eventsOfType("agent_start")).toHaveLength(0);
		expect(harness.eventsOfType("agent_settled")).toHaveLength(0);
	});

	// The inverse order also loses the completed prompt when navigation rebuilds the context.
	it.each(["input", "before_agent_start"] as const)("rejects navigation during an awaited %s hook", async (hook) => {
		const promptStarted = deferred();
		const promptReleased = deferred();
		const treeStarted = deferred();
		const treeReleased = deferred();
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					const pause = async () => {
						promptStarted.resolve();
						await promptReleased.promise;
					};
					if (hook === "input") pi.on("input", pause);
					else pi.on("before_agent_start", pause);
					pi.on("session_before_tree", async () => {
						treeStarted.resolve();
						await treeReleased.promise;
					});
				},
			],
		});
		harnesses.push(harness);
		const targetId = seedHistory(harness);
		harness.setResponses([fauxAssistantMessage("response to racing prompt")]);
		const prompt = harness.session.prompt("racing prompt", { source: "rpc" });
		await promptStarted.promise;
		const navigation = harness.session.navigateTree(targetId).then(
			() => undefined,
			(error: unknown) => error,
		);
		await Promise.race([treeStarted.promise, navigation]);
		try {
			promptReleased.resolve();
			await prompt;
		} finally {
			promptReleased.resolve();
			treeReleased.resolve();
		}
		const navigationError = await navigation;
		expect(navigationError).toEqual(
			expect.objectContaining({
				message: "Wait for the current response to finish before navigating the session tree.",
			}),
		);
		expect(getUserTexts(harness)).toContain("racing prompt");
		expect(harness.session.messages.map(getMessageText)).toContain("response to racing prompt");
		await expect(harness.session.navigateTree(targetId)).resolves.toMatchObject({ cancelled: false });
	});

	it.each(["handled", "rejected"] as const)(
		"allows navigation after a prompt is %s during preflight",
		async (outcome) => {
			const harness = await createHarness({
				withConfiguredAuth: outcome !== "rejected",
				extensionFactories: [
					(pi) => {
						if (outcome === "handled") pi.on("input", async () => ({ action: "handled" }));
					},
				],
			});
			harnesses.push(harness);
			const targetId = seedHistory(harness);
			const prompt = harness.session.prompt("not sent to the model");
			if (outcome === "handled") await prompt;
			else await expect(prompt).rejects.toThrow("No API key");
			await expect(harness.session.navigateTree(targetId)).resolves.toMatchObject({ cancelled: false });
		},
	);

	it.each(["command", "agent_settled"] as const)("allows navigation from an extension %s", async (event) => {
		let targetId = "";
		let navigated = false;
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					const navigate = async () => {
						await harness.session.navigateTree(targetId);
						navigated = true;
					};
					if (event === "command") pi.registerCommand("rewind", { handler: navigate });
					else pi.on("agent_settled", navigate);
				},
			],
		});
		harnesses.push(harness);
		targetId = seedHistory(harness);
		harness.setResponses([fauxAssistantMessage("done")]);
		await harness.session.prompt(event === "command" ? "/rewind" : "finish then rewind");
		expect(navigated).toBe(true);
		expect(harness.sessionManager.getLeafId()).toBe(targetId);
	});
});
