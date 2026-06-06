import { describe, expect, it } from "bun:test";
import { hookFetch } from "@oh-my-pi/pi-utils";
import type { AssistantMessage, Context, Model, Tool } from "../src";
import { streamBedrock } from "../src/providers/amazon-bedrock";
import { clearAwsCredentialCache } from "../src/providers/aws-credentials";
import { eventFrame, eventStreamResponse, snapshotAwsEnv } from "./helpers";

async function captureBedrockResult(
	model: Model<"bedrock-converse-stream">,
	context: Context,
	frames: Uint8Array[],
): Promise<AssistantMessage> {
	using _hook = hookFetch(() => eventStreamResponse(frames));
	return await streamBedrock(model, context, {}).result();
}

function abortedSignal(): AbortSignal {
	const controller = new AbortController();
	controller.abort();
	return controller.signal;
}

function baseModel(overrides: Partial<Model<"bedrock-converse-stream">> = {}): Model<"bedrock-converse-stream"> {
	return {
		id: "claude-sonnet-4",
		name: "Claude Sonnet 4",
		api: "bedrock-converse-stream",
		provider: "amazon-bedrock",
		baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 8_192,
		...overrides,
	};
}

const baseContext: Context = {
	systemPrompt: [],
	messages: [{ role: "user", content: "say hi", timestamp: Date.now() }],
};

interface BedrockPayload {
	messages?: Array<{
		role: string;
		content?: Array<Record<string, unknown>>;
	}>;
	toolConfig?: {
		tools?: Array<{ toolSpec: { name: string } }>;
		toolChoice?: { tool?: { name: string } };
	};
}

interface CapturedRequest {
	url: string;
	headers: Headers;
	payload: BedrockPayload;
}

async function captureBedrockRequest(
	model: Model<"bedrock-converse-stream">,
	context: Context = baseContext,
	options: Parameters<typeof streamBedrock>[2] = {},
): Promise<CapturedRequest> {
	let capturedUrl = "";
	let capturedHeaders: Headers | undefined;
	let capturedPayload: BedrockPayload = {};

	using _hook = hookFetch((input, init) => {
		capturedUrl = String(input);
		capturedHeaders = new Headers(init?.headers);
		return new Response('{"message":"unauthorized"}', { status: 401 });
	});

	await streamBedrock(model, context, {
		...options,
		onPayload: payload => {
			capturedPayload = payload as BedrockPayload;
		},
	}).result();

	expect(capturedHeaders).toBeDefined();
	return { url: capturedUrl, headers: capturedHeaders!, payload: capturedPayload };
}

async function captureBedrockPayload(
	model: Model<"bedrock-converse-stream">,
	context: Context = baseContext,
	options: Parameters<typeof streamBedrock>[2] = {},
): Promise<BedrockPayload> {
	const { promise, resolve } = Promise.withResolvers<BedrockPayload>();
	void streamBedrock(model, context, {
		signal: abortedSignal(),
		...options,
		onPayload: payload => {
			resolve(payload as BedrockPayload);
		},
	});
	return promise;
}

function assistantStub(
	content: AssistantMessage["content"],
	overrides: Partial<AssistantMessage> = {},
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "bedrock-converse-stream",
		provider: "amazon-bedrock",
		model: "claude-sonnet-4",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
		...overrides,
	};
}

const fooBarTool: Tool = {
	name: "foo.bar",
	description: "demo tool",
	parameters: {
		type: "object",
		properties: {},
	} as unknown as Tool["parameters"],
};

describe("amazon-bedrock custom endpoint routing", () => {
	describe("default bundled AWS routing", () => {
		it("ignores bundled amazonaws.com model.baseUrl and uses options.region for SigV4 host", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({ baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com" }),
					baseContext,
					{ region: "eu-west-1" },
				);

				expect(request.url).toBe(
					"https://bedrock-runtime.eu-west-1.amazonaws.com/model/claude-sonnet-4/converse-stream",
				);
				expect(request.headers.get("authorization")).toStartWith("AWS4-HMAC-SHA256 ");
				expect(request.headers.has("x-amz-date")).toBe(true);
			} finally {
				restoreAwsEnv();
			}
		});

		it("uses options.baseUrl override when model.baseUrl is bundled AWS", async () => {
			const request = await captureBedrockRequest(
				baseModel({ baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com" }),
				baseContext,
				{ baseUrl: "https://gateway.example.com" },
			);

			expect(request.url).toBe("https://gateway.example.com/model/claude-sonnet-4/converse-stream");
		});

		it("signs default AWS requests with SigV4 when no custom auth is configured", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_REGION = "us-west-2";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(baseModel());

				expect(request.url).toBe(
					"https://bedrock-runtime.us-west-2.amazonaws.com/model/claude-sonnet-4/converse-stream",
				);
				expect(request.headers.get("authorization")).toStartWith("AWS4-HMAC-SHA256 ");
				expect(request.headers.has("x-amz-date")).toBe(true);
			} finally {
				restoreAwsEnv();
			}
		});

		it("honors providers.amazon-bedrock baseUrl override on bundled models", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "amazon-bedrock",
						baseUrl: "https://bedrock-proxy.example.com",
					}),
				);

				expect(request.url).toBe(
					"https://bedrock-proxy.example.com/model/claude-sonnet-4/converse-stream",
				);
				expect(request.headers.get("authorization")).toBeNull();
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("preserves provider headers auth on bundled amazon-bedrock baseUrl override", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "amazon-bedrock",
						baseUrl: "https://bedrock-proxy.example.com",
						headers: { Authorization: "Bearer gateway-token" },
					}),
				);

				expect(request.url).toBe(
					"https://bedrock-proxy.example.com/model/claude-sonnet-4/converse-stream",
				);
				expect(request.headers.get("authorization")).toBe("Bearer gateway-token");
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});
	});

	describe("custom provider endpoint routing", () => {
		it("uses custom model.baseUrl verbatim", async () => {
			const request = await captureBedrockRequest(
				baseModel({
					provider: "custom-bedrock",
					baseUrl: "https://custom-bedrock.example.com/v1",
				}),
			);

			expect(request.url).toBe("https://custom-bedrock.example.com/v1/model/claude-sonnet-4/converse-stream");
		});

		it("strips trailing slash from custom model.baseUrl", async () => {
			const request = await captureBedrockRequest(
				baseModel({
					provider: "custom-bedrock",
					baseUrl: "https://custom-bedrock.example.com/",
				}),
			);

			expect(request.url).toBe("https://custom-bedrock.example.com/model/claude-sonnet-4/converse-stream");
		});

		it("encodes model id in the converse-stream path", async () => {
			const request = await captureBedrockRequest(
				baseModel({
					id: "anthropic/claude:sonnet",
					provider: "custom-bedrock",
					baseUrl: "https://custom-bedrock.example.com/v1",
				}),
			);

			expect(request.url).toBe(
				`https://custom-bedrock.example.com/v1/model/${encodeURIComponent("anthropic/claude:sonnet")}/converse-stream`,
			);
		});

		it("preserves custom regional AWS Bedrock model.baseUrl", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://bedrock-runtime.eu-west-1.amazonaws.com",
					}),
				);

				expect(request.url).toBe(
					"https://bedrock-runtime.eu-west-1.amazonaws.com/model/claude-sonnet-4/converse-stream",
				);
				const authorization = request.headers.get("authorization");
				expect(authorization).toStartWith("AWS4-HMAC-SHA256 ");
				expect(authorization).toContain("/eu-west-1/bedrock/");
				expect(authorization).not.toContain("/us-east-1/bedrock/");
				expect(request.headers.has("x-amz-date")).toBe(true);
			} finally {
				restoreAwsEnv();
			}
		});

		it("signs FIPS Bedrock endpoint overrides with the host region", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel(),
					baseContext,
					{ baseUrl: "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com" },
				);

				expect(request.url).toBe(
					"https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com/model/claude-sonnet-4/converse-stream",
				);
				const authorization = request.headers.get("authorization");
				expect(authorization).toStartWith("AWS4-HMAC-SHA256 ");
				expect(authorization).toContain("/us-gov-west-1/bedrock/");
				expect(authorization).not.toContain("/us-east-1/bedrock/");
			} finally {
				restoreAwsEnv();
			}
		});

		it("signs AWS Bedrock endpoints with explicit :443 port", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com:443",
					}),
				);

				expect(request.url).toBe(
					"https://bedrock-runtime.us-east-1.amazonaws.com:443/model/claude-sonnet-4/converse-stream",
				);
				const authorization = request.headers.get("authorization");
				expect(authorization).toStartWith("AWS4-HMAC-SHA256 ");
				expect(authorization).toContain("/us-east-1/bedrock/");
			} finally {
				restoreAwsEnv();
			}
		});

		it("signs AWS China Bedrock endpoints with SigV4", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://bedrock-runtime.cn-north-1.amazonaws.com.cn",
					}),
				);

				expect(request.url).toBe(
					"https://bedrock-runtime.cn-north-1.amazonaws.com.cn/model/claude-sonnet-4/converse-stream",
				);
				const authorization = request.headers.get("authorization");
				expect(authorization).toStartWith("AWS4-HMAC-SHA256 ");
				expect(authorization).toContain("/cn-north-1/bedrock/");
			} finally {
				restoreAwsEnv();
			}
		});
	});

	describe("custom auth and headers", () => {
		it("skips SigV4 when model.headers include X-Api-Key", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://custom-bedrock.example.com",
						headers: { "X-Api-Key": "secret-key" },
					}),
				);

				expect(request.headers.has("x-amz-date")).toBe(false);
				expect(request.headers.get("x-api-key")).toBe("secret-key");
			} finally {
				restoreAwsEnv();
			}
		});

		it("skips SigV4 when model.headers use lowercase authorization", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://custom-bedrock.example.com",
						headers: { authorization: "Bearer custom" },
					}),
				);

				expect(request.headers.has("x-amz-date")).toBe(false);
				expect(request.headers.get("authorization")).toBe("Bearer custom");
			} finally {
				restoreAwsEnv();
			}
		});

		it("skips SigV4 for no-auth custom endpoint (no bearer, no headers)", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://custom-bedrock.example.com",
					}),
				);

				expect(request.headers.has("x-amz-date")).toBe(false);
				expect(request.headers.has("authorization")).toBe(false);
				expect(request.url).toBe("https://custom-bedrock.example.com/model/claude-sonnet-4/converse-stream");
			} finally {
				restoreAwsEnv();
			}
		});

		it("merges options.headers into request headers", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://custom-bedrock.example.com",
					}),
					baseContext,
					{ headers: { "X-Request-Id": "req-123", "X-Tenant": "acme" } },
				);

				expect(request.headers.get("x-request-id")).toBe("req-123");
				expect(request.headers.get("x-tenant")).toBe("acme");
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("merges bearer apiKey with custom model headers", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://custom-bedrock.example.com",
						headers: { "X-Custom": "value" },
					}),
					baseContext,
					{ apiKey: "bedrock-token" },
				);

				expect(request.headers.get("authorization")).toBe("Bearer bedrock-token");
				expect(request.headers.get("x-custom")).toBe("value");
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("prefers bearer auth over SigV4 when AWS credentials are also available", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_BEARER_TOKEN_BEDROCK = "bedrock-api-key";
				Bun.env.AWS_REGION = "us-west-2";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(baseModel());

				expect(request.headers.get("authorization")).toBe("Bearer bedrock-api-key");
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});
	});

	describe("endpoint and auth decoupling", () => {
		it("signs AWS endpoint overrides (FIPS) with SigV4", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				Bun.env.AWS_BEDROCK_SKIP_AUTH = "1";
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(baseModel(), baseContext, {
					baseUrl: "https://bedrock-runtime-fips.us-east-1.amazonaws.com",
				});

				expect(request.url).toBe(
					"https://bedrock-runtime-fips.us-east-1.amazonaws.com/model/claude-sonnet-4/converse-stream",
				);
				expect(request.headers.get("authorization")).toStartWith("AWS4-HMAC-SHA256 ");
				expect(request.headers.has("x-amz-date")).toBe(true);
			} finally {
				restoreAwsEnv();
			}
		});

		it("prefers options.baseUrl over a custom model.baseUrl", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({ provider: "custom-bedrock", baseUrl: "https://model-endpoint.example.com" }),
					baseContext,
					{ baseUrl: "https://override.example.com" },
				);

				expect(request.url).toBe("https://override.example.com/model/claude-sonnet-4/converse-stream");
			} finally {
				restoreAwsEnv();
			}
		});

		it("honors options.headers authorization and skips SigV4 on AWS hosts", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_REGION = "us-east-1";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(baseModel(), baseContext, {
					headers: { authorization: "Bearer per-request" },
				});

				expect(request.headers.get("authorization")).toBe("Bearer per-request");
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("sends no Authorization for a no-auth (kNoAuth) custom endpoint", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({ provider: "custom-bedrock", baseUrl: "https://custom-bedrock.example.com" }),
					baseContext,
					{ apiKey: "N/A" },
				);

				expect(request.headers.has("authorization")).toBe(false);
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("ignores AWS_BEARER_TOKEN_BEDROCK when kNoAuth sentinel is present", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_BEARER_TOKEN_BEDROCK = "bedrock-api-key";
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({ provider: "custom-bedrock", baseUrl: "https://custom-bedrock.example.com" }),
					baseContext,
					{ apiKey: "N/A" },
				);

				expect(request.headers.has("authorization")).toBe(false);
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("skips SigV4 for kNoAuth even when baseUrl is an AWS hostname", async () => {
			const restoreAwsEnv = snapshotAwsEnv();
			try {
				delete Bun.env.AWS_ACCESS_KEY_ID;
				delete Bun.env.AWS_SECRET_ACCESS_KEY;
				delete Bun.env.AWS_BEARER_TOKEN_BEDROCK;
				delete Bun.env.AWS_BEDROCK_SKIP_AUTH;
				Bun.env.AWS_EC2_METADATA_DISABLED = "true";
				clearAwsCredentialCache();

				const request = await captureBedrockRequest(
					baseModel({
						provider: "custom-bedrock",
						baseUrl: "https://abc123.execute-api.us-east-1.amazonaws.com",
					}),
					baseContext,
					{ apiKey: "N/A" },
				);

				expect(request.url).toStartWith("https://abc123.execute-api.us-east-1.amazonaws.com/");
				expect(request.headers.has("authorization")).toBe(false);
				expect(request.headers.has("x-amz-date")).toBe(false);
			} finally {
				restoreAwsEnv();
			}
		});

		it("maps sanitized tool names back to the original on receive", async () => {
			const context: Context = { ...baseContext, tools: [fooBarTool] };
			const frames = [
				eventFrame("messageStart", { role: "assistant" }),
				eventFrame("contentBlockStart", {
					contentBlockIndex: 0,
					start: { toolUse: { toolUseId: "toolu_01", name: "foo_bar" } },
				}),
				eventFrame("contentBlockDelta", { contentBlockIndex: 0, delta: { toolUse: { input: "{}" } } }),
				eventFrame("contentBlockStop", { contentBlockIndex: 0 }),
				eventFrame("messageStop", { stopReason: "tool_use" }),
			];

			const result = await captureBedrockResult(
				baseModel({ provider: "custom-bedrock", baseUrl: "https://custom-bedrock.example.com" }),
				context,
				frames,
			);

			const toolCall = result.content.find(block => block.type === "toolCall");
			expect(toolCall?.type).toBe("toolCall");
			if (toolCall?.type === "toolCall") expect(toolCall.name).toBe("foo.bar");
		});
	});

	describe("payload conversion", () => {
		it("sanitizes tool definitions for Bedrock wire format", async () => {
			const context: Context = {
				...baseContext,
				tools: [fooBarTool],
			};

			const payload = await captureBedrockPayload(baseModel(), context, {
				toolChoice: { type: "tool", name: "foo.bar" },
			});

			expect(payload.toolConfig?.tools?.[0]?.toolSpec.name).toBe("foo_bar");
			expect(payload.toolConfig?.toolChoice?.tool?.name).toBe("foo_bar");
		});

		it("rejects colliding sanitized tool names before sending", async () => {
			const context: Context = {
				...baseContext,
				tools: [
					fooBarTool,
					{
						...fooBarTool,
						name: "foo/bar",
					},
				],
			};

			const result = await streamBedrock(baseModel(), context, {}).result();

			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain(
				'Bedrock tool name collision after sanitization: "foo.bar" and "foo/bar" both map to "foo_bar"',
			);
		});
		it("skips tool-name collision checks when toolChoice is none", async () => {
			const context: Context = {
				...baseContext,
				tools: [
					fooBarTool,
					{
						...fooBarTool,
						name: "foo/bar",
					},
				],
			};

			const payload = await captureBedrockPayload(baseModel(), context, { toolChoice: "none" });

			expect(payload.toolConfig).toBeUndefined();
		});

		it("sanitizes assistant tool replay names in message history", async () => {
			const context: Context = {
				systemPrompt: [],
				messages: [
					{ role: "user", content: "call the tool", timestamp: Date.now() },
					assistantStub([{ type: "toolCall", id: "toolu_01", name: "foo.bar", arguments: { x: 1 } }]),
				],
			};

			const payload = await captureBedrockPayload(baseModel(), context);
			const assistantMessage = payload.messages?.find(message => message.role === "assistant");
			const toolUse = assistantMessage?.content?.find(block => "toolUse" in block) as
				| { toolUse: { name: string } }
				| undefined;

			expect(toolUse?.toolUse.name).toBe("foo_bar");
		});

		it("keeps bare claude-* thinking blocks as reasoningContent when signature is present", async () => {
			const context: Context = {
				systemPrompt: [],
				messages: [
					{ role: "user", content: "think", timestamp: Date.now() },
					assistantStub(
						[
							{ type: "thinking", thinking: "internal reasoning", thinkingSignature: "sig-123" },
							{ type: "text", text: "answer" },
						],
						{
							provider: "custom-bedrock",
							api: "bedrock-converse-stream",
							model: "claude-sonnet-4",
						},
					),
				],
			};

			const payload = await captureBedrockPayload(
				baseModel({
					id: "claude-sonnet-4",
					provider: "custom-bedrock",
					baseUrl: "https://custom-bedrock.example.com",
				}),
				context,
			);

			const assistantMessage = payload.messages?.find(message => message.role === "assistant");
			const reasoningBlock = assistantMessage?.content?.find(block => "reasoningContent" in block) as
				| { reasoningContent: { reasoningText: { text: string; signature?: string } } }
				| undefined;
			const textBlock = assistantMessage?.content?.find(block => "text" in block) as { text: string } | undefined;

			expect(reasoningBlock?.reasoningContent.reasoningText).toMatchObject({
				text: "internal reasoning",
				signature: "sig-123",
			});
			expect(textBlock?.text).toBe("answer");
		});
	});
});
