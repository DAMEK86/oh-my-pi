import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Snowflake } from "@oh-my-pi/pi-utils";
import { ModelRegistry } from "../src/config/model-registry";
import { ModelsConfigSchema } from "../src/config/models-config-schema";
import { resetSettingsForTest } from "../src/config/settings";
import { AuthStorage } from "../src/session/auth-storage";

describe("models config: bedrock-converse-stream", () => {
	describe("schema", () => {
		test("accepts provider-level bedrock-converse-stream config", () => {
			const result = ModelsConfigSchema.safeParse({
				providers: {
					"custom-bedrock": {
						api: "bedrock-converse-stream",
						baseUrl: "https://custom-bedrock.example.com",
						headers: { "X-Api-Key": "test-key" },
						models: [{ id: "claude-sonnet-4", api: "bedrock-converse-stream" }],
					},
				},
			});

			expect(result.success).toBe(true);
		});

		test("accepts model-level bedrock-converse-stream overrides", () => {
			const result = ModelsConfigSchema.safeParse({
				providers: {
					"custom-bedrock": {
						api: "bedrock-converse-stream",
						baseUrl: "https://custom-bedrock.example.com",
						models: [
							{
								id: "claude-sonnet-4",
								api: "bedrock-converse-stream",
								baseUrl: "https://custom-bedrock.example.com/v2",
								headers: { "X-Api-Key": "model-key" },
							},
						],
					},
				},
			});

			expect(result.success).toBe(true);
		});

		test("rejects invalid provider api values", () => {
			const result = ModelsConfigSchema.safeParse({
				providers: {
					"custom-bedrock": {
						api: "not-a-real-api",
						baseUrl: "https://custom-bedrock.example.com",
						models: [{ id: "claude-sonnet-4" }],
					},
				},
			});

			expect(result.success).toBe(false);
		});
	});

	describe("registry", () => {
		let tempDir: string;
		let modelsJsonPath: string;
		let authStorage: AuthStorage;

		beforeEach(async () => {
			resetSettingsForTest();
			tempDir = path.join(os.tmpdir(), `pi-test-models-config-bedrock-${Snowflake.next()}`);
			fs.mkdirSync(tempDir, { recursive: true });
			modelsJsonPath = path.join(tempDir, "models.json");
			authStorage = await AuthStorage.create(path.join(tempDir, "testauth.db"));
		});

		afterEach(() => {
			resetSettingsForTest();
			authStorage.close();
			if (tempDir && fs.existsSync(tempDir)) {
				fs.rmSync(tempDir, { recursive: true });
			}
		});

		test("loads custom bedrock provider models with headers from models.json", () => {
			fs.writeFileSync(
				modelsJsonPath,
				JSON.stringify({
					providers: {
						"custom-bedrock": {
							api: "bedrock-converse-stream",
							baseUrl: "https://custom-bedrock.example.com",
							auth: "none",
							headers: { "X-Api-Key": "test-key" },
							models: [
								{
									id: "claude-sonnet-4",
									name: "Claude Sonnet 4",
									api: "bedrock-converse-stream",
									reasoning: false,
									input: ["text"],
									cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
									contextWindow: 200_000,
									maxTokens: 8192,
								},
							],
						},
					},
				}),
			);

			const registry = new ModelRegistry(authStorage, modelsJsonPath);
			const model = registry.find("custom-bedrock", "claude-sonnet-4");

			expect(model).toBeDefined();
			expect(model?.api).toBe("bedrock-converse-stream");
			expect(model?.baseUrl).toBe("https://custom-bedrock.example.com");
			expect(model?.headers?.["X-Api-Key"]).toBe("test-key");
		});
	});
});
