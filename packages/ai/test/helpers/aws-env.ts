import { clearAwsCredentialCache } from "@oh-my-pi/pi-ai/providers/aws-credentials";

/** AWS env vars that influence Bedrock credential/endpoint resolution. */
export const awsEnvKeys = [
	"AWS_ACCESS_KEY_ID",
	"AWS_SECRET_ACCESS_KEY",
	"AWS_SESSION_TOKEN",
	"AWS_PROFILE",
	"AWS_REGION",
	"AWS_DEFAULT_REGION",
	"AWS_CONFIG_FILE",
	"AWS_SHARED_CREDENTIALS_FILE",
	"AWS_EC2_METADATA_DISABLED",
	"AWS_BEARER_TOKEN_BEDROCK",
	"AWS_BEDROCK_SKIP_AUTH",
] as const;

/**
 * Snapshots the AWS env vars and returns a restore function. Tests mutate
 * `Bun.env` within a `try` and call the returned restorer in `finally` so the
 * mutation never leaks into other files. Also clears the credential cache so a
 * later test does not observe a stale resolution.
 */
export function snapshotAwsEnv(): () => void {
	const previous = new Map<string, string | undefined>();
	for (const key of awsEnvKeys) previous.set(key, Bun.env[key]);
	return () => {
		for (const key of awsEnvKeys) {
			const value = previous.get(key);
			if (value === undefined) delete Bun.env[key];
			else Bun.env[key] = value;
		}
		clearAwsCredentialCache();
	};
}
