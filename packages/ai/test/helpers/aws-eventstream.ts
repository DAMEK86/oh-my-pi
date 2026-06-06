import { crc32 } from "@oh-my-pi/pi-ai/providers/aws-eventstream";

// `application/vnd.amazon.eventstream` frame builder. Mirrors the codec so tests
// own the bytes; the decoder in `aws-eventstream.ts` is the production code under
// test. Encoding lives only here for fixture generation.

export function encodeStringHeader(name: string, value: string): Uint8Array {
	const nameBytes = new TextEncoder().encode(name);
	const valueBytes = new TextEncoder().encode(value);
	if (nameBytes.length > 255) throw new Error("name too long");
	const buf = new Uint8Array(1 + nameBytes.length + 1 + 2 + valueBytes.length);
	const view = new DataView(buf.buffer);
	let p = 0;
	view.setUint8(p, nameBytes.length);
	p += 1;
	buf.set(nameBytes, p);
	p += nameBytes.length;
	view.setUint8(p, 7); // string type
	p += 1;
	view.setUint16(p, valueBytes.length, false);
	p += 2;
	buf.set(valueBytes, p);
	return buf;
}

export function encodeFrame(headers: Record<string, string>, payload: Uint8Array): Uint8Array {
	const headerChunks = Object.keys(headers).map(name => encodeStringHeader(name, headers[name]));
	const headerLen = headerChunks.reduce((s, c) => s + c.length, 0);
	const headerBytes = new Uint8Array(headerLen);
	let off = 0;
	for (const c of headerChunks) {
		headerBytes.set(c, off);
		off += c.length;
	}
	const total = 4 + 4 + 4 + headerLen + payload.length + 4;
	const out = new Uint8Array(total);
	const view = new DataView(out.buffer);
	view.setUint32(0, total, false);
	view.setUint32(4, headerLen, false);
	view.setUint32(8, crc32(out.subarray(0, 8)), false);
	out.set(headerBytes, 12);
	out.set(payload, 12 + headerLen);
	view.setUint32(total - 4, crc32(out.subarray(0, total - 4)), false);
	return out;
}

/** Convenience for an `event`-typed JSON frame (the shape Bedrock emits). */
export function eventFrame(eventType: string, payload: Record<string, unknown>): Uint8Array {
	return encodeFrame(
		{ ":message-type": "event", ":event-type": eventType },
		new TextEncoder().encode(JSON.stringify(payload)),
	);
}

export function streamFrom(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
	let i = 0;
	return new ReadableStream({
		pull(controller) {
			if (i < chunks.length) controller.enqueue(chunks[i++]);
			else controller.close();
		},
	});
}

/** Wraps frames in a 200 eventstream `Response`, as a streamed Bedrock body. */
export function eventStreamResponse(frames: Uint8Array[]): Response {
	return new Response(streamFrom(frames), {
		status: 200,
		headers: { "content-type": "application/vnd.amazon.eventstream" },
	});
}
