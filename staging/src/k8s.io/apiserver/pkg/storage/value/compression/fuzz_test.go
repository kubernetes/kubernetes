/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package compression

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math/rand/v2"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
)

// FuzzUnframe treats the reader's input as hostile, since providers that don't
// authenticate on read let anyone who can write to etcd choose these bytes.
func FuzzUnframe(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x00}) // just the discriminator, shorter than a header
	f.Add(storedFrame(nil))
	f.Add(storedFrame([]byte("plaintext")))
	f.Add(mustDeflate(f, bytes.Repeat([]byte("kubernetes"), 200)))
	f.Add(mustDeflate(f, []byte("small")))

	f.Add([]byte(`{"apiVersion":"v1"}`))
	f.Add([]byte{0x6b, 0x38, 0x73, 0x00, 0x0a, 0x01, 0x41}) // a bare protobuf value: k8s\x00 magic and a tiny body
	f.Add([]byte{0xd9, 0xd9, 0xf7, 0xa0})                   // a bare CBOR value: self-describe tag and an empty map

	f.Add(rawFrame(0xff, nil))                                                                      // a frame with unknown algorithm 0xff
	f.Add(rawFrame(algDeflate, nil))                                                                // a deflate frame with no declared length and no stream
	f.Add(rawFrame(algDeflate, []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f})) // a declared length that overflows uint64
	f.Add(rawFrame(algDeflate, []byte{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80})) // a declared length whose varint is cut off
	f.Add(declaredDeflate(32, rawDeflate(f, make([]byte, 4<<20))))

	f.Fuzz(func(t *testing.T, data []byte) {
		// unframe must not write to data, which still belongs to the caller.
		original := bytes.Clone(data)
		out, kind, err := unframe(context.Background(), data)

		if !bytes.Equal(data, original) {
			t.Fatal("unframe mutated its input")
		}
		// Recompute the checksum so the checks below don't trust unframe's own.
		intact := len(original) >= headerSize && binary.LittleEndian.Uint32(original[crcOffset:headerSize]) == checksum(original)

		if err != nil {
			// Metrics and corruption handling tell errors apart by sentinel, so an
			// unknown error would slip past both.
			if !errors.Is(err, errChecksum) && !errors.Is(err, errFormat) && !errors.Is(err, errUnsupported) {
				t.Fatalf("error %v must be a checksum, format or unsupported-algorithm readError", err)
			}
			// A matching checksum means the bytes weren't damaged in storage, so no
			// other problem counts as corruption.
			if intact && value.IsDataCorruption(err) {
				t.Fatal("a frame with a valid checksum must never be reported as corruption")
			}
			if out != nil {
				t.Fatalf("output must be nil on error, got %d bytes", len(out))
			}
			return
		}

		// On success, the output must match what the reported kind promises.
		switch kind {
		case unframed:
			if !bytes.Equal(out, original) || isFramed(original) {
				t.Fatal("only a value without the discriminator may pass through, and only verbatim")
			}
		case framedStored:
			if !intact || !bytes.Equal(out, original[headerSize:]) {
				t.Fatal("a stored frame must be intact and return its body verbatim")
			}
		case framedDeflate:
			// Parse the declared length here too, rather than trust unframe's parse.
			declared, w := binary.Uvarint(original[headerSize:])
			if !intact || w <= 0 || uint64(len(out)) != declared || len(out) > MaxPlaintextBytes {
				t.Fatalf("a deflate frame must be intact and inflate to its declared length %d within the ceiling, got %d", declared, len(out))
			}
		default:
			t.Fatalf("unknown frame kind %v", kind)
		}
	})
}

// FuzzTransformerRoundTrip checks that whatever the writer produces, the reader
// recovers exactly and doesn't report it stale.
func FuzzTransformerRoundTrip(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte("small"))
	f.Add([]byte{0x00, 0x6b, 0x38, 0x73, 0x01}) // plaintext that starts like a deflate frame header
	f.Add(bytes.Repeat([]byte("compressible"), 200))
	f.Add([]byte{0x6b, 0x38, 0x73, 0x00, 0x0a, 0x01, 0x41}) // a bare protobuf value: k8s\x00 magic and a tiny body

	// Random bytes don't compress, so above the threshold the writer falls back to
	// a stored frame. A fixed seed keeps the input the same on every run.
	incompressible := make([]byte, 2*DefaultMinSize)
	_, _ = rand.NewChaCha8([32]byte{}).Read(incompressible)
	f.Add(incompressible)

	policies := []*Policy{nil, newPolicy(entry(deflate, "configmaps")), newPolicy(withMinSize(entry(deflate, "configmaps"), MinConfigurableMinSize))}
	f.Fuzz(func(t *testing.T, data []byte) {
		for _, p := range policies {
			tr := WithCompression(newRecording(), p, testGR)
			stored, err := tr.TransformToStorage(context.Background(), data, testDataCtx)
			if err != nil {
				t.Fatalf("a write must never fail: %v", err)
			}
			out, stale, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx)
			if err != nil {
				t.Fatalf("a value just written must be readable: %v", err)
			}
			if !bytes.Equal(out, data) {
				t.Fatal("round trip lost data")
			}
			if stale {
				t.Fatal("a value written under a policy must not be stale under the same policy")
			}
		}
	})
}
