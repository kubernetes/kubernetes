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
	"compress/flate"
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	goruntime "runtime"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	example "k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage/value"
)

// rawFrame builds a frame with a valid checksum around an arbitrary algorithm
// byte and body, for cases the writer would never produce.
func rawFrame(alg byte, body []byte) []byte {
	f := make([]byte, headerSize, headerSize+len(body))
	copy(f, frameMagic[:])
	f[4] = alg
	return seal(append(f, body...))
}

// declaredDeflate builds a deflate frame with a valid checksum whose declared
// length n and stream can disagree, which deflateFrame never writes.
func declaredDeflate(n uint64, stream []byte) []byte {
	var lenBuf [binary.MaxVarintLen64]byte
	w := binary.PutUvarint(lenBuf[:], n)
	// The body is n as a uvarint, then the stream. [:w:w] caps the capacity so
	// append copies instead of writing into lenBuf.
	return rawFrame(algDeflate, append(lenBuf[:w:w], stream...))
}

// rawDeflate compresses b into a complete DEFLATE stream with no frame around it.
func rawDeflate(t testing.TB, b []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw, err := flate.NewWriter(&buf, deflateLevel)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := zw.Write(b); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// unterminatedDeflate flushes without closing, so the stream yields its data but
// never reaches a final block.
func unterminatedDeflate(t testing.TB, b []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw, err := flate.NewWriter(&buf, deflateLevel)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := zw.Write(b); err != nil {
		t.Fatal(err)
	}
	if err := zw.Flush(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// randomBytes never starts with 0x00.
func randomBytes(n int) []byte {
	b := make([]byte, n)
	if _, err := io.ReadFull(rand.Reader, b); err != nil {
		panic(err)
	}
	if n > 0 && b[0] == frameDiscriminator {
		b[0] = 0xff
	}
	return b
}

// assertReadError also checks what value.IsDataCorruption says, since that is
// what the unsafe-deletion layer acts on.
func assertReadError(t *testing.T, err, want error) {
	t.Helper()
	var re *readError
	if !errors.As(err, &re) || !errors.Is(err, want) {
		t.Fatalf("err = %v, want a readError matching %v", err, want)
	}
	if corrupt := value.IsDataCorruption(err); corrupt != errors.Is(want, errChecksum) {
		t.Fatalf("err = %v: value.IsDataCorruption = %v", err, corrupt)
	}
}

// TestFrameByteExact pins the literal at-rest bytes, so the format can't drift
// and strand data written by an earlier release.
func TestFrameByteExact(t *testing.T) {
	t.Run("stored", func(t *testing.T) {
		got := storedFrame([]byte("hello"))
		want := []byte{0x00, 0x6b, 0x38, 0x73, 0x00, 0x51, 0xbe, 0x28, 0x0e, 'h', 'e', 'l', 'l', 'o'}
		if !bytes.Equal(got, want) {
			t.Fatalf("stored frame changed:\n got % x\nwant % x", got, want)
		}
	})

	t.Run("deflate header", func(t *testing.T) {
		got, err := deflateFrame(bytes.Repeat([]byte("a"), 4096))
		if err != nil {
			t.Fatal(err)
		}
		if want := []byte{0x00, 0x6b, 0x38, 0x73, 0x01}; !bytes.HasPrefix(got, want) {
			t.Fatalf("deflate frame prefix changed: got % x, want % x", got[:min(len(got), 16)], want)
		}
		// 4096 is the uvarint 0x80 0x20, right after the checksum.
		if want := []byte{0x80, 0x20}; !bytes.Equal(got[headerSize:headerSize+2], want) {
			t.Fatalf("declared length moved: got % x, want % x", got[headerSize:headerSize+2], want)
		}
		if binary.LittleEndian.Uint32(got[crcOffset:headerSize]) != checksum(got) {
			t.Fatal("deflate frame carries a wrong checksum")
		}
	})

}

// TestDecodesPinnedFrames decodes frames this release wrote, checked in as
// literals. Never regenerate them: a reader that can't decode them would strand
// data at rest, whatever the writer does now.
func TestDecodesPinnedFrames(t *testing.T) {
	for _, tc := range []struct {
		name  string
		frame []byte
		want  []byte
		kind  frameKind
	}{
		{
			name:  "stored",
			frame: []byte{0x00, 0x6b, 0x38, 0x73, 0x00, 0x51, 0xbe, 0x28, 0x0e, 'h', 'e', 'l', 'l', 'o'},
			want:  []byte("hello"),
			kind:  framedStored,
		},
		{
			name: "deflate",
			frame: []byte{
				0x00, 0x6b, 0x38, 0x73, 0x01, 0x3c, 0x0c, 0x8d, 0x99, 0x80, 0x01, 0xbc,
				0xc7, 0xb1, 0x09, 0x00, 0x30, 0x08, 0x04, 0xc0, 0x55, 0xb2, 0x40, 0xc8,
				0x4e, 0xc1, 0xaf, 0x44, 0x14, 0x5e, 0x0b, 0xb7, 0x77, 0x0b, 0xcb, 0xd3,
				0xfa, 0xb8, 0x6c, 0x26, 0xec, 0x85, 0xcb, 0xd9, 0xf6, 0x04, 0x00, 0x00,
				0xff, 0xff,
			},
			want: bytes.Repeat([]byte("kube-system/pod "), 8),
			kind: framedDeflate,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, kind, err := unframe(context.Background(), tc.frame)
			if err != nil || kind != tc.kind || !bytes.Equal(out, tc.want) {
				t.Fatalf("kind %v, err %v, got %q", kind, err, out)
			}
		})
	}
}

func TestRoundTrip(t *testing.T) {
	lengths := []int{0, 1, 2, 7, 8, 9, 10, 16, 17, 128, 1023, 1024, 1025, 4096, 65536, 1 << 20, MaxPlaintextBytes}
	kinds := map[string]func(n int) []byte{
		"compressible":   func(n int) []byte { return bytes.Repeat([]byte("abcd"), n/4+1)[:n] },
		"incompressible": randomBytes,
		"protobuf-like": func(n int) []byte {
			b := append([]byte{0x6b, 0x38, 0x73, 0x00}, bytes.Repeat([]byte("metadata"), n)...)
			return b[:max(n, 4)]
		},
	}

	for kindName, gen := range kinds {
		for _, n := range lengths {
			t.Run(fmt.Sprintf("%s/%d", kindName, n), func(t *testing.T) {
				plaintext := gen(n)

				out, kind, err := unframe(context.Background(), storedFrame(plaintext))
				if err != nil || kind != framedStored || !bytes.Equal(out, plaintext) {
					t.Fatalf("stored: kind %v, err %v, equal %v", kind, err, bytes.Equal(out, plaintext))
				}

				if n == 0 {
					return // the writer never deflates an empty value; TestUnframeRejections covers a declared length of 0
				}
				framed, err := deflateFrame(plaintext)
				if err != nil {
					t.Fatal(err)
				}
				out, kind, err = unframe(context.Background(), framed)
				if err != nil || kind != framedDeflate || !bytes.Equal(out, plaintext) {
					t.Fatalf("deflate: kind %v, err %v, equal %v", kind, err, bytes.Equal(out, plaintext))
				}
			})
		}
	}
}

// TestLegacyPassthrough checks that a value without a frame, such as one written
// before compression was enabled, comes back untouched.
func TestLegacyPassthrough(t *testing.T) {
	cases := map[string][]byte{
		"protobuf":  append([]byte{0x6b, 0x38, 0x73, 0x00}, []byte("\n\x0fexample payload")...),
		"json":      []byte(`{"apiVersion":"v1","kind":"Pod"}`),
		"cbor":      {0xd9, 0xd9, 0xf7, 0xa1, 0x61, 'a', 0x01},
		"empty":     {},
		"encrypted": []byte("k8s:enc:aescbc:v1:key1:\x01\x02\x03"),
		// Shorter than a frame header but not a frame, so it must come back untouched.
		"single byte": {0x7b},
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			in := bytes.Clone(data)
			out, kind, err := unframe(context.Background(), in)
			if err != nil || kind != unframed || !bytes.Equal(out, data) {
				t.Fatalf("kind %v, err %v, got % x, want % x", kind, err, out, data)
			}
		})
	}
}

// TestUnframeRejections covers every reachable rejection branch and checks each
// lands in exactly one of the three error classes.
func TestUnframeRejections(t *testing.T) {
	valid := mustDeflate(t, bytes.Repeat([]byte("kubernetes"), 300))
	damaged := bytes.Clone(valid)
	damaged[len(damaged)/2] ^= 0x10

	badCRC := bytes.Clone(valid)
	badCRC[crcOffset] ^= 0xff

	wrongMagic := bytes.Clone(valid)
	wrongMagic[1] = 'x'
	seal(wrongMagic)

	// Behind a valid checksum a wrong magic is malformed whatever the algorithm
	// byte says, since every writer emits the same fixed prefix.
	wrongMagicFutureAlg := rawFrame(0x7f, nil)
	wrongMagicFutureAlg[1] = 'x'
	seal(wrongMagicFutureAlg)

	cases := []struct {
		name   string
		data   []byte
		want   error
		reason string
	}{
		{"discriminator only", []byte{0x00}, errChecksum, ""},
		{"truncated header", valid[:headerSize-1], errChecksum, ""},
		{"checksum mismatch", damaged, errChecksum, ""},
		{"checksum bytes damaged", badCRC, errChecksum, ""},

		{"wrong magic behind a valid checksum", wrongMagic, errFormat, "bad magic"},
		{"wrong magic wins over an unknown algorithm", wrongMagicFutureAlg, errFormat, "bad magic"},
		{"declared length zero", declaredDeflate(0, rawDeflate(t, nil)), errFormat, "invalid declared length"},
		{"declared length above the ceiling", declaredDeflate(MaxPlaintextBytes+1, rawDeflate(t, []byte("x"))), errFormat, "invalid declared length"},
		{"declared length absurd", declaredDeflate(1<<40, rawDeflate(t, []byte("x"))), errFormat, "invalid declared length"},
		{"truncated varint", rawFrame(algDeflate, []byte{0x80}), errFormat, "invalid declared length"},
		{"no deflate stream", declaredDeflate(16, nil), errFormat, "stream shorter than declared length"},
		// A DEFLATE block starts with 3 bits read from the low end of the byte: a
		// last-block flag, then a 2-bit block type. 0xff gives type 3, which the
		// format reserves, so the inflater rejects the stream at its first block.
		{"corrupt deflate stream", declaredDeflate(16, []byte{0xff, 0xff, 0xff, 0xff}), errFormat, "corrupt deflate stream"},
		{"stream shorter than declared", declaredDeflate(1024, rawDeflate(t, bytes.Repeat([]byte("a"), 512))), errFormat, "stream shorter than declared length"},
		{"stream longer than declared", declaredDeflate(512, rawDeflate(t, bytes.Repeat([]byte("a"), 1024))), errFormat, "stream longer than declared length"},
		{"stream without a final block", declaredDeflate(1024, unterminatedDeflate(t, bytes.Repeat([]byte("a"), 1024))), errFormat, "stream does not end at its declared length"},

		// The next algorithm ID a release would assign. With no body, a reader that
		// parsed the body before checking the algorithm would call it malformed.
		{"algorithm 0x02", rawFrame(0x02, nil), errUnsupported, "0x2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, _, err := unframe(context.Background(), tc.data)
			assertReadError(t, err, tc.want)
			if out != nil {
				t.Errorf("output must be nil on error, got %d bytes", len(out))
			}
			var re *readError
			if tc.reason != "" && (!errors.As(err, &re) || re.reason != tc.reason) {
				t.Errorf("reason of %v, want %q", err, tc.reason)
			}
		})
	}
}

// TestEverySingleBitFlipIsCorruption checks the guarantee KEP-3926 recovery
// needs: one flipped bit anywhere in a frame is reported as corruption. A flip
// after byte 0 fails the checksum. A flip in byte 0 makes the frame look unframed
// and skips the checksum, so the storage decoders for built-in resources and
// custom resources must reject it instead.
func TestEverySingleBitFlipIsCorruption(t *testing.T) {
	scheme := runtime.NewScheme()
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	universal := serializer.NewCodecFactory(scheme).UniversalDeserializer()
	pb := protobuf.NewSerializer(scheme, scheme)
	builtIn := recognizer.NewDecoder(pb, universal, runtime.NewBase64Serializer(nil, universal))

	ucbor := cbor.NewSerializer(unstructuredscheme.NewUnstructuredCreator(), unstructuredscheme.NewUnstructuredObjectTyper())
	customResource := recognizer.NewDecoder(ucbor, unstructured.UnstructuredJSONScheme)
	widget := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "example.com/v1",
		"kind":       "Widget",
		"metadata":   map[string]interface{}{"name": "w", "namespace": "default"},
	}}

	for name, tc := range map[string]struct {
		decoder runtime.Decoder
		data    []byte
	}{
		"built-in protobuf":    {builtIn, encode(t, pb, typicalPod())},
		"custom resource json": {customResource, encode(t, unstructured.UnstructuredJSONScheme, widget)},
		"custom resource cbor": {customResource, encode(t, ucbor, widget)},
	} {
		if _, _, err := tc.decoder.Decode(tc.data, nil, nil); err != nil {
			t.Fatalf("%s: precondition: the object must decode before it's damaged, got %v", name, err)
		}
		for frameName, frame := range map[string][]byte{"stored": storedFrame(tc.data), "deflate": mustDeflate(t, tc.data)} {
			t.Run(name+"/"+frameName, func(t *testing.T) {
				for bit := 0; bit < len(frame)*8; bit++ {
					flipped := bytes.Clone(frame)
					flipped[bit/8] ^= 1 << (bit % 8)

					out, kind, err := unframe(context.Background(), flipped)
					if bit < 8 {
						if err != nil || kind != unframed || !bytes.Equal(out, flipped) {
							t.Fatalf("bit %d: a flipped discriminator must pass through as unframed, got kind %v err %v", bit, kind, err)
						}
						if _, _, err := tc.decoder.Decode(out, nil, nil); err == nil {
							t.Fatalf("bit %d: the storage decoder accepted a frame with a flipped discriminator", bit)
						}
						continue
					}
					if !errors.Is(err, errChecksum) || !value.IsDataCorruption(err) {
						t.Fatalf("bit %d: err = %v, want a checksum error reported as data corruption", bit, err)
					}
				}
			})
		}
	}
}

// TestAppendedBytes covers trailing data. The checksum catches bytes appended
// as-is. If whoever appends them also recomputes the checksum, DEFLATE ignores
// them and the object still decodes correctly.
func TestAppendedBytes(t *testing.T) {
	plaintext := bytes.Repeat([]byte("a"), 64)
	frame := mustDeflate(t, plaintext)

	appended := append(bytes.Clone(frame), 0x99, 0x99)
	if _, _, err := unframe(context.Background(), appended); !errors.Is(err, errChecksum) {
		t.Fatalf("appended bytes without a new checksum: err = %v, want a checksum error", err)
	}

	resealed := seal(append(bytes.Clone(frame), 0x99, 0x99))
	out, kind, err := unframe(context.Background(), resealed)
	if err != nil || kind != framedDeflate || !bytes.Equal(out, plaintext) {
		t.Fatalf("appended bytes with a new checksum: kind %v, err %v", kind, err)
	}
}

func TestDeclaredLengthCheckedBeforeAllocation(t *testing.T) {
	for _, declared := range []uint64{MaxPlaintextBytes + 1, 1 << 30} {
		t.Run(fmt.Sprint(declared), func(t *testing.T) {
			bomb := declaredDeflate(declared, []byte{0x01, 0x02, 0x03})
			const iterations = 200

			// TotalAlloc counts the whole test binary, so don't mark tests in this package t.Parallel().
			var before, after goruntime.MemStats
			goruntime.GC()
			goruntime.ReadMemStats(&before)
			for range iterations {
				if _, _, err := unframe(context.Background(), bomb); !errors.Is(err, errFormat) {
					t.Fatalf("bomb must be rejected, got %v", err)
				}
			}
			goruntime.ReadMemStats(&after)

			if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 256*1024 {
				t.Fatalf("rejecting %d bombs allocated %d bytes; the declared length must be checked before allocating", iterations, allocated)
			}
		})
	}
}

// TestDecompressionBombBounded reads a frame that declares 1 KiB but whose stream
// inflates to 8 MiB. The reader must reject it soon after the first 1 KiB, so a
// small value in etcd can't cost megabytes of memory or CPU on every read.
func TestDecompressionBombBounded(t *testing.T) {
	// 8 MiB of zeros deflates to about 8 KiB, so the bomb is small at rest.
	stream := rawDeflate(t, make([]byte, 8<<20))
	if len(stream) > 64<<10 {
		t.Fatalf("precondition: expected a small stream, got %d bytes", len(stream))
	}
	bomb := declaredDeflate(1024, stream)
	const iterations = 50

	// TotalAlloc counts the whole test binary, so don't mark tests in this package t.Parallel().
	var before, after goruntime.MemStats
	goruntime.GC()
	goruntime.ReadMemStats(&before)
	for range iterations {
		if _, _, err := unframe(context.Background(), bomb); !errors.Is(err, errFormat) {
			t.Fatalf("a stream longer than declared must be rejected, got %v", err)
		}
	}
	goruntime.ReadMemStats(&after)

	// Buffering the output would cost 8 MiB a read; 1 MiB leaves room for the inflater's own buffers.
	if perIteration := (after.TotalAlloc - before.TotalAlloc) / iterations; perIteration > 1<<20 {
		t.Fatalf("allocated %d bytes per rejection; the stream must stop at its declared length", perIteration)
	}

	// A reader can also reject the bomb correctly but inflate all 8 MiB first to
	// find where it ends. That burns CPU on every read while allocating almost
	// nothing, so the allocation check above misses it. Counting the stream bytes
	// the inflater reads catches it: stopping just past 1 KiB needs a small prefix.
	src := &countingReader{r: bytes.NewReader(stream)}
	if _, err := inflateExact(src, 1024); !errors.Is(err, errFormat) {
		t.Fatalf("err = %v, want a format error", err)
	}
	if src.n > len(stream)/10 {
		t.Fatalf("the inflater consumed %d of %d stream bytes; it must stop just past the declared length", src.n, len(stream))
	}
}

// countingReader counts the bytes the inflater reads. ReadByte keeps flate from
// wrapping it in a bufio.Reader, whose read-ahead would inflate the count.
type countingReader struct {
	r *bytes.Reader
	n int
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}

func (c *countingReader) ReadByte() (byte, error) {
	b, err := c.r.ReadByte()
	if err == nil {
		c.n++
	}
	return b, err
}

// TestCancelWhileWaiting checks a read already parked on a full budget returns
// once its context is cancelled, instead of waiting for a slot it no longer wants.
func TestCancelWhileWaiting(t *testing.T) {
	// Take every inflation slot, so the read below has to wait for one.
	held := 0
	for saturated := false; !saturated; {
		select {
		case inflateBudget <- struct{}{}:
			held++
		default:
			saturated = true
		}
	}
	// Give the slots back when the test ends.
	defer func() {
		for i := 0; i < held; i++ {
			<-inflateBudget
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// Start the read in the background, where it blocks waiting for a slot.
	done := make(chan error, 1)
	go func() {
		_, _, err := unframe(ctx, mustDeflate(t, bytes.Repeat([]byte("a"), 2048)))
		done <- err
	}()
	// No answer within 50 ms means the read is waiting rather than failing early.
	// A slow scheduler can only make this pass, never fail.
	select {
	case err := <-done:
		t.Fatalf("a read must wait while the budget is full, got err %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	// Cancel while it waits. The read must return now, not when a slot frees up.
	cancel()
	select {
	case err := <-done:
		// The caller still sees context.Canceled, and it isn't reported as corruption.
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		assertReadError(t, err, errCanceled)
	// 5 s only guards against a hang; a cancelled read returns almost at once.
	case <-time.After(5 * time.Second):
		t.Fatal("a cancelled read kept waiting for a slot")
	}
}

func TestMaxInflightInflations(t *testing.T) {
	defer goruntime.GOMAXPROCS(goruntime.GOMAXPROCS(0))
	for _, tc := range []struct{ procs, want int }{{1, minInflations}, {8, 16}, {500, maxInflations}} {
		goruntime.GOMAXPROCS(tc.procs)
		if got := maxInflightInflations(); got != tc.want {
			t.Errorf("GOMAXPROCS=%d: got %d, want %d", tc.procs, got, tc.want)
		}
	}
}

// TestInflateBudgetContention drives a read through the wait and back out, so a
// leaked slot shows up here instead of as every read in the process blocking.
func TestInflateBudgetContention(t *testing.T) {
	// Take every inflation slot, so the read below has to wait for one.
	held := 0
	for saturated := false; !saturated; {
		select {
		case inflateBudget <- struct{}{}:
			held++
		default:
			saturated = true
		}
	}

	plaintext := bytes.Repeat([]byte("a"), 2048)
	frame := mustDeflate(t, plaintext)
	type result struct {
		out []byte
		err error
	}
	// Start the read in the background, where it blocks waiting for a slot.
	done := make(chan result, 1)
	go func() {
		out, _, err := unframe(context.Background(), frame)
		done <- result{out, err}
	}()
	// No answer within 50 ms means the read is waiting rather than failing early.
	select {
	case r := <-done:
		t.Fatalf("a read must wait while the budget is full, got err %v", r.err)
	case <-time.After(50 * time.Millisecond):
	}

	// Free one slot. The waiting read takes it and must finish with the right output.
	<-inflateBudget
	held--
	r := <-done
	if r.err != nil || !bytes.Equal(r.out, plaintext) {
		t.Fatalf("the waiting read: err %v", r.err)
	}
	// Give back the rest. If the read had kept its slot, the budget wouldn't be empty.
	for ; held > 0; held-- {
		<-inflateBudget
	}
	if n := len(inflateBudget); n != 0 {
		t.Fatalf("%d slots are still held after every read finished", n)
	}
}

func TestConcurrentFraming(t *testing.T) {
	const goroutines, iterations = 16, 50
	errCh := make(chan error, goroutines)
	for g := range goroutines {
		go func(g int) {
			for i := range iterations {
				plaintext := bytes.Repeat([]byte(fmt.Sprintf("goroutine-%d-iteration-%d-", g, i)), 64)
				framed, err := deflateFrame(plaintext)
				if err != nil {
					errCh <- err
					return
				}
				out, _, err := unframe(context.Background(), framed)
				if err != nil {
					errCh <- err
					return
				}
				if !bytes.Equal(out, plaintext) {
					errCh <- fmt.Errorf("goroutine %d iteration %d: pooled coder state leaked between calls", g, i)
					return
				}
			}
			errCh <- nil
		}(g)
	}
	for range goroutines {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}

// TestInflatedOutputDoesNotReuseInput checks an inflated value gets its own
// buffer, so a later change to the input can't change it. Stored and unframed
// values are slices of the input, the same way the identity provider returns them.
func TestInflatedOutputDoesNotReuseInput(t *testing.T) {
	framed := mustDeflate(t, bytes.Repeat([]byte("payload"), 512))
	out, _, err := unframe(context.Background(), framed)
	if err != nil {
		t.Fatal(err)
	}
	if &out[0] == &framed[0] {
		t.Fatal("decompressed output reuses the frame's memory")
	}
}

// TestErrorMessageOmitsRejectionReason checks a rejected frame's error message is
// just the sentinel's, with the reason kept only for the log. Clients see the
// message, and the reason would tell someone tampering with stored bytes which
// check they failed.
func TestErrorMessageOmitsRejectionReason(t *testing.T) {
	_, _, err := unframe(context.Background(), declaredDeflate(1<<40, nil))
	if err.Error() != errFormat.Error() {
		t.Fatalf("error message %q includes more than %q", err.Error(), errFormat.Error())
	}
	var re *readError
	if !errors.As(err, &re) || re.reason == "" {
		t.Fatalf("the reason must still be available for the log, got %#v", err)
	}
}

func mustDeflate(t testing.TB, b []byte) []byte {
	t.Helper()
	out, err := deflateFrame(b)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// storageEncoders returns every serializer the codec factory offers, by media
// type, so one added to the factory later is covered here without a code change.
// CBOR is opt-in, as it is when CBORServingAndStorage is on.
func storageEncoders(t *testing.T) map[string]runtime.Encoder {
	t.Helper()
	scheme := runtime.NewScheme()
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	codecs := serializer.NewCodecFactory(scheme, serializer.WithSerializer(cbor.NewSerializerInfo))
	out := map[string]runtime.Encoder{}
	for _, info := range codecs.SupportedMediaTypes() {
		out[info.MediaType] = info.Serializer
	}
	return out
}

// typicalPod is a small object of the shape storage serializes. The tests that
// use it depend on the serializer, not on the object's contents.
func typicalPod() runtime.Object {
	return &examplev1.Pod{
		TypeMeta:   metav1.TypeMeta{APIVersion: examplev1.SchemeGroupVersion.String(), Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{Name: "nginx", Namespace: "default"},
		Spec:       examplev1.PodSpec{Hostname: "nginx", NodeName: "node-1"},
		Status:     examplev1.PodStatus{Phase: examplev1.PodPhase("Running"), Message: "running"},
	}
}

func encode(t *testing.T, encoder runtime.Encoder, obj runtime.Object) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := encoder.Encode(obj, &buf); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("encoder produced no output")
	}
	return buf.Bytes()
}

// TestNoStorageSerializerEmitsFrameDiscriminator checks no storage serializer
// starts its output with 0x00. If one did, all its values would be framed even
// with compression off, and older releases couldn't read them.
func TestNoStorageSerializerEmitsFrameDiscriminator(t *testing.T) {
	// One encoded object doesn't prove a serializer never starts with 0x00, so each
	// media type here was checked by hand. A new one fails until someone checks it.
	checked := sets.New(runtime.ContentTypeJSON, runtime.ContentTypeYAML, runtime.ContentTypeProtobuf, runtime.ContentTypeCBOR)
	for mediaType, encoder := range storageEncoders(t) {
		t.Run(mediaType, func(t *testing.T) {
			if !checked.Has(mediaType) {
				t.Errorf("new serializer %s: check its output can never start with 0x00, then add it to checked", mediaType)
			}
			data := encode(t, encoder, typicalPod())
			if data[0] == frameDiscriminator {
				t.Fatalf("%s emitted a leading %#x: % x", mediaType, frameDiscriminator, data[:min(len(data), 32)])
			}
			out, kind, err := unframe(context.Background(), data)
			if err != nil || kind != unframed || !bytes.Equal(out, data) {
				t.Fatalf("serialized output must pass through: kind %v, err %v", kind, err)
			}
		})
	}
}

// TestUnstructuredJSONEmitsNoFrameDiscriminator covers custom resources stored as
// JSON. Their encoder, UnstructuredJSONScheme, isn't in the codec factory, so
// TestNoStorageSerializerEmitsFrameDiscriminator doesn't see it. Their CBOR
// encoder is the same one the factory provides.
func TestUnstructuredJSONEmitsNoFrameDiscriminator(t *testing.T) {
	obj := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "example.com/v1",
		"kind":       "Widget",
		"metadata":   map[string]interface{}{"name": "w", "namespace": "default"},
	}}
	data := encode(t, unstructured.UnstructuredJSONScheme, obj)
	if data[0] == frameDiscriminator {
		t.Fatalf("unstructured JSON emitted a leading %#x", frameDiscriminator)
	}
	out, kind, err := unframe(context.Background(), data)
	if err != nil || kind != unframed || !bytes.Equal(out, data) {
		t.Fatalf("unstructured JSON must pass through: kind %v, err %v", kind, err)
	}
}
