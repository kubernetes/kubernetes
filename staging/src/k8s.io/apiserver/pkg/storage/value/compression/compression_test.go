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
	"errors"
	"fmt"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
)

var (
	testGR       = gr("", "configmaps")
	testDataCtx  = value.DefaultContext("/registry/configmaps/default/test")
	compressible = bytes.Repeat([]byte("kubernetes-api-object-"), 200)
)

// recordingTransformer stands in for the encryption chain. It records what it's
// handed on write, so a test can see whether a frame reached it.
type recordingTransformer struct {
	delegate value.Transformer
	written  [][]byte
	stale    bool
	readErr  error
	writeErr error
}

func newRecording() *recordingTransformer {
	return &recordingTransformer{delegate: identity.NewEncryptCheckTransformer()}
}

func (r *recordingTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	if r.writeErr != nil {
		return nil, r.writeErr
	}
	r.written = append(r.written, bytes.Clone(data))
	return r.delegate.TransformToStorage(ctx, data, dataCtx)
}

func (r *recordingTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if r.readErr != nil {
		return nil, r.stale, r.readErr
	}
	out, _, err := r.delegate.TransformFromStorage(ctx, data, dataCtx)
	return out, r.stale, err
}

func (r *recordingTransformer) lastWrite() []byte {
	if len(r.written) == 0 {
		return nil
	}
	return r.written[len(r.written)-1]
}

// mustTransformToStorage calls tr.TransformToStorage and fails the test on an error.
func mustTransformToStorage(t testing.TB, tr value.Transformer, data []byte) []byte {
	t.Helper()
	stored, err := tr.TransformToStorage(context.Background(), data, testDataCtx)
	if err != nil {
		t.Fatal(err)
	}
	return stored
}

func policyFor(t testing.TB, resources ...string) *Policy {
	t.Helper()
	if len(resources) == 0 {
		return newPolicy()
	}
	return newPolicy(entry(deflate, resources...))
}

// TestRealEncryptionChainOrdering uses real AES-GCM to check that the frame sits
// inside the ciphertext, the envelope prefix is unchanged and the stored value
// shrinks. Both forms read back with the policy on or off.
func TestRealEncryptionChainOrdering(t *testing.T) {
	aes, _, err := aestransformer.NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		t.Fatal(err)
	}
	const prefix = "k8s:enc:aesgcm:v1:key1:"
	encryption := value.NewPrefixTransformers(nil, value.PrefixTransformer{Prefix: []byte(prefix), Transformer: aes})

	compressed := WithCompression(encryption, policyFor(t, "configmaps"), testGR)
	plain := WithCompression(encryption, nil, testGR)

	withCompression := mustTransformToStorage(t, compressed, compressible)
	withoutCompression := mustTransformToStorage(t, plain, compressible)
	for name, stored := range map[string][]byte{"compressed": withCompression, "uncompressed": withoutCompression} {
		if !bytes.HasPrefix(stored, []byte(prefix)) {
			t.Fatalf("%s: envelope prefix changed: % x", name, stored[:min(len(stored), 32)])
		}
	}
	if len(withCompression) >= len(withoutCompression) {
		t.Errorf("compressed value (%d bytes) should be smaller than uncompressed (%d)", len(withCompression), len(withoutCompression))
	}

	inner, _, err := aes.TransformFromStorage(context.Background(), withCompression[len(prefix):], testDataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(inner, frameMagic[:]) {
		t.Fatalf("the frame must live inside the ciphertext, got % x", inner[:min(len(inner), 16)])
	}

	for readerName, reader := range map[string]value.Transformer{"policy on": compressed, "policy off": plain} {
		for storedName, stored := range map[string][]byte{"compressed": withCompression, "uncompressed": withoutCompression} {
			out, _, err := reader.TransformFromStorage(context.Background(), stored, testDataCtx)
			if err != nil || !bytes.Equal(out, compressible) {
				t.Fatalf("%s reading %s value: err %v", readerName, storedName, err)
			}
		}
	}
}

// TestDefaultInertness protects tests elsewhere that read raw etcd bytes. With no
// policy, an empty one, or one that doesn't select the resource, the decorator
// changes nothing in either direction.
func TestDefaultInertness(t *testing.T) {
	for _, policy := range []*Policy{nil, policyFor(t), policyFor(t, "pods")} {
		rec := newRecording()
		tr := WithCompression(rec, policy, testGR)
		for _, data := range [][]byte{compressible, []byte("{}"), {}, append([]byte{0x6b, 0x38, 0x73, 0x00}, compressible...)} {
			stored := mustTransformToStorage(t, tr, data)
			if !bytes.Equal(rec.lastWrite(), data) {
				t.Fatalf("inert decorator changed the bytes handed to the delegate: % x", rec.lastWrite()[:min(len(rec.lastWrite()), 16)])
			}
			out, stale, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx)
			if err != nil || stale || !bytes.Equal(out, data) {
				t.Fatalf("inert decorator on read: stale %v, err %v", stale, err)
			}
		}
	}
}

// TestSizeBandRoundTrip checks both edges of the band the policy frames, DefaultMinSize
// to MaxPlaintextBytes, for compressible and incompressible values, and that each
// reads back unchanged. The policy frames nothing above the ceiling, so after
// compression is turned off, every compressed value still fits etcd's request
// limit when it's rewritten uncompressed.
func TestSizeBandRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name      string
		size      int
		wantFrame bool
	}{
		{"below threshold", DefaultMinSize - 1, false},
		{"at threshold", DefaultMinSize, true},
		{"inside", 64 << 10, true},
		{"at ceiling", MaxPlaintextBytes, true},
		{"above ceiling", MaxPlaintextBytes + 1, false},
	} {
		for kind, gen := range map[string]func(int) []byte{
			"compressible":   func(n int) []byte { return bytes.Repeat([]byte("a"), n) },
			"incompressible": randomBytes,
		} {
			t.Run(tc.name+"/"+kind, func(t *testing.T) {
				data := gen(tc.size)
				rec := newRecording()
				tr := WithCompression(rec, policyFor(t, "configmaps"), testGR)
				stored := mustTransformToStorage(t, tr, data)
				if got := isFramed(rec.lastWrite()); got != tc.wantFrame {
					t.Fatalf("framed = %v, want %v", got, tc.wantFrame)
				}
				out, stale, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx)
				if err != nil || stale || !bytes.Equal(out, data) {
					t.Fatalf("round trip: stale %v, err %v", stale, err)
				}
			})
		}
	}
}

// TestNeverExpand checks that an incompressible value in the band goes into a
// stored frame, which adds only the 9-byte header. It can't be written bare
// instead, or every read would report it stale.
func TestNeverExpand(t *testing.T) {
	rec := newRecording()
	tr := WithCompression(rec, policyFor(t, "configmaps"), testGR)
	data := randomBytes(4096)
	mustTransformToStorage(t, tr, data)
	written := rec.lastWrite()
	if !isFramed(written) || written[4] != algStored || len(written) != headerSize+len(data) {
		t.Fatalf("want a %d-byte stored frame, got %d bytes with algorithm %#x", headerSize+len(data), len(written), written[4])
	}
}

func TestStalenessTruthTable(t *testing.T) {
	big := bytes.Repeat([]byte("a"), 4096)
	small := bytes.Repeat([]byte("a"), DefaultMinSize-1)
	incompressible := randomBytes(4096)

	for _, tc := range []struct {
		name      string
		policyOn  bool
		onDisk    []byte
		plaintext []byte
		wantStale bool
	}{
		{"unframed, policy off, big", false, big, big, false},
		{"unframed, policy on, big", true, big, big, true},
		{"unframed, policy on, small", true, small, small, false},
		{"deflate, policy on, big", true, mustDeflate(t, big), big, false},
		{"deflate, policy off, big", false, mustDeflate(t, big), big, true},
		{"stored, policy on, big", true, storedFrame(incompressible), incompressible, false},
		{"stored, policy off, big", false, storedFrame(incompressible), incompressible, true},
		{"stored, policy on, small", true, storedFrame(small), small, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var resources []string
			if tc.policyOn {
				resources = []string{"configmaps"}
			}
			tr := WithCompression(newRecording(), policyFor(t, resources...), testGR)
			out, stale, err := tr.TransformFromStorage(context.Background(), tc.onDisk, testDataCtx)
			if err != nil || !bytes.Equal(out, tc.plaintext) || stale != tc.wantStale {
				t.Fatalf("stale %v (want %v), err %v", stale, tc.wantStale, err)
			}
		})
	}
}

// TestStalenessCombinesWithDelegate checks that neither layer masks the other,
// for every stored form and with the policy on and off: a key rotation must still
// migrate a correctly framed value, and a framing change must still migrate a
// value the encryption layer is happy with.
func TestStalenessCombinesWithDelegate(t *testing.T) {
	big := randomBytes(4096)
	forms := map[string][]byte{"bare": big, "deflate": mustDeflate(t, big), "stored": storedFrame(big)}
	for formName, onDisk := range forms {
		for _, policyOn := range []bool{false, true} {
			for _, delegateStale := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/policy=%v/delegate=%v", formName, policyOn, delegateStale), func(t *testing.T) {
					var resources []string
					if policyOn {
						resources = []string{"configmaps"}
					}
					rec := newRecording()
					rec.stale = delegateStale
					tr := WithCompression(rec, policyFor(t, resources...), testGR)
					_, stale, err := tr.TransformFromStorage(context.Background(), onDisk, testDataCtx)
					if err != nil {
						t.Fatal(err)
					}
					// Framing alone makes it stale when it's framed with the policy off, or bare with it on.
					framingStale := isFramed(onDisk) != policyOn
					if want := delegateStale || framingStale; stale != want {
						t.Fatalf("stale = %v, want %v", stale, want)
					}
				})
			}
		}
	}
}

// TestKeyRotationUnderCompression uses real AES-GCM keys. A value written under
// the old primary key reads back stale once a new key is primary, even though its
// framing is right, and one rewrite re-encrypts it and keeps it compressed.
func TestKeyRotationUnderCompression(t *testing.T) {
	key1, _, err := aestransformer.NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		t.Fatal(err)
	}
	key2, _, err := aestransformer.NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		t.Fatal(err)
	}
	k1 := value.PrefixTransformer{Prefix: []byte("k8s:enc:aesgcm:v1:key1:"), Transformer: key1}
	k2 := value.PrefixTransformer{Prefix: []byte("k8s:enc:aesgcm:v1:key2:"), Transformer: key2}
	policy := policyFor(t, "configmaps")
	before := WithCompression(value.NewPrefixTransformers(nil, k1, k2), policy, testGR)
	after := WithCompression(value.NewPrefixTransformers(nil, k2, k1), policy, testGR)

	stored := mustTransformToStorage(t, before, compressible)
	out, stale, err := after.TransformFromStorage(context.Background(), stored, testDataCtx)
	if err != nil || !bytes.Equal(out, compressible) || !stale {
		t.Fatalf("after rotation: stale %v, err %v", stale, err)
	}

	rewritten := mustTransformToStorage(t, after, out)
	if !bytes.HasPrefix(rewritten, k2.Prefix) {
		t.Fatalf("the rewrite must use the new primary key, got % x", rewritten[:min(len(rewritten), 24)])
	}
	inner, _, err := key2.TransformFromStorage(context.Background(), rewritten[len(k2.Prefix):], testDataCtx)
	if err != nil || !isFramed(inner) || inner[4] != algDeflate {
		t.Fatalf("the rewrite must stay compressed: err %v", err)
	}
	if _, stale, err := after.TransformFromStorage(context.Background(), rewritten, testDataCtx); err != nil || stale {
		t.Fatalf("after one rewrite: stale %v, err %v", stale, err)
	}
}

// TestCrossConfigurationReads writes under one configuration and reads under
// another, across the threshold and ceiling boundaries. Every read returns the
// value, and it's stale exactly when the reader would store it differently.
func TestCrossConfigurationReads(t *testing.T) {
	configs := map[string]*Policy{
		"nil":          nil,
		"None":         newPolicy(entry(none, "configmaps")),
		"default":      policyFor(t, "configmaps"),
		"minSize 64":   newPolicy(withMinSize(entry(deflate, "configmaps"), 64)),
		"minSize 64Ki": newPolicy(withMinSize(entry(deflate, "configmaps"), 64<<10)),
	}
	sizes := []int{0, 63, 64, 65, 1023, 1024, 1025, 64<<10 - 1, 64 << 10, MaxPlaintextBytes, MaxPlaintextBytes + 1}
	kinds := map[string]func(int) []byte{
		"compressible":   func(n int) []byte { return bytes.Repeat([]byte("abcd"), n/4+1)[:n] },
		"incompressible": randomBytes,
		"escaped": func(n int) []byte {
			if n == 0 {
				return nil
			}
			return append([]byte{0x00}, bytes.Repeat([]byte("abcd"), n/4+1)[:n-1]...)
		},
	}

	for kindName, gen := range kinds {
		for _, size := range sizes {
			data := gen(size)
			// An escaped value needs its leading 0x00, so it can't be empty.
			if data == nil && kindName == "escaped" {
				continue
			}
			for writerName, writerPolicy := range configs {
				rec := newRecording()
				writer := WithCompression(rec, writerPolicy, testGR).(*transformer)
				stored, err := writer.TransformToStorage(context.Background(), data, testDataCtx)
				if err != nil {
					t.Fatalf("%s/%d written under %s: %v", kindName, size, writerName, err)
				}
				// If the writer and willFrame disagreed, every read would ask for a rewrite.
				framed := isFramed(rec.lastWrite())
				if framed != writer.willFrame(data) {
					t.Fatalf("%s/%d written under %s: framed %v", kindName, size, writerName, framed)
				}
				for readerName, readerPolicy := range configs {
					reader := WithCompression(newRecording(), readerPolicy, testGR).(*transformer)
					out, stale, err := reader.TransformFromStorage(context.Background(), stored, testDataCtx)
					if err != nil || !bytes.Equal(out, data) {
						t.Fatalf("%s/%d written under %s, read under %s: err %v", kindName, size, writerName, readerName, err)
					}
					// Stale exactly when the stored form isn't what the reader would write.
					if want := framed != reader.willFrame(data); stale != want {
						t.Fatalf("%s/%d written under %s, read under %s: stale %v, want %v", kindName, size, writerName, readerName, stale, want)
					}
				}
			}
		}
	}
}

// TestStalenessConvergesInOneRewrite checks there's no rewrite loop: once
// rewritten, a value is never reported stale again, in either direction.
func TestStalenessConvergesInOneRewrite(t *testing.T) {
	for name, data := range map[string][]byte{"compressible": bytes.Repeat([]byte("a"), 4096), "incompressible": randomBytes(4096)} {
		for _, policyOn := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/policy=%v", name, policyOn), func(t *testing.T) {
				// Start from the form the opposite setting wrote, so the first read must ask for a rewrite.
				var resources []string
				onDisk := mustDeflate(t, data)
				if policyOn {
					resources, onDisk = []string{"configmaps"}, data
				}
				tr := WithCompression(newRecording(), policyFor(t, resources...), testGR)

				if _, stale, err := tr.TransformFromStorage(context.Background(), onDisk, testDataCtx); err != nil || !stale {
					t.Fatalf("the starting form must be stale: stale %v, err %v", stale, err)
				}
				rewritten := mustTransformToStorage(t, tr, data)
				for i := range 2 {
					out, stale, err := tr.TransformFromStorage(context.Background(), rewritten, testDataCtx)
					if err != nil || stale || !bytes.Equal(out, data) {
						t.Fatalf("read %d after one rewrite: stale %v, err %v", i+1, stale, err)
					}
				}
			})
		}
	}
}

// TestMinSizeGatesFramingAndStaleness checks a 512-byte value is framed under a
// 256-byte minSize but not the default 1 KiB, and that each reader judges a
// stored value against its own minSize when deciding whether it's stale.
func TestMinSizeGatesFramingAndStaleness(t *testing.T) {
	low := WithCompression(newRecording(), newPolicy(withMinSize(entry(deflate, "configmaps"), 256)), testGR)
	def := WithCompression(newRecording(), policyFor(t, "configmaps"), testGR)
	payload := bytes.Repeat([]byte("kube-system/pod-abcdef 0123456789 "), 16)[:512]

	stored := mustTransformToStorage(t, low, payload)
	if !isFramed(stored) {
		t.Fatal("512 bytes must be framed under a 256-byte threshold")
	}
	if bare, err := def.TransformToStorage(context.Background(), payload, testDataCtx); err != nil || isFramed(bare) {
		t.Fatalf("512 bytes must stay bare under the default threshold: err %v", err)
	}

	// The threshold only gates writes. A frame written under a lower threshold
	// reads back anywhere, and staleness follows each reader's own threshold.
	out, stale, err := def.TransformFromStorage(context.Background(), stored, testDataCtx)
	if err != nil || !bytes.Equal(out, payload) || !stale {
		t.Fatalf("default reader: stale %v, err %v", stale, err)
	}
	if _, stale, err := low.TransformFromStorage(context.Background(), stored, testDataCtx); err != nil || stale {
		t.Fatalf("low reader on its own frame: stale %v, err %v", stale, err)
	}
	if _, stale, err := low.TransformFromStorage(context.Background(), payload, testDataCtx); err != nil || !stale {
		t.Fatalf("low reader on a bare value above its threshold: stale %v, err %v", stale, err)
	}
}

// TestEscapedValues checks a value that already starts with 0x00. With the policy
// on or off, the writer accepts it and wraps it in a stored frame, and every read
// returns it unchanged and not stale.
func TestEscapedValues(t *testing.T) {
	ambiguous := append([]byte{0x00}, bytes.Repeat([]byte("x"), 64)...)
	for _, resources := range [][]string{nil, {"configmaps"}} {
		rec := newRecording()
		tr := WithCompression(rec, policyFor(t, resources...), testGR)

		stored, err := tr.TransformToStorage(context.Background(), ambiguous, testDataCtx)
		if err != nil {
			t.Fatalf("policy %v: escape, don't reject: %v", resources, err)
		}
		if w := rec.lastWrite(); !isFramed(w) || w[4] != algStored {
			t.Fatalf("policy %v: want a stored frame, got % x", resources, w[:min(len(w), 12)])
		}
		for i := range 3 {
			out, stale, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx)
			if err != nil || stale || !bytes.Equal(out, ambiguous) {
				t.Fatalf("policy %v read %d: stale %v, err %v", resources, i+1, stale, err)
			}
		}
	}
}

// TestDelegateErrorsPropagate checks that when encryption fails on a read or a
// write, the caller gets that same error back from the compression layer.
func TestDelegateErrorsPropagate(t *testing.T) {
	readErr, writeErr := errors.New("decryption failed"), errors.New("encryption failed")

	rec := newRecording()
	rec.readErr = readErr
	tr := WithCompression(rec, policyFor(t, "configmaps"), testGR)
	if _, _, err := tr.TransformFromStorage(context.Background(), []byte("anything"), testDataCtx); !errors.Is(err, readErr) {
		t.Fatalf("read: err = %v, want %v", err, readErr)
	}

	rec = newRecording()
	rec.writeErr = writeErr
	tr = WithCompression(rec, policyFor(t, "configmaps"), testGR)
	if _, err := tr.TransformToStorage(context.Background(), compressible, testDataCtx); !errors.Is(err, writeErr) {
		t.Fatalf("write: err = %v, want %v", err, writeErr)
	}
}

// TestReadSurfacesErrorClasses checks that each read error class reaches the
// caller through the decorator, and that value.IsDataCorruption reports only the
// checksum failure as corruption.
func TestReadSurfacesErrorClasses(t *testing.T) {
	damaged := mustDeflate(t, compressible)
	damaged[headerSize+3] ^= 0x01

	tr := WithCompression(newRecording(), nil, testGR)
	for name, tc := range map[string]struct {
		stored []byte
		want   error
	}{
		"checksum mismatch": {damaged, errChecksum},
		"malformed frame":   {declaredDeflate(16, nil), errFormat},
		"unknown algorithm": {rawFrame(0xfe, nil), errUnsupported},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, err := tr.TransformFromStorage(context.Background(), tc.stored, testDataCtx)
			assertReadError(t, err, tc.want)
		})
	}
}

// TestReadIsNeverGatedByPolicy checks a framed value reads back under any
// policy. If reading depended on the policy, dropping a flag would strand data.
func TestReadIsNeverGatedByPolicy(t *testing.T) {
	stored := mustTransformToStorage(t, WithCompression(newRecording(), policyFor(t, "*.*"), testGR), compressible)
	for name, policy := range map[string]*Policy{
		"nil":                nil,
		"empty":              policyFor(t),
		"different resource": policyFor(t, "pods"),
		"same resource":      policyFor(t, "configmaps"),
		"None":               newPolicy(entry(none, "*.*")),
	} {
		t.Run(name, func(t *testing.T) {
			out, _, err := WithCompression(newRecording(), policy, testGR).TransformFromStorage(context.Background(), stored, testDataCtx)
			if err != nil || !bytes.Equal(out, compressible) {
				t.Fatalf("a framed value must read back under any policy: %v", err)
			}
		})
	}
}

func TestSecretsNeedExplicitNaming(t *testing.T) {
	for _, tc := range []struct {
		name       string
		entries    []entryConfig
		wantFramed bool
	}{
		{"everything wildcard", []entryConfig{entry(deflate, "*.*")}, false},
		{"core group wildcard", []entryConfig{entry(deflate, "*.")}, false},
		{"named", []entryConfig{entry(deflate, "secrets")}, true},
		{"named alongside a wildcard", []entryConfig{entry(deflate, "secrets"), entry(deflate, "*.*")}, true},
		{"named with None", []entryConfig{entry(none, "secrets")}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := newRecording()
			tr := WithCompression(rec, newPolicy(tc.entries...), SecretsGroupResource)
			mustTransformToStorage(t, tr, compressible)
			if got := isFramed(rec.lastWrite()); got != tc.wantFramed {
				t.Fatalf("framed = %v, want %v", got, tc.wantFramed)
			}
		})
	}
}

func TestConcurrentTransformer(t *testing.T) {
	// One transformer shared by every goroutine, as storage shares it across requests.
	tr := WithCompression(identity.NewEncryptCheckTransformer(), policyFor(t, "configmaps"), testGR)
	const goroutines = 12
	errCh := make(chan error, goroutines)
	for g := range goroutines {
		go func(g int) {
			for i := range 25 {
				// Each value is unique, so a mix-up between goroutines shows as a mismatch.
				data := bytes.Repeat([]byte(fmt.Sprintf("g%d-i%d-", g, i)), 300)
				// t.Fatal can't be called off the test goroutine, so errors go through errCh.
				stored, err := tr.TransformToStorage(context.Background(), data, testDataCtx)
				if err != nil {
					errCh <- err
					return
				}
				out, _, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx)
				if err != nil {
					errCh <- err
					return
				}
				if !bytes.Equal(out, data) {
					errCh <- fmt.Errorf("g%d i%d: data mismatch", g, i)
					return
				}
			}
			errCh <- nil
		}(g)
	}
	// Every goroutine sends exactly one result, nil when all its round trips passed.
	for range goroutines {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}
