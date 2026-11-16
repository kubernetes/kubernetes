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

package transformation

import (
	"bytes"
	"compress/flate"
	"context"
	"encoding/binary"
	"encoding/hex"
	"hash/crc32"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/compression"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	// A planted unreadable value stalls the secrets watch cache. A LIST that the
	// cache answers then waits for it to catch up before falling back to etcd, or
	// gets a 429 if the cache has gone unready. With that cache off, every read in
	// these tests goes to etcd.
	noSecretsWatchCache = "--watch-cache-sizes=secrets#0"

	formatErrorsMetric         = "apiserver_storage_compression_format_errors_total"
	unsupportedAlgorithmMetric = "apiserver_storage_compression_unsupported_algorithm_total"

	pinnedSecretNamespace = "storage-compression"
	pinnedSecretName      = "pinned-deflate-frame"
	// pinnedFrameHex is a Deflate frame written by compression.WithCompression in the
	// release that introduced frames. It holds the protobuf-encoded v1 Secret
	// storage-compression/pinned-deflate-frame (type Opaque, data.note "compress me "
	// repeated 8 times).
	//
	// NEVER regenerate it. It anchors the KEP-6264 skew rule: every later release must
	// still read what that release wrote.
	pinnedFrameHex = "006b387301b35acdcac401cab62866e0e2e1622a3314620b4e4d2e4a2d115acd" +
		"c865ce2552909997979aa29b929a96935892aa9b5694989b2ac420255c5c925f" +
		"94989eaa9b9c9f5b50945a5c9c999fa7c4a0c560c460c1e0c42094c1c592975f" +
		"922a94909c9f5b50945a5cac909baa00534a4db6149b7f41626169aa148312036000"
)

var (
	secretsGR   = schema.GroupResource{Resource: "secrets"}
	compressAll = compression.Rule{Algorithm: compression.AlgorithmDeflate, MinSize: 1}
	castagnoli  = crc32.MakeTable(crc32.Castagnoli)
)

// sealFrame writes the CRC32C that compression's format.go defines into bytes
// 5-8, so a hand-built frame passes its checksum whatever else is wrong with it.
func sealFrame(frame []byte) []byte {
	crc := crc32.Update(crc32.Checksum(frame[:5], castagnoli), castagnoli, frame[9:])
	binary.LittleEndian.PutUint32(frame[5:9], crc)
	return frame
}

// lyingLengthFrame builds a Deflate frame, following the layout in format.go,
// whose header declares one byte more than its stream inflates to.
func lyingLengthFrame(t *testing.T, plaintext []byte) []byte {
	t.Helper()
	var stream bytes.Buffer
	zw, err := flate.NewWriter(&stream, flate.BestSpeed)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := zw.Write(plaintext); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	frame := binary.AppendUvarint([]byte{0x00, 'k', '8', 's', 0x01, 0, 0, 0, 0}, uint64(len(plaintext)+1))
	return sealFrame(append(frame, stream.Bytes()...))
}

// aesCBCTransformer encrypts and decrypts with the key and prefix that the
// aescbc provider in the test's encryption config uses.
func (e *transformTest) aesCBCTransformer(t *testing.T) value.Transformer {
	t.Helper()
	config, err := e.getEncryptionConfig()
	if err != nil {
		t.Fatal(err)
	}
	block, err := newAESCipher(config.AESCBC.Keys[0].Secret)
	if err != nil {
		t.Fatal(err)
	}
	return value.NewPrefixTransformers(nil, value.PrefixTransformer{Prefix: []byte(aesCBCPrefix), Transformer: aestransformer.NewCBCTransformer(block)})
}

// rawSecret returns the value etcd holds for a secret, or nil if there's none.
func (e *transformTest) rawSecret(t *testing.T, namespace, name string) []byte {
	t.Helper()
	resp, err := e.readRawRecordFromETCD(e.getETCDPathForResource(e.storageConfig.Prefix, "", "secrets", name, namespace))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Kvs) == 0 {
		return nil
	}
	return resp.Kvs[0].Value
}

func (e *transformTest) plantSecret(t *testing.T, namespace, name string, v []byte) {
	t.Helper()
	if _, err := e.writeRawRecordToETCD(e.getETCDPathForResource(e.storageConfig.Prefix, "", "secrets", name, namespace), v); err != nil {
		t.Fatal(err)
	}
}

// secretsCounter returns a storage compression counter's value for secrets, or 0
// before its first sample. The test apiserver runs in this process and records
// into legacyregistry.
func secretsCounter(t *testing.T, name string) float64 {
	t.Helper()
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, m := range family.GetMetric() {
			for _, label := range m.GetLabel() {
				if label.GetName() == "resource" && label.GetValue() == secretsGR.String() {
					return m.GetCounter().GetValue()
				}
			}
		}
	}
	return 0
}

// TestIntactFramesAreNotCorruptObjects plants intact frames that can't be read: an
// unknown algorithm id, and a Deflate frame with a lying length. GET, LIST and
// unsafe deletion fail with InternalError, not StoreReadError; etcd keeps the value.
func TestIntactFramesAreNotCorruptObjects(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, true)

	tests := []struct {
		namespace    string
		frame        func(t *testing.T, plaintext []byte) []byte
		wantCounter  string
		quietCounter string
	}{
		{
			namespace: "unknown-algorithm",
			frame: func(*testing.T, []byte) []byte {
				return sealFrame([]byte{0x00, 'k', '8', 's', 0xfe, 0, 0, 0, 0})
			},
			wantCounter:  unsupportedAlgorithmMetric,
			quietCounter: formatErrorsMetric,
		},
		{
			namespace:    "lying-declared-length",
			frame:        lyingLengthFrame,
			wantCounter:  formatErrorsMetric,
			quietCounter: unsupportedAlgorithmMetric,
		},
	}
	for _, tc := range tests {
		t.Run(tc.namespace, func(t *testing.T) {
			// Each case gets its own apiserver. The apiserver's Secrets informer keeps
			// re-reading a frame it can't read, which would change the next case's counters.
			test, err := newTransformTest(t, transformTestConfig{extraFlags: []string{noSecretsWatchCache}})
			if err != nil {
				t.Fatalf("failed to set up test: %v", err)
			}
			defer test.cleanUp()
			ctx := context.Background()
			secrets := test.restClient.CoreV1().Secrets(tc.namespace)
			if _, err := test.createNamespace(tc.namespace); err != nil {
				t.Fatal(err)
			}
			// The test overwrites planted in etcd below and leaves readable intact.
			for _, name := range []string{"readable", "planted"} {
				if _, err := test.createSecret(name, tc.namespace); err != nil {
					t.Fatal(err)
				}
			}
			// Replace planted's stored bytes with an intact frame this apiserver can't read.
			frame := tc.frame(t, test.rawSecret(t, tc.namespace, "planted"))
			test.plantSecret(t, tc.namespace, "planted", frame)
			wantBefore, quietBefore := secretsCounter(t, tc.wantCounter), secretsCounter(t, tc.quietCounter)
			// GET, LIST and unsafe DELETE all fail with InternalError, never StoreReadError.
			want := wantAPIStatusError{reason: metav1.StatusReasonInternalError}

			_, err = secrets.Get(ctx, "planted", metav1.GetOptions{})
			want.verify(t, err)
			// The LIST fails as a whole instead of returning the readable sibling.
			_, err = secrets.List(ctx, metav1.ListOptions{})
			want.verify(t, err)
			// Unsafe deletion is refused, so the stored value stays as planted.
			err = secrets.Delete(ctx, "planted", metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: new(true)})
			want.verify(t, err)
			if got := test.rawSecret(t, tc.namespace, "planted"); !bytes.Equal(got, frame) {
				t.Errorf("unsafe deletion changed the planted value: got %x, want %x", got, frame)
			}
			// The counters show which check rejected the frame, without matching error text.
			if secretsCounter(t, tc.wantCounter) == wantBefore {
				t.Errorf("%s didn't count the planted frame", tc.wantCounter)
			}
			if secretsCounter(t, tc.quietCounter) != quietBefore {
				t.Errorf("%s counted the planted frame", tc.quietCounter)
			}
		})
	}
}

// TestPinnedFrameDecodes plants pinnedFrameHex at a fixed etcd path and reads it
// through the apiserver.
func TestPinnedFrameDecodes(t *testing.T) {
	test, err := newTransformTest(t, transformTestConfig{})
	if err != nil {
		t.Fatalf("failed to set up test: %v", err)
	}
	defer test.cleanUp()

	frame, err := hex.DecodeString(pinnedFrameHex)
	if err != nil {
		t.Fatal(err)
	}
	test.plantSecret(t, pinnedSecretNamespace, pinnedSecretName, frame)
	got, err := test.restClient.CoreV1().Secrets(pinnedSecretNamespace).Get(context.Background(), pinnedSecretName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to read the pinned frame: %v", err)
	}
	if want := strings.Repeat("compress me ", 8); got.Name != pinnedSecretName || string(got.Data["note"]) != want {
		t.Errorf("got secret %s with note %q, want %s with note %q", got.Name, got.Data["note"], pinnedSecretName, want)
	}
}

// TestBitFlippedFrameCorruptObjectDeletion flips a bit inside a frame, with no
// encryption and with aescbc. The LIST reports a corrupt object, the frame's
// checksum rejects it, and unsafe deletion removes only that object. aescbc
// doesn't authenticate what it decrypts, so damaged ciphertext usually decrypts
// without an error and only the frame's checksum catches it.
func TestBitFlippedFrameCorruptObjectDeletion(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, true)
	tests := []struct {
		name             string
		encryptionConfig string
	}{
		{name: "no encryption"},
		{name: "aescbc", encryptionConfig: aesCBCConfigYAML},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			test, err := newTransformTest(t, transformTestConfig{transformerConfigYAML: tc.encryptionConfig, extraFlags: []string{noSecretsWatchCache}})
			if err != nil {
				t.Fatalf("failed to set up test: %v", err)
			}
			defer test.cleanUp()
			encryption := identity.NewEncryptCheckTransformer()
			if tc.encryptionConfig != "" {
				encryption = test.aesCBCTransformer(t)
			}

			ctx := context.Background()
			secrets := test.restClient.CoreV1().Secrets(testNamespace)
			for _, name := range []string{"readable", "planted"} {
				if _, err := test.createSecret(name, testNamespace); err != nil {
					t.Fatal(err)
				}
			}
			dataCtx := value.DefaultContext(test.getETCDPathForResource(test.storageConfig.Prefix, "", "secrets", "planted", testNamespace))
			plaintext, _, err := encryption.TransformFromStorage(ctx, test.rawSecret(t, testNamespace, "planted"), dataCtx)
			if err != nil {
				t.Fatal(err)
			}
			frame, err := compression.WithCompression(identity.NewEncryptCheckTransformer(), compressAll, secretsGR).TransformToStorage(ctx, plaintext, dataCtx)
			if err != nil {
				t.Fatal(err)
			}
			// Flip the bit before encrypting so that decryption succeeds and only the
			// checksum can catch it.
			frame[len(frame)-1] ^= 0x01
			stored, err := encryption.TransformToStorage(ctx, frame, dataCtx)
			if err != nil {
				t.Fatal(err)
			}
			test.plantSecret(t, testNamespace, "planted", stored)
			formatBefore, unsupportedBefore := secretsCounter(t, formatErrorsMetric), secretsCounter(t, unsupportedAlgorithmMetric)

			_, err = secrets.Get(ctx, "planted", metav1.GetOptions{})
			wantAPIStatusError{reason: metav1.StatusReasonInternalError}.verify(t, err)
			_, err = secrets.List(ctx, metav1.ListOptions{})
			wantAPIStatusError{reason: metav1.StatusReasonStoreReadError}.verify(t, err)
			// A format error means the compression reader caught the flip, not the decoder.
			if secretsCounter(t, formatErrorsMetric) == formatBefore {
				t.Errorf("%s didn't count the flipped frame", formatErrorsMetric)
			}
			if secretsCounter(t, unsupportedAlgorithmMetric) != unsupportedBefore {
				t.Errorf("%s counted the flipped frame", unsupportedAlgorithmMetric)
			}
			err = secrets.Delete(ctx, "planted", metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: new(true)})
			wantNoError{}.verify(t, err)
			_, err = secrets.Get(ctx, "planted", metav1.GetOptions{})
			wantAPIStatusError{reason: metav1.StatusReasonNotFound}.verify(t, err)
			if got := test.rawSecret(t, testNamespace, "planted"); got != nil {
				t.Errorf("unsafe deletion left %x in etcd", got)
			}
			// Unsafe deletion removed only the planted secret, so the namespace lists again.
			list, err := secrets.List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("failed to list secrets after the unsafe deletion: %v", err)
			}
			var names []string
			for _, s := range list.Items {
				names = append(names, s.Name)
			}
			if len(names) != 1 || names[0] != "readable" {
				t.Errorf("got secrets %v after the unsafe deletion, want only [readable]", names)
			}
		})
	}
}
