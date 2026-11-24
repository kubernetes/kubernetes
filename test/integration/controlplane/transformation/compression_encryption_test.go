//go:build !windows

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
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	apiserverv1 "k8s.io/apiserver/pkg/apis/apiserver/v1"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/storage/value"
	kmsv2mock "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing/v2"
	kmsv2api "k8s.io/kms/apis/v2"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	aesGCMKey2Prefix = "k8s:enc:aesgcm:v1:key2:"
	compressionKMS   = "compression-kms"

	aesGCMKey1Provider = `    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`
	aesCBCKey1Provider = `    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`
	// key2 writes and key1 still reads, the state of a cluster partway
	// through rotating its aesgcm key.
	aesGCMKey2PrimaryProvider = `    - aesgcm:
        keys:
        - name: key2
          secret: cm90YXRlZCBrZXkgMTZiIQ==
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`
	kmsv2Provider = `    - kms:
        apiVersion: v2
        name: ` + compressionKMS + `
        endpoint: unix:///@` + compressionKMS + `.sock
`
)

// encryptAtRest returns an EncryptionConfiguration that encrypts configmaps
// and secrets with providers.
func encryptAtRest(providers string) string {
	return `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - configmaps
    - secrets
    providers:
` + providers
}

// kmsv2Unsealer opens envelopes from the KMS v2 provider name. plugin unwraps
// the DEK source that each envelope carries, and that opens the payload.
func kmsv2Unsealer(plugin *kmsv2mock.Base64Plugin, name string) unSealSecret {
	return func(ctx context.Context, cipherText []byte, dataCtx value.Context, _ apiserverv1.ProviderConfiguration) ([]byte, error) {
		envelope := envelopekmsv2{providerName: name, useSeed: encryptionconfig.GetKDF(name)}
		envelope.rawEnvelope = append([]byte(envelope.prefix()), cipherText...)
		wrapped, err := envelope.cipherTextDEKSource()
		if err != nil {
			return nil, err
		}
		resp, err := plugin.Decrypt(ctx, &kmsv2api.DecryptRequest{Uid: string(uuid.NewUUID()), Ciphertext: wrapped})
		if err != nil {
			return nil, err
		}
		envelope.plainTextDEKSource = resp.Plaintext
		return envelope.plainTextPayload(string(dataCtx.AuthenticatedData()))
	}
}

// unsealRaw reads an object's value in testNamespace straight from etcd,
// checks that it's encrypted under prefix, and returns what unseal decrypts
// it to. unseal gets the first provider of the test's encryption config.
func (e *transformTest) unsealRaw(t *testing.T, gr schema.GroupResource, name, prefix string, unseal unSealSecret) []byte {
	t.Helper()
	raw := e.rawValue(t, gr, testNamespace, name)
	if !bytes.HasPrefix(raw, []byte(prefix)) {
		t.Fatalf("%s %s is stored starting %q, want it encrypted under %q", gr.Resource, name, raw[:min(len(raw), len(prefix))], prefix)
	}
	config, err := e.getEncryptionConfig()
	if err != nil {
		t.Fatal(err)
	}
	dataCtx := value.DefaultContext(e.getETCDPathForResource(e.storageConfig.Prefix, gr.Group, gr.Resource, name, testNamespace))
	plaintext, err := unseal(testContext(t), raw[len(prefix):], dataCtx, *config)
	if err != nil {
		t.Fatalf("failed to decrypt %s %s: %v", gr.Resource, name, err)
	}
	return plaintext
}

// TestStorageCompressionUnderEncryption checks that compression runs before
// encryption: a configmap decrypts to a frame, while a secret, which '*.'
// doesn't select, decrypts to bare protobuf.
func TestStorageCompressionUnderEncryption(t *testing.T) {
	policy := compressionConfig(compressionEntry("*.", "Deflate", ""))
	tests := []struct {
		name     string
		provider string
		prefix   string
		unseal   unSealSecret
		// kms, if set, names the mock KMS v2 plugin to start. Its unsealer
		// replaces unseal.
		kms string
	}{
		{name: "aesgcm", provider: aesGCMKey1Provider, prefix: aesGCMPrefix, unseal: unSealWithGCMTransformer},
		{name: "aescbc", provider: aesCBCKey1Provider, prefix: aesCBCPrefix, unseal: unSealWithCBCTransformer},
		{name: "kms v2", provider: kmsv2Provider, prefix: envelopekmsv2{providerName: compressionKMS}.prefix(), kms: compressionKMS},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			unseal := tc.unseal
			if tc.kms != "" {
				unseal = kmsv2Unsealer(kmsv2mock.NewBase64Plugin(t, "@"+tc.kms+".sock"), tc.kms)
			}
			test, err := newTransformTest(t, transformTestConfig{
				transformerConfigYAML: encryptAtRest(tc.provider),
				extraFlags:            compressionFlags(t, policy, "1.38"),
			})
			if err != nil {
				t.Fatalf("failed to start kube-apiserver: %v", err)
			}
			defer test.cleanUp()

			cm := compressibleConfigMap("compressible")
			test.mustCreateConfigMap(t, cm)
			if got := formOf(test.unsealRaw(t, configMapsGVR.GroupResource(), cm.Name, tc.prefix, unseal)); got != formDeflate {
				t.Errorf("configmap %s decrypts to %s, want %s", cm.Name, got, formDeflate)
			}
			test.wantConfigMapContent(t, cm)

			ctx := testContext(t)
			secrets := test.restClient.CoreV1().Secrets(testNamespace)
			secret := &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "compressible"},
				Data:       map[string][]byte{"data": []byte(compressibleText(2400))},
			}
			if _, err := secrets.Create(ctx, secret, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create secret %s: %v", secret.Name, err)
			}
			if got := formOf(test.unsealRaw(t, secretsGR, secret.Name, tc.prefix, unseal)); got != formUnframed {
				t.Errorf("secret %s decrypts to %s, want %s", secret.Name, got, formUnframed)
			}
			got, err := secrets.Get(ctx, secret.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to read secret %s: %v", secret.Name, err)
			}
			if !bytes.Equal(got.Data["data"], secret.Data["data"]) {
				t.Errorf("secret %s read back with different data", secret.Name)
			}
		})
	}
}

// TestStorageCompressionKeyRotation writes a compressed configmap under one
// aesgcm key, then restarts with a second key in front of it and the same
// compression policy. Only the encryption provider finds the old value stale,
// and the update that rewrites it has to put the same kind of frame under the
// new key.
func TestStorageCompressionKeyRotation(t *testing.T) {
	storageConfig := framework.SharedEtcd()
	policy := compressionConfig(compressionEntry("configmaps", "Deflate", ""))
	cm := compressibleConfigMap("rotated")
	start := func(t *testing.T, providers string) *transformTest {
		t.Helper()
		test, err := newTransformTest(t, transformTestConfig{
			transformerConfigYAML: encryptAtRest(providers),
			storageConfig:         storageConfig,
			extraFlags:            compressionFlags(t, policy, "1.38"),
		})
		if err != nil {
			t.Fatalf("failed to start kube-apiserver: %v", err)
		}
		return test
	}

	if !t.Run("written under key1", func(t *testing.T) {
		test := start(t, aesGCMKey1Provider)
		defer test.cleanUp()
		test.mustCreateConfigMap(t, cm)
		if got := formOf(test.unsealRaw(t, configMapsGVR.GroupResource(), cm.Name, aesGCMPrefix, unSealWithGCMTransformer)); got != formDeflate {
			t.Errorf("configmap %s decrypts to %s, want %s", cm.Name, got, formDeflate)
		}
	}) {
		return
	}

	t.Run("key2 made primary", func(t *testing.T) {
		test := start(t, aesGCMKey2PrimaryProvider)
		defer test.cleanUp()
		// The policy hasn't changed, so the framing half of the read never
		// reports stale. Any rewrite here is the encryption provider's.
		configMapRewrites := map[string]string{"resource": "configmaps"}
		rewritesBefore := counterValue(t, rewritesNeededMetric, configMapRewrites)

		// key1 still decrypts it, and a read alone doesn't write it back.
		test.wantConfigMapContent(t, cm)
		if raw := test.rawValue(t, configMapsGVR.GroupResource(), testNamespace, cm.Name); !bytes.HasPrefix(raw, []byte(aesGCMPrefix)) {
			t.Errorf("reading configmap %s re-encrypted it, want it left under key1 until something writes it", cm.Name)
		}
		// The read reported key1's value stale, so the first no-op update rewrites
		// it under key2. The compression layer frames it again on the way.
		if !test.noOpUpdate(t, configMapsGVR, cm.Name) {
			t.Errorf("the first no-op update of configmap %s kept its resource version, want key1's stale value rewritten", cm.Name)
		}
		// key2 is listed first, so unsealRaw decrypts with it.
		if got := formOf(test.unsealRaw(t, configMapsGVR.GroupResource(), cm.Name, aesGCMKey2Prefix, unSealWithGCMTransformer)); got != formDeflate {
			t.Errorf("configmap %s decrypts under key2 to %s, want %s", cm.Name, got, formDeflate)
		}
		// Now it's under the primary key and framed as the policy wants, so nothing
		// is stale and the second no-op update writes nothing.
		if test.noOpUpdate(t, configMapsGVR, cm.Name) {
			t.Errorf("the second no-op update of configmap %s changed its resource version again", cm.Name)
		}
		test.wantConfigMapContent(t, cm)
		// A flat counter shows the framing never asked for the rewrite.
		if got := counterValue(t, rewritesNeededMetric, configMapRewrites); got != rewritesBefore {
			t.Errorf("%s for configmaps went from %v to %v, want no framing rewrites", rewritesNeededMetric, rewritesBefore, got)
		}
	})
}
