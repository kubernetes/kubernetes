//go:build !windows
// +build !windows

/*
Copyright 2022 The Kubernetes Authors.

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
	"crypto/aes"
	"crypto/cipher"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	clientv3 "go.etcd.io/etcd/client/v3"

	corev1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2"
	kmsv2mock "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing/v2"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	kmsv2api "k8s.io/kms/apis/v2"
	kmsv2svc "k8s.io/kms/pkg/service"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	secretstore "k8s.io/kubernetes/pkg/registry/core/secret/storage"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

type envelopekmsv2 struct {
	providerName       string
	rawEnvelope        []byte
	plainTextDEKSource []byte
	useSeed            bool
}

func (r envelopekmsv2) prefix() string {
	return fmt.Sprintf("k8s:enc:kms:v2:%s:", r.providerName)
}

func (r envelopekmsv2) prefixLen() int {
	return len(r.prefix())
}

func (r envelopekmsv2) cipherTextDEKSource() ([]byte, error) {
	o := &kmstypes.EncryptedObject{}
	if err := proto.Unmarshal(r.rawEnvelope[r.startOfPayload(r.providerName):], o); err != nil {
		return nil, err
	}

	if err := kmsv2.ValidateEncryptedObject(o); err != nil {
		return nil, err
	}

	if r.useSeed && o.EncryptedDEKSourceType != kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED {
		return nil, fmt.Errorf("wrong type used with useSeed=true")
	}

	if !r.useSeed && o.EncryptedDEKSourceType != kmstypes.EncryptedDEKSourceType_AES_GCM_KEY {
		return nil, fmt.Errorf("wrong type used with useSeed=false")
	}

	return o.EncryptedDEKSource, nil
}

func (r envelopekmsv2) startOfPayload(_ string) int {
	return r.prefixLen()
}

func (r envelopekmsv2) cipherTextPayload() ([]byte, error) {
	o := &kmstypes.EncryptedObject{}
	if err := proto.Unmarshal(r.rawEnvelope[r.startOfPayload(r.providerName):], o); err != nil {
		return nil, err
	}

	if err := kmsv2.ValidateEncryptedObject(o); err != nil {
		return nil, err
	}

	return o.EncryptedData, nil
}

func (r envelopekmsv2) plainTextPayload(secretETCDPath string) ([]byte, error) {
	var transformer value.Read
	var err error
	if r.useSeed {
		transformer, err = aestransformer.NewHKDFExtendedNonceGCMTransformer(r.plainTextDEKSource)
	} else {
		var block cipher.Block
		block, err = aes.NewCipher(r.plainTextDEKSource)
		if err != nil {
			return nil, err
		}
		transformer, err = aestransformer.NewGCMTransformer(block)
	}
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	dataCtx := value.DefaultContext(secretETCDPath)

	data, err := r.cipherTextPayload()
	if err != nil {
		return nil, fmt.Errorf("failed to get cipher text payload: %v", err)
	}
	plainSecret, _, err := transformer.TransformFromStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to transform from storage via AESGCM, err: %w", err)
	}

	return plainSecret, nil
}

// TestKMSv2Provider is an integration test between KubeAPI, ETCD and KMSv2 Plugin
// Concretely, this test verifies the following integration contracts:
// 1. Raw records in ETCD that were processed by KMSv2 Provider should be prefixed with k8s:enc:kms:v2:<plugin name>:
// 2. Data Encryption Key (DEK) / DEK seed should be generated by envelopeTransformer and passed to KMS gRPC Plugin
// 3. KMS gRPC Plugin should encrypt the DEK/seed with a Key Encryption Key (KEK) and pass it back to envelopeTransformer
// 4. The cipherTextPayload (ex. Secret) should be encrypted via AES GCM transform / extended nonce GCM
// 5. kmstypes.EncryptedObject structure should be serialized and deposited in ETCD
func TestKMSv2Provider(t *testing.T) {
	t.Run("regular gcm", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, false)()
		testKMSv2Provider(t)
	})

	t.Run("extended nonce gcm", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, true)()
		testKMSv2Provider(t)
	})
}

func testKMSv2Provider(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()

	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`

	providerName := "kms-provider"
	pluginMock := kmsv2mock.NewBase64Plugin(t, "@kms-provider.sock")

	test, err := newTransformTest(t, encryptionConfig, false, "", nil)
	if err != nil {
		t.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	defer test.cleanUp()

	test.secret, err = test.createSecret(testSecret, testNamespace)
	if err != nil {
		t.Fatalf("Failed to create test secret, error: %v", err)
	}

	plainTextDEKSource := pluginMock.LastEncryptRequest()

	secretETCDPath := test.getETCDPathForResource(test.storageConfig.Prefix, "", "secrets", test.secret.Name, test.secret.Namespace)
	rawEnvelope, err := test.getRawSecretFromETCD()
	if err != nil {
		t.Fatalf("failed to read %s from etcd: %v", secretETCDPath, err)
	}

	envelopeData := envelopekmsv2{
		providerName:       providerName,
		rawEnvelope:        rawEnvelope,
		plainTextDEKSource: plainTextDEKSource,
		useSeed:            utilfeature.DefaultFeatureGate.Enabled(features.KMSv2KDF),
	}

	wantPrefix := envelopeData.prefix()
	if !bytes.HasPrefix(rawEnvelope, []byte(wantPrefix)) {
		t.Fatalf("expected secret to be prefixed with %s, but got %s", wantPrefix, rawEnvelope)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ciphertext, err := envelopeData.cipherTextDEKSource()
	if err != nil {
		t.Fatalf("failed to get ciphertext DEK/seed from KMSv2 Plugin: %v", err)
	}
	decryptResponse, err := pluginMock.Decrypt(ctx, &kmsv2api.DecryptRequest{Uid: string(uuid.NewUUID()), Ciphertext: ciphertext})
	if err != nil {
		t.Fatalf("failed to decrypt DEK, %v", err)
	}
	dekSourcePlainAsWouldBeSeenByETCD := decryptResponse.Plaintext

	if !bytes.Equal(plainTextDEKSource, dekSourcePlainAsWouldBeSeenByETCD) {
		t.Fatalf("expected plainTextDEKSource %v to be passed to KMS Plugin, but got %s",
			plainTextDEKSource, dekSourcePlainAsWouldBeSeenByETCD)
	}

	plainSecret, err := envelopeData.plainTextPayload(secretETCDPath)
	if err != nil {
		t.Fatalf("failed to transform from storage via AESGCM, err: %v", err)
	}

	if !strings.Contains(string(plainSecret), secretVal) {
		t.Fatalf("expected %q after decryption, but got %q", secretVal, string(plainSecret))
	}

	secretClient := test.restClient.CoreV1().Secrets(testNamespace)
	// Secrets should be un-enveloped on direct reads from Kube API Server.
	s, err := secretClient.Get(ctx, testSecret, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get Secret from %s, err: %v", testNamespace, err)
	}
	if secretVal != string(s.Data[secretKey]) {
		t.Fatalf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
	}
}

// TestKMSv2ProviderKeyIDStaleness is an integration test between KubeAPI and KMSv2 Plugin
// Concretely, this test verifies the following contracts for no-op updates:
// 1. When the key ID is unchanged, the resource version must not change
// 2. When the key ID changes, the resource version changes (but only once)
// 3. For all subsequent updates, the resource version must not change
// 4. When kms-plugin is down, expect creation of new pod and encryption to succeed while the DEK/seed is valid
// 5. when kms-plugin is down, no-op update for a pod should succeed and not result in RV change while the DEK/seed is valid
// 6. When kms-plugin is down, expect creation of new pod and encryption to fail once the DEK/seed is invalid
// 7. when kms-plugin is down, no-op update for a pod should succeed and not result in RV change even once the DEK/seed is valid
func TestKMSv2ProviderKeyIDStaleness(t *testing.T) {
	t.Run("regular gcm", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, false)()
		testKMSv2ProviderKeyIDStaleness(t)
	})

	t.Run("extended nonce gcm", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, true)()
		testKMSv2ProviderKeyIDStaleness(t)
	})
}

func testKMSv2ProviderKeyIDStaleness(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()

	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - pods
    - deployments.apps
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`
	pluginMock := kmsv2mock.NewBase64Plugin(t, "@kms-provider.sock")

	test, err := newTransformTest(t, encryptionConfig, false, "", nil)
	if err != nil {
		t.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	defer test.cleanUp()

	dynamicClient := dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)

	testPod, err := test.createPod(testNamespace, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to create test pod, error: %v, ns: %s", err, testNamespace)
	}
	version1 := testPod.GetResourceVersion()

	// 1. no-op update for the test pod should not result in any RV change
	updatedPod, err := test.inplaceUpdatePod(testNamespace, testPod, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to update test pod, error: %v, ns: %s", err, testNamespace)
	}
	version2 := updatedPod.GetResourceVersion()
	if version1 != version2 {
		t.Fatalf("Resource version should not have changed. old pod: %v, new pod: %v", testPod, updatedPod)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)

	useSeed := utilfeature.DefaultFeatureGate.Enabled(features.KMSv2KDF)

	var firstEncryptedDEKSource []byte
	var f checkFunc
	if useSeed {
		f = func(_ int, _ uint64, etcdKey string, obj kmstypes.EncryptedObject) {
			firstEncryptedDEKSource = obj.EncryptedDEKSource

			if obj.KeyID != "1" {
				t.Errorf("key %s: want key ID %s, got %s", etcdKey, "1", obj.KeyID)
			}
		}
	} else {
		f = func(_ int, counter uint64, etcdKey string, obj kmstypes.EncryptedObject) {
			firstEncryptedDEKSource = obj.EncryptedDEKSource

			if obj.KeyID != "1" {
				t.Errorf("key %s: want key ID %s, got %s", etcdKey, "1", obj.KeyID)
			}

			// with the first key we perform encryption during the following steps:
			// - create
			const want = 1_000_000_000 + 1 // zero value of counter is one billion
			if want != counter {
				t.Errorf("key %s: counter nonce is invalid: want %d, got %d", etcdKey, want, counter)
			}
		}
	}
	assertPodDEKSources(ctx, t, test.kubeAPIServer.ServerOpts.Etcd.StorageConfig,
		1, 1, "k8s:enc:kms:v2:kms-provider:", f,
	)
	if len(firstEncryptedDEKSource) == 0 {
		t.Fatal("unexpected empty DEK or seed")
	}

	// 2. no-op update for the test pod with keyID update should result in RV change
	pluginMock.UpdateKeyID()
	if err := kmsv2mock.WaitForBase64PluginToBeUpdated(pluginMock); err != nil {
		t.Fatalf("Failed to update keyID for plugin, err: %v", err)
	}
	// Wait 1 sec (poll interval to check resource version) until a resource version change is detected or timeout at 1 minute.

	version3 := ""
	err = wait.Poll(time.Second, time.Minute,
		func() (bool, error) {
			t.Log("polling for in-place update rv change")
			updatedPod, err = test.inplaceUpdatePod(testNamespace, updatedPod, dynamicClient)
			if err != nil {
				return false, err
			}
			version3 = updatedPod.GetResourceVersion()
			if version1 != version3 {
				return true, nil
			}
			return false, nil
		})
	if err != nil {
		t.Fatalf("Failed to detect one resource version update within the allotted time after keyID is updated and pod has been inplace updated, err: %v, ns: %s", err, testNamespace)
	}

	if version1 == version3 {
		t.Fatalf("Resource version should have changed after keyID update. old pod: %v, new pod: %v", testPod, updatedPod)
	}

	var wantCount uint64 = 1_000_000_000 // zero value of counter is one billion
	wantCount++                          // in place update with RV change

	// with the second key we perform encryption during the following steps:
	// - in place update with RV change
	// - delete (which does an update to set deletion timestamp)
	// - create
	var checkDEK checkFunc
	if useSeed {
		checkDEK = func(_ int, _ uint64, etcdKey string, obj kmstypes.EncryptedObject) {
			if len(obj.EncryptedDEKSource) == 0 {
				t.Error("unexpected empty DEK source")
			}

			if bytes.Equal(obj.EncryptedDEKSource, firstEncryptedDEKSource) {
				t.Errorf("key %s: incorrectly has the same ESEED", etcdKey)
			}

			if obj.KeyID != "2" {
				t.Errorf("key %s: want key ID %s, got %s", etcdKey, "2", obj.KeyID)
			}
		}
	} else {
		checkDEK = func(_ int, counter uint64, etcdKey string, obj kmstypes.EncryptedObject) {
			if len(obj.EncryptedDEKSource) == 0 {
				t.Error("unexpected empty DEK source")
			}

			if bytes.Equal(obj.EncryptedDEKSource, firstEncryptedDEKSource) {
				t.Errorf("key %s: incorrectly has the same EDEK", etcdKey)
			}

			if obj.KeyID != "2" {
				t.Errorf("key %s: want key ID %s, got %s", etcdKey, "2", obj.KeyID)
			}

			if wantCount != counter {
				t.Errorf("key %s: counter nonce is invalid: want %d, got %d", etcdKey, wantCount, counter)
			}
		}
	}

	// 3. no-op update for the updated pod should not result in RV change
	updatedPod, err = test.inplaceUpdatePod(testNamespace, updatedPod, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to update test pod, error: %v, ns: %s", err, testNamespace)
	}
	version4 := updatedPod.GetResourceVersion()
	if version3 != version4 {
		t.Fatalf("Resource version should not have changed again after the initial version updated as a result of the keyID update. old pod: %v, new pod: %v", testPod, updatedPod)
	}

	// delete the pod so that it can be recreated
	if err := test.deletePod(testNamespace, dynamicClient); err != nil {
		t.Fatalf("failed to delete test pod: %v", err)
	}
	wantCount++ // we cannot assert against the counter being 2 since the pod gets deleted

	// 4. when kms-plugin is down, expect creation of new pod and encryption to succeed because the DEK is still valid
	pluginMock.EnterFailedState()
	mustBeUnHealthy(t, "/kms-providers",
		"internal server error: kms-provider-0: rpc error: code = FailedPrecondition desc = failed precondition - key disabled",
		test.kubeAPIServer.ClientConfig)

	newPod, err := test.createPod(testNamespace, dynamicClient)
	if err != nil {
		t.Fatalf("Create test pod should have succeeded due to valid DEK, ns: %s, got: %v", testNamespace, err)
	}
	wantCount++
	version5 := newPod.GetResourceVersion()

	// 5. when kms-plugin is down and DEK is valid, no-op update for a pod should succeed and not result in RV change
	updatedPod, err = test.inplaceUpdatePod(testNamespace, newPod, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to perform no-op update on pod when kms-plugin is down, error: %v, ns: %s", err, testNamespace)
	}
	version6 := updatedPod.GetResourceVersion()
	if version5 != version6 {
		t.Fatalf("Resource version should not have changed again after the initial version updated as a result of the keyID update. old pod: %v, new pod: %v", newPod, updatedPod)
	}

	// Invalidate the DEK by moving the current time forward
	origNowFunc := kmsv2.NowFunc
	t.Cleanup(func() { kmsv2.NowFunc = origNowFunc })
	kmsv2.NowFunc = func() time.Time { return origNowFunc().Add(5 * time.Minute) }

	// 6. when kms-plugin is down, expect creation of new pod and encryption to fail because the DEK is invalid
	_, err = test.createPod(testNamespace, dynamicClient)
	if err == nil || !strings.Contains(err.Error(), `encryptedDEKSource with keyID hash "sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" expired at 2`) {
		t.Fatalf("Create test pod should have failed due to encryption, ns: %s, got: %v", testNamespace, err)
	}

	// 7. when kms-plugin is down and DEK is invalid, no-op update for a pod should succeed and not result in RV change
	updatedNewPod, err := test.inplaceUpdatePod(testNamespace, newPod, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to perform no-op update on pod when kms-plugin is down, error: %v, ns: %s", err, testNamespace)
	}
	version7 := updatedNewPod.GetResourceVersion()
	if version5 != version7 {
		t.Fatalf("Resource version should not have changed again after the initial version updated as a result of the keyID update. old pod: %v, new pod: %v", newPod, updatedNewPod)
	}

	assertPodDEKSources(ctx, t, test.kubeAPIServer.ServerOpts.Etcd.StorageConfig,
		1, 1, "k8s:enc:kms:v2:kms-provider:", checkDEK,
	)

	// fix plugin and wait for new writes to start working again
	kmsv2.NowFunc = origNowFunc
	pluginMock.ExitFailedState()
	err = wait.Poll(time.Second, 3*time.Minute,
		func() (bool, error) {
			t.Log("polling for plugin to be functional")
			_, err = test.createDeployment("panda", testNamespace)
			if err != nil {
				t.Logf("failed to create deployment, plugin is likely still unhealthy: %v", err)
			}
			return err == nil, nil
		})
	if err != nil {
		t.Fatalf("failed to restore plugin health, err: %v, ns: %s", err, testNamespace)
	}

	// 8. confirm that no-op update for a pod succeeds and still does not result in RV change
	updatedNewPod2, err := test.inplaceUpdatePod(testNamespace, updatedNewPod, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to perform no-op update on pod when kms-plugin is up, error: %v, ns: %s", err, testNamespace)
	}
	version8 := updatedNewPod2.GetResourceVersion()
	if version7 != version8 {
		t.Fatalf("Resource version should not have changed after plugin health is restored. old pod: %v, new pod: %v", updatedNewPod, updatedNewPod2)
	}

	// flip the current config
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, !useSeed)()

	// 9. confirm that no-op update for a pod results in RV change due to KDF config change
	var version9 string
	err = wait.Poll(time.Second, 3*time.Minute,
		func() (bool, error) {
			t.Log("polling for in-place update rv change due to KDF config change")
			updatedNewPod2, err = test.inplaceUpdatePod(testNamespace, updatedNewPod2, dynamicClient)
			if err != nil {
				return false, err
			}
			version9 = updatedNewPod2.GetResourceVersion()
			if version8 != version9 {
				return true, nil
			}
			return false, nil
		})
	if err != nil {
		t.Fatalf("Failed to detect one resource version update within the allotted time after KDF config change and pod has been inplace updated, err: %v, ns: %s", err, testNamespace)
	}
}

func TestKMSv2ProviderDEKSourceReuse(t *testing.T) {
	t.Run("regular gcm", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, false)()
		testKMSv2ProviderDEKSourceReuse(t,
			func(i int, counter uint64, etcdKey string, obj kmstypes.EncryptedObject) {
				if obj.KeyID != "1" {
					t.Errorf("key %s: want key ID %s, got %s", etcdKey, "1", obj.KeyID)
				}

				// zero value of counter is one billion so the first value will be one billion plus one
				// hence we add that to our zero based index to calculate the expected nonce
				if uint64(i+1_000_000_000+1) != counter {
					t.Errorf("key %s: counter nonce is invalid: want %d, got %d", etcdKey, i+1, counter)
				}
			},
		)
	})

	t.Run("extended nonce gcm", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, true)()
		testKMSv2ProviderDEKSourceReuse(t,
			func(_ int, _ uint64, etcdKey string, obj kmstypes.EncryptedObject) {
				if obj.KeyID != "1" {
					t.Errorf("key %s: want key ID %s, got %s", etcdKey, "1", obj.KeyID)
				}
			},
		)
	})
}

func testKMSv2ProviderDEKSourceReuse(t *testing.T, f checkFunc) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	t.Cleanup(cancel)

	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - pods
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`
	_ = kmsv2mock.NewBase64Plugin(t, "@kms-provider.sock")

	test, err := newTransformTest(t, encryptionConfig, false, "", nil)
	if err != nil {
		t.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	t.Cleanup(test.cleanUp)

	client := kubernetes.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)

	const podCount = 1_000

	for i := 0; i < podCount; i++ {
		if _, err := client.CoreV1().Pods(testNamespace).Create(ctx, &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("dek-reuse-%04d", i+1), // making creation order match returned list order / nonce counter
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  "busybox",
						Image: "busybox",
					},
				},
			},
		}, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	assertPodDEKSources(ctx, t, test.kubeAPIServer.ServerOpts.Etcd.StorageConfig,
		podCount, 1, // key ID does not change during the test so we should only have a single DEK
		"k8s:enc:kms:v2:kms-provider:", f,
	)
}

type checkFunc func(i int, counter uint64, etcdKey string, obj kmstypes.EncryptedObject)

func assertPodDEKSources(ctx context.Context, t *testing.T, config storagebackend.Config, podCount, dekSourcesCount int, kmsPrefix string, f checkFunc) {
	t.Helper()

	rawClient, etcdClient, err := integration.GetEtcdClients(config.Transport)
	if err != nil {
		t.Fatalf("failed to create etcd client: %v", err)
	}
	t.Cleanup(func() { _ = rawClient.Close() })

	response, err := etcdClient.Get(ctx, "/"+config.Prefix+"/pods/"+testNamespace+"/", clientv3.WithPrefix())
	if err != nil {
		t.Fatal(err)
	}

	if len(response.Kvs) != podCount {
		t.Fatalf("expected %d KVs, but got %d", podCount, len(response.Kvs))
	}

	useSeed := utilfeature.DefaultFeatureGate.Enabled(features.KMSv2KDF)

	out := make([]kmstypes.EncryptedObject, len(response.Kvs))
	for i, kv := range response.Kvs {
		v := bytes.TrimPrefix(kv.Value, []byte(kmsPrefix))
		if err := proto.Unmarshal(v, &out[i]); err != nil {
			t.Fatal(err)
		}

		if err := kmsv2.ValidateEncryptedObject(&out[i]); err != nil {
			t.Fatal(err)
		}

		var infoLen int
		if useSeed {
			infoLen = 32
		}

		info := out[i].EncryptedData[:infoLen]
		nonce := out[i].EncryptedData[infoLen : 12+infoLen]
		randN := nonce[:4]
		count := nonce[4:]

		if bytes.Equal(randN, make([]byte, len(randN))) {
			t.Errorf("key %s: got all zeros for first four bytes", string(kv.Key))
		}

		if useSeed {
			if bytes.Equal(info, make([]byte, infoLen)) {
				t.Errorf("key %s: got all zeros for info", string(kv.Key))
			}
		}

		counter := binary.LittleEndian.Uint64(count)
		f(i, counter, string(kv.Key), out[i])
	}

	uniqueDEKSources := sets.NewString()
	for _, object := range out {
		object := object
		uniqueDEKSources.Insert(string(object.EncryptedDEKSource))
		if useSeed {
			if object.EncryptedDEKSourceType != kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED {
				t.Errorf("invalid type: %d", object.EncryptedDEKSourceType)
			}
		} else {
			if object.EncryptedDEKSourceType != kmstypes.EncryptedDEKSourceType_AES_GCM_KEY {
				t.Errorf("invalid type: %d", object.EncryptedDEKSourceType)
			}
		}
	}

	if uniqueDEKSources.Has("") {
		t.Error("unexpected empty DEK source seen")
	}

	if uniqueDEKSources.Len() != dekSourcesCount {
		t.Errorf("expected %d DEK sources, got: %d", dekSourcesCount, uniqueDEKSources.Len())
	}
}

func TestKMSv2Healthz(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, randomBool())()

	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
       apiVersion: v2
       name: provider-1
       endpoint: unix:///@kms-provider-1.sock
    - kms:
       apiVersion: v2
       name: provider-2
       endpoint: unix:///@kms-provider-2.sock
`

	pluginMock1 := kmsv2mock.NewBase64Plugin(t, "@kms-provider-1.sock")
	pluginMock2 := kmsv2mock.NewBase64Plugin(t, "@kms-provider-2.sock")

	test, err := newTransformTest(t, encryptionConfig, false, "", nil)
	if err != nil {
		t.Fatalf("Failed to start kube-apiserver, error: %v", err)
	}
	defer test.cleanUp()

	// Name of the healthz check is always "kms-provider-0" and it covers all kms plugins.

	// Stage 1 - Since all kms-plugins are guaranteed to be up,
	// the healthz check should be OK.
	mustBeHealthy(t, "/kms-providers", "ok", test.kubeAPIServer.ClientConfig)

	// Stage 2 - kms-plugin for provider-1 is down. Therefore, expect the healthz check
	// to fail and report that provider-1 is down
	pluginMock1.EnterFailedState()
	mustBeUnHealthy(t, "/kms-providers",
		"internal server error: kms-provider-0: rpc error: code = FailedPrecondition desc = failed precondition - key disabled",
		test.kubeAPIServer.ClientConfig)
	pluginMock1.ExitFailedState()

	// Stage 3 - kms-plugin for provider-1 is now up. Therefore, expect the health check for provider-1
	// to succeed now, but provider-2 is now down.
	pluginMock2.EnterFailedState()
	mustBeUnHealthy(t, "/kms-providers",
		"internal server error: kms-provider-1: rpc error: code = FailedPrecondition desc = failed precondition - key disabled",
		test.kubeAPIServer.ClientConfig)
	pluginMock2.ExitFailedState()

	// Stage 4 - All kms-plugins are once again up,
	// the healthz check should be OK.
	mustBeHealthy(t, "/kms-providers", "ok", test.kubeAPIServer.ClientConfig)

	// Stage 5 - All kms-plugins are unhealthy at the same time and we can observe both failures.
	pluginMock1.EnterFailedState()
	pluginMock2.EnterFailedState()
	mustBeUnHealthy(t, "/kms-providers",
		"internal server error: "+
			"[kms-provider-0: failed to perform status section of the healthz check for KMS Provider provider-1, error: rpc error: code = FailedPrecondition desc = failed precondition - key disabled,"+
			" kms-provider-1: failed to perform status section of the healthz check for KMS Provider provider-2, error: rpc error: code = FailedPrecondition desc = failed precondition - key disabled]",
		test.kubeAPIServer.ClientConfig)
}

func TestKMSv2SingleService(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2KDF, randomBool())()

	var kmsv2Calls int
	origEnvelopeKMSv2ServiceFactory := encryptionconfig.EnvelopeKMSv2ServiceFactory
	encryptionconfig.EnvelopeKMSv2ServiceFactory = func(ctx context.Context, endpoint, providerName string, callTimeout time.Duration) (kmsv2svc.Service, error) {
		kmsv2Calls++
		return origEnvelopeKMSv2ServiceFactory(ctx, endpoint, providerName, callTimeout)
	}
	t.Cleanup(func() {
		encryptionconfig.EnvelopeKMSv2ServiceFactory = origEnvelopeKMSv2ServiceFactory
	})

	// check resources provided by the three servers that we have wired together
	// - pods and config maps from KAS
	// - CRDs and CRs from API extensions
	// - API services from aggregator
	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - pods
    - configmaps
    - customresourcedefinitions.apiextensions.k8s.io
    - pandas.awesome.bears.com
    - apiservices.apiregistration.k8s.io
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`

	_ = kmsv2mock.NewBase64Plugin(t, "@kms-provider.sock")

	test, err := newTransformTest(t, encryptionConfig, false, "", nil)
	if err != nil {
		t.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	t.Cleanup(test.cleanUp)

	// the storage registry for CRs is dynamic so create one to exercise the wiring
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(test.kubeAPIServer.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	gvr := schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v1", Resource: "pandas"}
	stub := etcd.GetEtcdStorageData()[gvr].Stub
	dynamicClient, obj, err := etcd.JSONToUnstructured(stub, "", &meta.RESTMapping{
		Resource:         gvr,
		GroupVersionKind: gvr.GroupVersion().WithKind("Panda"),
		Scope:            meta.RESTScopeRoot,
	}, dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig))
	if err != nil {
		t.Fatal(err)
	}
	_, err = dynamicClient.Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if kmsv2Calls != 1 {
		t.Fatalf("expected a single call to KMS v2 service factory: %v", kmsv2Calls)
	}
}

// TestKMSv2FeatureFlag is an integration test between KubeAPI and ETCD
// Concretely, this test verifies the following:
// 1. When feature flag is not enabled, loading a encryptionConfig with KMSv2 should fail
// 2. When feature flag is enabled, loading a encryptionConfig with KMSv2 should work
// 3. When feature flag is disabled, loading a encryptionConfig with a non-v2 provider should work.
// without performing a storage migration, decryption of existing data encrypted with v2 should fail for Get and List operations.
// New data stored in etcd will no longer be encrypted using the external kms provider with v2 API.
// 4. when feature flag is re-enabled, loading a encryptionConfig with the same KMSv2 plugin from 2 should work,
// decryption of data encrypted with v2 should work
func TestKMSv2FeatureFlag(t *testing.T) {
	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`
	providerName := "kms-provider"
	pluginMock := kmsv2mock.NewBase64Plugin(t, "@kms-provider.sock")
	storageConfig := framework.SharedEtcd()

	// When feature flag is enabled, loading a encryptionConfig with KMSv1 and v2 should work
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()
	test, err := newTransformTest(t, encryptionConfig, false, "", storageConfig)
	if err != nil {
		t.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	defer func() {
		test.cleanUp()
	}()

	test.secret, err = test.createSecret(testSecret, testNamespace)
	if err != nil {
		t.Fatalf("Failed to create test secret, error: %v", err)
	}

	// Since Data Encryption Key (DEK) is randomly generated, we need to ask KMS Mock for it.
	plainTextDEKSource := pluginMock.LastEncryptRequest()

	secretETCDPath := test.getETCDPathForResource(test.storageConfig.Prefix, "", "secrets", test.secret.Name, test.secret.Namespace)
	rawEnvelope, err := test.getRawSecretFromETCD()
	if err != nil {
		t.Fatalf("failed to read %s from etcd: %v", secretETCDPath, err)
	}

	envelopeData := envelopekmsv2{
		providerName:       providerName,
		rawEnvelope:        rawEnvelope,
		plainTextDEKSource: plainTextDEKSource,
	}

	wantPrefix := envelopeData.prefix()
	if !bytes.HasPrefix(rawEnvelope, []byte(wantPrefix)) {
		t.Fatalf("expected secret to be prefixed with %s, but got %s", wantPrefix, rawEnvelope)
	}

	ctx := testContext(t)
	ciphertext, err := envelopeData.cipherTextDEKSource()
	if err != nil {
		t.Fatalf("failed to get ciphertext DEK from KMSv2 Plugin: %v", err)
	}
	decryptResponse, err := pluginMock.Decrypt(ctx, &kmsv2api.DecryptRequest{Uid: string(uuid.NewUUID()), Ciphertext: ciphertext})
	if err != nil {
		t.Fatalf("failed to decrypt DEK, %v", err)
	}
	dekPlainAsWouldBeSeenByETCD := decryptResponse.Plaintext

	if !bytes.Equal(plainTextDEKSource, dekPlainAsWouldBeSeenByETCD) {
		t.Fatalf("expected plainTextDEKSource %v to be passed to KMS Plugin, but got %s",
			plainTextDEKSource, dekPlainAsWouldBeSeenByETCD)
	}

	plainSecret, err := envelopeData.plainTextPayload(secretETCDPath)
	if err != nil {
		t.Fatalf("failed to transform from storage via AESGCM, err: %v", err)
	}

	if !strings.Contains(string(plainSecret), secretVal) {
		t.Fatalf("expected %q after decryption, but got %q", secretVal, string(plainSecret))
	}

	secretClient := test.restClient.CoreV1().Secrets(testNamespace)
	// Secrets should be un-enveloped on direct reads from Kube API Server.
	s, err := secretClient.Get(ctx, testSecret, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get Secret from %s, err: %v", testNamespace, err)
	}
	if secretVal != string(s.Data[secretKey]) {
		t.Fatalf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
	}
	test.shutdownAPIServer()

	// When KMSv2 feature flag is disabled, loading a encryptionConfig with a non-v2 provider should work. without performing a storage migration, decryption of existing data encrypted with v2 should fail for Get and List operations.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, false)()

	encryptionConfig1 := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`
	test, err = newTransformTest(t, encryptionConfig1, false, "", storageConfig)
	if err != nil {
		t.Fatalf("Failed to restart api server, error: %v", err)
	}

	_, err = test.createSecret("test2", testNamespace)
	if err != nil {
		t.Fatalf("Failed to create test secret, error: %v", err)
	}
	test.runResource(t, unSealWithCBCTransformer, aesCBCPrefix, "", "v1", "secrets", "test2", testNamespace)

	secretClient = test.restClient.CoreV1().Secrets(testNamespace)

	// Getting an old secret that was encrypted by another provider should fail
	_, err = secretClient.Get(ctx, testSecret, metav1.GetOptions{})
	if err == nil || !strings.Contains(err.Error(), "no matching prefix found") {
		t.Fatalf("using a new provider, get Secret %s from %s should return err containing: no matching prefix found. Got err: %v", testSecret, testNamespace, err)
	}
	// List all cluster wide secrets should fail
	_, err = test.restClient.CoreV1().Secrets("").List(ctx, metav1.ListOptions{})
	if err == nil || !strings.Contains(err.Error(), "no matching prefix found") {
		t.Fatalf("using a new provider, LIST all Secrets should return err containing: no matching prefix found. Got err: %v", err)
	}
	test.shutdownAPIServer()

	// when feature flag is re-enabled, loading a encryptionConfig with the same KMSv2 plugin before the restart should work, decryption of data encrypted with v2 should work
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()

	test, err = newTransformTest(t, encryptionConfig, false, "", storageConfig)
	if err != nil {
		t.Fatalf("Failed to restart api server, error: %v", err)
	}

	// Getting an old secret that was encrypted by the same plugin should not fail.
	s, err = test.restClient.CoreV1().Secrets(testNamespace).Get(
		ctx,
		testSecret,
		metav1.GetOptions{},
	)
	if err != nil {
		t.Fatalf("failed to read secret, err: %v", err)
	}
	if secretVal != string(s.Data[secretKey]) {
		t.Fatalf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
	}
	secretClient = test.restClient.CoreV1().Secrets(testNamespace)
	// Getting an old secret that was encrypted by another plugin should fail
	_, err = secretClient.Get(ctx, "test2", metav1.GetOptions{})
	if err == nil || !strings.Contains(err.Error(), "no matching prefix found") {
		t.Fatalf("after re-enabling feature gate, get test2 Secret from %s should return err containing: no matching prefix found. actual err: %v", testNamespace, err)
	}
}

var benchSecret *api.Secret

func BenchmarkKMSv2KDF(b *testing.B) {
	b.StopTimer()

	klog.SetOutput(io.Discard)
	klog.LogToStderr(false)

	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.KMSv2, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.KMSv2KDF, false)()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	b.Cleanup(cancel)

	ctx = request.WithNamespace(ctx, testNamespace)

	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`
	_ = kmsv2mock.NewBase64Plugin(b, "@kms-provider.sock")

	test, err := newTransformTest(b, encryptionConfig, false, "", nil)
	if err != nil {
		b.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	b.Cleanup(test.cleanUp)

	client := kubernetes.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)

	restOptionsGetter := getRESTOptionsGetterForSecrets(b, test)

	secretStorage, err := secretstore.NewREST(restOptionsGetter)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(secretStorage.Destroy)

	const dataLen = 1_000

	secrets := make([]*api.Secret, dataLen)

	for i := 0; i < dataLen; i++ {
		secrets[i] = &api.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-secret-%d", i),
				Namespace: testNamespace,
			},
			Data: map[string][]byte{
				"lots_of_data": bytes.Repeat([]byte{1, 3, 3, 7}, i*dataLen/4),
			},
		}
	}

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := secretStorage.DeleteCollection(ctx, noValidation, &metav1.DeleteOptions{}, nil)
		if err != nil {
			b.Fatal(err)
		}

		for i := 0; i < dataLen; i++ {
			out, err := secretStorage.Create(ctx, secrets[i], noValidation, &metav1.CreateOptions{})
			if err != nil {
				b.Fatal(err)
			}

			benchSecret = out.(*api.Secret)

			out, err = secretStorage.Get(ctx, benchSecret.Name, &metav1.GetOptions{})
			if err != nil {
				b.Fatal(err)
			}

			benchSecret = out.(*api.Secret)
		}
	}
	b.StopTimer()

	secretList, err := client.CoreV1().Secrets(testNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		b.Fatal(err)
	}

	if secretLen := len(secretList.Items); secretLen != dataLen {
		b.Errorf("unexpected secret len: want %d, got %d", dataLen, secretLen)
	}
}

func getRESTOptionsGetterForSecrets(t testing.TB, test *transformTest) generic.RESTOptionsGetter {
	t.Helper()

	s := test.kubeAPIServer.ServerOpts

	etcdConfigCopy := *s.Etcd
	etcdConfigCopy.SkipHealthEndpoints = true                     // avoid running health check go routines
	etcdConfigCopy.EncryptionProviderConfigAutomaticReload = true // hack to use DynamicTransformers in t.Cleanup below

	// mostly copied from BuildGenericConfig

	genericConfig := genericapiserver.NewConfig(legacyscheme.Codecs)

	genericConfig.MergedResourceConfig = controlplane.DefaultAPIResourceConfigSource()

	if err := s.APIEnablement.ApplyTo(genericConfig, controlplane.DefaultAPIResourceConfigSource(), legacyscheme.Scheme); err != nil {
		t.Fatal(err)
	}

	storageFactoryConfig := kubeapiserver.NewStorageFactoryConfig()
	storageFactoryConfig.APIResourceConfig = genericConfig.MergedResourceConfig
	storageFactory, err := storageFactoryConfig.Complete(&etcdConfigCopy).New()
	if err != nil {
		t.Fatal(err)
	}
	if err := etcdConfigCopy.ApplyWithStorageFactoryTo(storageFactory, genericConfig); err != nil {
		t.Fatal(err)
	}

	transformers, ok := genericConfig.ResourceTransformers.(*encryptionconfig.DynamicTransformers)
	if !ok {
		t.Fatalf("incorrect type for ResourceTransformers: %T", genericConfig.ResourceTransformers)
	}

	t.Cleanup(func() {
		// this is a hack to cause the existing transformers to shutdown
		transformers.Set(nil, nil, nil, 0)
		time.Sleep(10 * time.Second) // block this cleanup for longer than kmsCloseGracePeriod
	})

	if genericConfig.RESTOptionsGetter == nil {
		t.Fatal("not REST options found")
	}

	opts, err := genericConfig.RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: "", Resource: "secrets"})
	if err != nil {
		t.Fatal(err)
	}

	if err := runtime.CheckCodec(opts.StorageConfig.Codec, &api.Secret{},
		schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Secret"}); err != nil {
		t.Fatal(err)
	}

	return genericConfig.RESTOptionsGetter
}

func noValidation(_ context.Context, _ runtime.Object) error { return nil }

var benchRESTSecret *corev1.Secret

func BenchmarkKMSv2REST(b *testing.B) {
	b.StopTimer()

	klog.SetOutput(io.Discard)
	klog.LogToStderr(false)

	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.KMSv2, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.KMSv2KDF, false)()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	b.Cleanup(cancel)

	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
       apiVersion: v2
       name: kms-provider
       endpoint: unix:///@kms-provider.sock
`
	_ = kmsv2mock.NewBase64Plugin(b, "@kms-provider.sock")

	test, err := newTransformTest(b, encryptionConfig, false, "", nil)
	if err != nil {
		b.Fatalf("failed to start KUBE API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	b.Cleanup(test.cleanUp)

	client := kubernetes.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)

	const dataLen = 1_000

	secretStorage := client.CoreV1().Secrets(testNamespace)

	secrets := make([]*corev1.Secret, dataLen)

	for i := 0; i < dataLen; i++ {
		secrets[i] = &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-secret-%d", i),
				Namespace: testNamespace,
			},
			Data: map[string][]byte{
				"lots_of_data": bytes.Repeat([]byte{1, 3, 3, 7}, i*dataLen/4),
			},
		}
	}

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		err := secretStorage.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{})
		if err != nil {
			b.Fatal(err)
		}

		for i := 0; i < dataLen; i++ {
			out, err := secretStorage.Create(ctx, secrets[i], metav1.CreateOptions{})
			if err != nil {
				b.Fatal(err)
			}

			benchRESTSecret = out

			out, err = secretStorage.Get(ctx, benchRESTSecret.Name, metav1.GetOptions{})
			if err != nil {
				b.Fatal(err)
			}

			benchRESTSecret = out
		}
	}
	b.StopTimer()

	secretList, err := client.CoreV1().Secrets(testNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		b.Fatal(err)
	}

	if secretLen := len(secretList.Items); secretLen != dataLen {
		b.Errorf("unexpected secret len: want %d, got %d", dataLen, secretLen)
	}
}

func randomBool() bool { return utilrand.Int()%2 == 1 }
