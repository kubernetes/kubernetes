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
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

const (
	// Encryption configs targeting foos.cr.bar.com instead of secrets.
	// Same key material as the secret tests — we just change the resource identifier.
	crAESGCMConfigYAML = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - foos.cr.bar.com
    providers:
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`

	crAESCBCConfigYAML = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - foos.cr.bar.com
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`
)

var fooGVR = schema.GroupVersionResource{
	Group:    "cr.bar.com",
	Version:  "v1",
	Resource: "foos",
}

// TestCRAllowUnsafeMalformedObjectDeletionFeature mirrors
// TestAllowUnsafeMalformedObjectDeletionFeature but uses a Custom Resource
// (foos.cr.bar.com) instead of Secrets. This exercises the dynamic storage
// registry code path that is distinct from built-in resources.
func TestCRAllowUnsafeMalformedObjectDeletionFeature(t *testing.T) {
	tests := []struct {
		featureEnabled                         bool
		encryptionBrokenFn                     func(t *testing.T, got apierrors.APIStatus) bool
		corruptObjGetPreDelete                 verifier
		corruptObjDeleteWithoutOption          verifier
		corruptObjDeleteWithOption             verifier
		corruptObjDeleteWithOptionAndPrivilege verifier
		corruptObjGetPostDelete                verifier
		corruptObjGetFromListerCachePostDelete verifier
	}{
		{
			featureEnabled: true,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: StorageError: corrupt object") &&
					strings.Contains(got.Status().Message, "data from the storage is not transformable revision=0: no matching prefix found")
			},
			corruptObjGetPreDelete:        wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithoutOption: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithOption: wantAPIStatusError{
				reason:          metav1.StatusReasonForbidden,
				messageContains: `not permitted to do "unsafe-delete-ignore-read-errors"`,
			},
			corruptObjDeleteWithOptionAndPrivilege: wantNoError{},
			corruptObjGetPostDelete:                wantAPIStatusError{reason: metav1.StatusReasonNotFound},
			corruptObjGetFromListerCachePostDelete: wantAPIStatusError{reason: metav1.StatusReasonNotFound},
		},
		{
			featureEnabled: false,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: no matching prefix found")
			},
			corruptObjGetPreDelete:                 wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithoutOption:          wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithOption:             wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithOptionAndPrivilege: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjGetPostDelete:                wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjGetFromListerCachePostDelete: wantNoError{},
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf(
			"%s/%t", string(genericfeatures.AllowUnsafeMalformedObjectDeletion), tc.featureEnabled,
		), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(
				t, utilfeature.DefaultFeatureGate,
				genericfeatures.AllowUnsafeMalformedObjectDeletion, tc.featureEnabled,
			)

			test, err := newTransformTest(t, transformTestConfig{
				transformerConfigYAML: crAESGCMConfigYAML,
				reload:                true,
			})
			if err != nil {
				t.Fatalf("failed to setup test, error was %v", err)
			}
			defer test.cleanUp()

			// Register the foos CRD (only the first entry — foos.cr.bar.com is namespaced)
			etcd.CreateTestCRDs(
				t,
				apiextensionsclientset.NewForConfigOrDie(test.kubeAPIServer.ClientConfig),
				false,
				etcd.GetCustomResourceDefinitionData()[0],
			)

			// a) set up a distinct client for the test user with the least privileges
			testUser := "croc"
			testUserConfig := restclient.CopyConfig(test.kubeAPIServer.ClientConfig)
			testUserConfig.Impersonate.UserName = testUser
			testUserDynClient := dynamic.NewForConfigOrDie(testUserConfig)
			adminClient := test.restClient
			adminDynClient := dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)

			// b) grant the test user initial permissions on foos (not unsafe-delete yet)
			grantUserVerbsOnResource(
				t,
				adminClient,
				testUser,
				testNamespace,
				[]string{"create", "get", "delete", "update"},
				"cr.bar.com",
				"foos",
			)

			fooCorrupt := "foo-with-unsafe-delete"
			// c) create and delete the CR — no error expected
			createFooCR(t, testUserDynClient, fooCorrupt, testNamespace)
			err = testUserDynClient.Resource(fooGVR).
				Namespace(testNamespace).
				Delete(context.Background(), fooCorrupt, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to delete, have error: %v", testNamespace, fooCorrupt, err)
			}

			// d) re-create the CR
			fooObj := createFooCR(t, testUserDynClient, fooCorrupt, testNamespace)

			// e) update the CR with a finalizer
			if err := unstructured.SetNestedStringSlice(
				fooObj.Object,
				[]string{"test.k8s.io/fake"},
				"metadata", "finalizers",
			); err != nil {
				t.Fatalf("failed to set finalizers: %v", err)
			}
			fooObj, err = testUserDynClient.Resource(fooGVR).
				Namespace(testNamespace).
				Update(context.Background(), fooObj, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to add finalizer to the CR, error: %v", err)
			}

			// f) verify the CR is encrypted in etcd
			test.runResource(
				test.TContext,
				unSealWithGCMTransformer,
				aesGCMPrefix,
				"cr.bar.com",
				"v1",
				"foos",
				fooObj.GetName(),
				fooObj.GetNamespace(),
			)

			// g) set up a dynamic informer to track the CR in the cache
			factory := dynamicinformer.NewFilteredDynamicSharedInformerFactory(
				adminDynClient,
				time.Minute,
				testNamespace,
				nil,
			)
			informer := factory.ForResource(fooGVR)

			finalEvent := make(chan struct{})
			finalFooName := "final-foo"
			_, err = informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj any) {
					if accessor, err := meta.Accessor(obj); err == nil && accessor.GetName() == finalFooName {
						close(finalEvent)
					}
				},
			})
			if err != nil {
				t.Fatalf("unexpected error from AddEventHandler: %v", err)
			}

			lister := informer.Lister()
			informerCtx, informerCancel := context.WithCancel(test)
			defer informerCancel()
			factory.Start(informerCtx.Done())
			waitForSyncCtx, waitForSyncCancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
			defer waitForSyncCancel()
			if !cache.WaitForCacheSync(waitForSyncCtx.Done(), informer.Informer().HasSynced) {
				t.Fatalf("caches failed to sync")
			}

			// h) the CR should be readable from the cache
			if _, err := lister.ByNamespace(testNamespace).Get(fooCorrupt); err != nil {
				t.Errorf("unexpected failure getting the CR from lister cache: %v", err)
			}

			// i) break encryption by swapping config to AESCBC
			now := time.Now()
			encryptionConf := filepath.Join(test.configDir, encryptionConfigFileName)
			body, _ := os.ReadFile(encryptionConf)
			t.Logf("file before write: %s", body)
			if err := os.WriteFile(encryptionConf, []byte(crAESCBCConfigYAML), 0o644); err != nil {
				t.Fatalf("failed to write encryption config that's going to make decryption fail")
			}
			body, _ = os.ReadFile(encryptionConf)
			t.Logf("file after write: %s", body)

			// j) wait for the breaking changes to take effect
			// Dynamic encryption config reload takes ~60s.
			err = wait.PollUntilContextTimeout(t.Context(), 1*time.Second, 2*time.Minute, true, func(ctx context.Context) (done bool, err error) {
				_, err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Get(ctx, fooCorrupt, metav1.GetOptions{})
				var got apierrors.APIStatus
				if !errors.As(err, &got) {
					return false, nil
				}
				if done := tc.encryptionBrokenFn(t, got); done {
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Fatalf("encryption never broke: %v", err)
			}
			t.Logf("it took %s for the apiserver to reload the encryption config", time.Since(now))

			// k) create a new CR, and then delete it — should work fine
			fooNormal := "bar-with-normal-delete"
			createFooCR(t, testUserDynClient, fooNormal, testNamespace)
			err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Delete(context.Background(), fooNormal, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to delete, got error: %v", testNamespace, fooNormal, err)
			}

			// l) GET the corrupt CR — expect failure
			_, err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Get(context.Background(), fooCorrupt, metav1.GetOptions{})
			tc.corruptObjGetPreDelete.verify(t, err)
			if _, err := lister.ByNamespace(testNamespace).Get(fooCorrupt); err != nil {
				t.Errorf("unexpected failure getting the CR from the cache: %v", err)
			}

			// m) normal delete — expect error
			err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Delete(context.Background(), fooCorrupt, metav1.DeleteOptions{})
			tc.corruptObjDeleteWithoutOption.verify(t, err)

			// n) delete with unsafe option but no privilege — expect forbidden or internal error
			options := metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To(true),
			}
			err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Delete(context.Background(), fooCorrupt, options)
			tc.corruptObjDeleteWithOption.verify(t, err)

			// o) grant the test user the unsafe-delete-ignore-read-errors verb
			grantUserVerbsOnResource(t, adminClient, testUser, testNamespace, []string{"unsafe-delete-ignore-read-errors"}, "cr.bar.com", "foos")

			// p) try unsafe delete again — should succeed when feature enabled
			err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Delete(context.Background(), fooCorrupt, options)
			tc.corruptObjDeleteWithOptionAndPrivilege.verify(t, err)

			// q) final GET should return NotFound after deletion (when feature enabled)
			_, err = testUserDynClient.Resource(fooGVR).Namespace(testNamespace).Get(context.Background(), fooCorrupt, metav1.GetOptions{})
			tc.corruptObjGetPostDelete.verify(t, err)

			// r) create the final CR and wait for the informer to catch up
			createFooCR(t, adminDynClient, finalFooName, testNamespace)
			select {
			case <-finalEvent:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("timed out waiting for the informer to catch up")
			}

			// s) read the corrupt object from the cache — should be gone when feature enabled
			_, err = lister.ByNamespace(testNamespace).Get(fooCorrupt)
			tc.corruptObjGetFromListerCachePostDelete.verify(t, err)
		})
	}
}

// TestCRListCorruptObjects mirrors TestListCorruptObjects but uses Custom
// Resources (foos.cr.bar.com). It verifies that LIST properly reports corrupt
// CR objects when the AllowUnsafeMalformedObjectDeletion feature is enabled.
func TestCRListCorruptObjects(t *testing.T) {
	foos := []string{"corrupt-a", "corrupt-b", "corrupt-c"}

	tests := []struct {
		featureEnabled     bool
		foos               []string
		encryptionBrokenFn func(t *testing.T, got apierrors.APIStatus)
		listAfter          verifier
	}{
		{
			foos:           foos,
			featureEnabled: true,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) {
				status := got.Status()
				if status.Reason != metav1.StatusReasonInternalError {
					t.Errorf("Invalid reason, got: %q, want: %q", status.Reason, metav1.StatusReasonInternalError)
				}
				corruptObjectMsg := "Internal error occurred: StorageError: corrupt object"
				if !strings.Contains(status.Message, corruptObjectMsg) {
					t.Errorf("Message should include %q, but got: %q", corruptObjectMsg, status.Message)
				}
				messageAuthenticationFailedMsg := "data from the storage is not transformable revision=0: cipher: message authentication failed"
				if !strings.Contains(status.Message, messageAuthenticationFailedMsg) {
					t.Errorf("Message should include %q, but got: %q", messageAuthenticationFailedMsg, status.Message)
				}
			},
			listAfter: wantAPIStatusError{
				reason:          metav1.StatusReasonStoreReadError,
				messageContains: "failed to read one or more foos.cr.bar.com from the storage",
				more: func(t *testing.T, err apierrors.APIStatus) {
					t.Helper()

					details := err.Status().Details
					if details == nil {
						t.Errorf("expected Details in APIStatus, but got: %#v", err)
						return
					}
					if want, got := len(foos), len(details.Causes); want != got {
						t.Errorf("expected to have %d in APIStatus, but got: %d", want, got)
					}
					for _, cause := range details.Causes {
						if want, got := metav1.CauseTypeUnexpectedServerResponse, cause.Type; want != got {
							t.Errorf("expected to cause type to be %s, but got: %s", want, got)
						}
					}
					for _, want := range foos {
						var found bool
						for _, got := range details.Causes {
							if strings.HasSuffix(got.Field, want) {
								found = true
								break
							}
						}
						if !found {
							t.Errorf("want key: %q in the Fields: %#v", want, details.Causes)
						}
					}
				},
			},
		},
		{
			foos:           foos,
			featureEnabled: false,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) {
				status := got.Status()
				if status.Reason != metav1.StatusReasonInternalError {
					t.Errorf("Invalid reason, got: %q, want: %q", status.Reason, metav1.StatusReasonInternalError)
				}
				noMatchingPrefixFoundMsg := "Internal error occurred: cipher: message authentication failed"
				if !strings.Contains(status.Message, noMatchingPrefixFoundMsg) {
					t.Errorf("Message should include %q, but got: %q", noMatchingPrefixFoundMsg, status.Message)
				}
			},
			listAfter: wantAPIStatusError{
				reason:          metav1.StatusReasonInternalError,
				messageContains: "unable to transform key",
			},
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf(
			"%s/%t", string(genericfeatures.AllowUnsafeMalformedObjectDeletion), tc.featureEnabled,
		), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(
				t, utilfeature.DefaultFeatureGate,
				genericfeatures.AllowUnsafeMalformedObjectDeletion, tc.featureEnabled,
			)
			storageConfig := framework.SharedEtcd()
			test, err := newTransformTest(t, transformTestConfig{
				transformerConfigYAML: crAESGCMConfigYAML,
				reload:                true,
				storageConfig:         storageConfig,
			})
			if err != nil {
				t.Fatalf("failed to setup test, error was %v", err)
			}
			defer test.cleanUp()
			ctx := context.Background()

			// Register the foos CRD
			etcd.CreateTestCRDs(
				t,
				apiextensionsclientset.NewForConfigOrDie(test.kubeAPIServer.ClientConfig),
				false,
				etcd.GetCustomResourceDefinitionData()[0],
			)

			adminDynClient := dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)

			// a) create a number of Foo CRs in the test namespace
			for _, name := range tc.foos {
				createFooCR(t, adminDynClient, name, testNamespace)
			}

			// b) list the CRs before we break encryption
			result, err := adminDynClient.Resource(fooGVR).Namespace(testNamespace).List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("listing foos failed unexpectedly with: %v", err)
			}
			if want, got := len(tc.foos), len(result.Items); got < want {
				t.Fatalf("expected at least %d foos, but got: %d", want, got)
			}

			// c) corrupt the etcd data directly — truncate each value by one byte
			client, err := clientv3.New(clientv3.Config{Endpoints: storageConfig.Transport.ServerList})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := client.Close(); err != nil {
					t.Fatal(err)
				}
			}()
			resp, err := client.Get(ctx, "/"+storageConfig.Prefix+"/cr.bar.com/foos/", clientv3.WithPrefix())
			if err != nil {
				t.Fatal(err)
			}
			if len(resp.Kvs) != len(tc.foos) {
				t.Fatalf("Expected %d number of keys, got: %d", len(tc.foos), len(resp.Kvs))
			}
			for _, kv := range resp.Kvs {
				_, err = client.Put(ctx, string(kv.Key), string(kv.Value)[:len(kv.Value)-1])
				if err != nil {
					t.Fatal(err)
				}
			}

			// d) verify GET on a single corrupt CR returns the expected error
			_, err = adminDynClient.Resource(fooGVR).Namespace(testNamespace).Get(ctx, tc.foos[0], metav1.GetOptions{})
			if err != nil {
				t.Logf("get returned error: %#v message: %s", err, err.Error())
			}

			var got apierrors.APIStatus
			if !errors.As(err, &got) {
				t.Fatalf("encryption never broke: %v", err)
			}
			tc.encryptionBrokenFn(t, got)

			// e) LIST should return expected error
			_, err = adminDynClient.Resource(fooGVR).Namespace(testNamespace).List(ctx, metav1.ListOptions{})
			tc.listAfter.verify(t, err)
		})
	}
}

// createFooCR creates a Foo custom resource in the given namespace using the dynamic client.
func createFooCR(t *testing.T, dynamicClient dynamic.Interface, name, namespace string) *unstructured.Unstructured {
	t.Helper()

	obj := &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": "cr.bar.com/v1",
			"kind":       "Foo",
			"metadata": map[string]any{
				"name":      name,
				"namespace": namespace,
			},
			"color": "blue",
		},
	}

	created, err := dynamicClient.Resource(fooGVR).
		Namespace(namespace).
		Create(context.TODO(), obj, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create Foo CR %s/%s: %v", namespace, name, err)
	}

	return created
}
