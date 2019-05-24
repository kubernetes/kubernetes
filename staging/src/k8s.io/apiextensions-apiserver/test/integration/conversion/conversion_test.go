/*
Copyright 2019 The Kubernetes Authors.

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

package conversion

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	serveroptions "k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	etcd3watcher "k8s.io/apiserver/pkg/storage/etcd3"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apiextensions-apiserver/test/integration/storage"
)

type Checker func(t *testing.T, ctc *conversionTestContext)

func checks(checkers ...Checker) []Checker {
	return checkers
}

func TestWebhookConverter(t *testing.T) {
	testWebhookConverter(t, false)
}

func TestWebhookConverterWithPruning(t *testing.T) {
	testWebhookConverter(t, true)
}

func testWebhookConverter(t *testing.T, pruning bool) {
	tests := []struct {
		group   string
		handler http.Handler
		checks  []Checker
	}{
		{
			group:   "noop-converter",
			handler: NewObjectConverterWebhookHandler(t, noopConverter),
			checks:  checks(validateStorageVersion, validateServed, validateMixedStorageVersions("v1alpha1", "v1beta1")), // no v1beta2 as the schema differs
		},
		{
			group:   "nontrivial-converter",
			handler: NewObjectConverterWebhookHandler(t, nontrivialConverter),
			checks:  checks(validateStorageVersion, validateServed, validateMixedStorageVersions("v1alpha1", "v1beta1", "v1beta2"), validateNonTrivialConverted, validateNonTrivialConvertedList, validateStoragePruning),
		},
		{
			group:   "empty-response",
			handler: NewReviewWebhookHandler(t, emptyResponseConverter),
			checks:  checks(expectConversionFailureMessage("empty-response", "expected 1 converted objects")),
		},
		{
			group:   "failure-message",
			handler: NewReviewWebhookHandler(t, failureResponseConverter("custom webhook conversion error")),
			checks:  checks(expectConversionFailureMessage("failure-message", "custom webhook conversion error")),
		},
	}

	// TODO: Added for integration testing of conversion webhooks, where decode errors due to conversion webhook failures need to be tested.
	// Maybe we should identify conversion webhook related errors in decoding to avoid triggering this? Or maybe having this special casing
	// of test cases in production code should be removed?
	etcd3watcher.TestOnlySetFatalOnDecodeError(false)
	defer etcd3watcher.TestOnlySetFatalOnDecodeError(true)

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CustomResourceWebhookConversion, true)()
	tearDown, config, options, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		tearDown()
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		tearDown()
		t.Fatal(err)
	}
	defer tearDown()

	crd := multiVersionFixture.DeepCopy()
	crd.Spec.PreserveUnknownFields = pointer.BoolPtr(!pruning)

	RESTOptionsGetter := serveroptions.NewCRDRESTOptionsGetter(*options.RecommendedOptions.Etcd)
	restOptions, err := RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural})
	if err != nil {
		t.Fatal(err)
	}
	etcdClient, _, err := storage.GetEtcdClients(restOptions.StorageConfig.Transport)
	if err != nil {
		t.Fatal(err)
	}
	defer etcdClient.Close()

	etcdObjectReader := storage.NewEtcdObjectReader(etcdClient, &restOptions, crd)
	ctcTearDown, ctc := newConversionTestContext(t, apiExtensionsClient, dynamicClient, etcdObjectReader, crd)
	defer ctcTearDown()

	// read only object to read at a different version than stored when we need to force conversion
	marker, err := ctc.versionedClient("marker", "v1beta1").Create(newConversionMultiVersionFixture("marker", "marker", "v1beta1"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range tests {
		t.Run(test.group, func(t *testing.T) {
			upCh, handler := closeOnCall(test.handler)
			tearDown, webhookClientConfig, err := StartConversionWebhookServer(handler)
			if err != nil {
				t.Fatal(err)
			}
			defer tearDown()

			ctc.setConversionWebhook(t, webhookClientConfig)
			defer ctc.removeConversionWebhook(t)

			// wait until new webhook is called the first time
			if err := wait.PollImmediate(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
				_, err := ctc.versionedClient(marker.GetNamespace(), "v1alpha1").Get(marker.GetName(), metav1.GetOptions{})
				select {
				case <-upCh:
					return true, nil
				default:
					t.Logf("Waiting for webhook to become effective, getting marker object: %v", err)
					return false, nil
				}
			}); err != nil {
				t.Fatal(err)
			}

			for i, checkFn := range test.checks {
				name := fmt.Sprintf("check-%d", i)
				t.Run(name, func(t *testing.T) {
					defer ctc.setAndWaitStorageVersion(t, "v1beta1")
					ctc.namespace = fmt.Sprintf("webhook-conversion-%s-%s", test.group, name)
					checkFn(t, ctc)
				})
			}
		})
	}
}

func validateStorageVersion(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace

	for _, version := range ctc.crd.Spec.Versions {
		t.Run(version.Name, func(t *testing.T) {
			name := "storageversion-" + version.Name
			client := ctc.versionedClient(ns, version.Name)
			obj, err := client.Create(newConversionMultiVersionFixture(ns, name, version.Name), metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			ctc.setAndWaitStorageVersion(t, "v1beta2")

			obj, err = client.Get(obj.GetName(), metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}

			ctc.setAndWaitStorageVersion(t, "v1beta1")
		})
	}
}

// validateMixedStorageVersions ensures that identical custom resources written at different storage versions
// are readable and remain the same.
func validateMixedStorageVersions(versions ...string) func(t *testing.T, ctc *conversionTestContext) {
	return func(t *testing.T, ctc *conversionTestContext) {
		ns := ctc.namespace
		clients := ctc.versionedClients(ns)

		// Create CRs at all storage versions
		objNames := []string{}
		for _, version := range versions {
			ctc.setAndWaitStorageVersion(t, version)

			name := "mixedstorage-stored-as-" + version
			obj, err := clients[version].Create(newConversionMultiVersionFixture(ns, name, version), metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			objNames = append(objNames, obj.GetName())
		}

		// Ensure copies of an object have the same fields and values at each custom resource definition version regardless of storage version
		for clientVersion, client := range clients {
			t.Run(clientVersion, func(t *testing.T) {
				o1, err := client.Get(objNames[0], metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				for _, objName := range objNames[1:] {
					o2, err := client.Get(objName, metav1.GetOptions{})
					if err != nil {
						t.Fatal(err)
					}

					// ignore metadata for comparison purposes
					delete(o1.Object, "metadata")
					delete(o2.Object, "metadata")
					if !reflect.DeepEqual(o1.Object, o2.Object) {
						t.Errorf("Expected custom resource to be same regardless of which storage version is used to create, but got: %s", cmp.Diff(o1, o2))
					}
				}
			})
		}
	}
}

func validateServed(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace

	for _, version := range ctc.crd.Spec.Versions {
		t.Run(version.Name, func(t *testing.T) {
			name := "served-" + version.Name
			client := ctc.versionedClient(ns, version.Name)
			obj, err := client.Create(newConversionMultiVersionFixture(ns, name, version.Name), metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			ctc.setServed(t, version.Name, false)
			ctc.waitForServed(t, version.Name, false, client, obj)
			ctc.setServed(t, version.Name, true)
			ctc.waitForServed(t, version.Name, true, client, obj)
		})
	}
}

func validateNonTrivialConverted(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace

	for _, createVersion := range ctc.crd.Spec.Versions {
		t.Run(fmt.Sprintf("getting objects created as %s", createVersion.Name), func(t *testing.T) {
			name := "converted-" + createVersion.Name
			client := ctc.versionedClient(ns, createVersion.Name)

			fixture := newConversionMultiVersionFixture(ns, name, createVersion.Name)
			if !*ctc.crd.Spec.PreserveUnknownFields {
				if err := unstructured.SetNestedField(fixture.Object, "foo", "garbage"); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := client.Create(fixture, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}

			// verify that the right, pruned version is in storage
			obj, err := ctc.etcdObjectReader.GetStoredCustomResource(ns, name)
			if err != nil {
				t.Fatal(err)
			}
			verifyMultiVersionObject(t, "v1beta1", obj)

			for _, getVersion := range ctc.crd.Spec.Versions {
				client := ctc.versionedClient(ns, getVersion.Name)
				obj, err := client.Get(name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				verifyMultiVersionObject(t, getVersion.Name, obj)
			}
		})
	}
}

func validateNonTrivialConvertedList(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace + "-list"

	names := sets.String{}
	for _, createVersion := range ctc.crd.Spec.Versions {
		name := "converted-" + createVersion.Name
		client := ctc.versionedClient(ns, createVersion.Name)
		fixture := newConversionMultiVersionFixture(ns, name, createVersion.Name)
		if !*ctc.crd.Spec.PreserveUnknownFields {
			if err := unstructured.SetNestedField(fixture.Object, "foo", "garbage"); err != nil {
				t.Fatal(err)
			}
		}
		_, err := client.Create(fixture, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		names.Insert(name)
	}

	for _, listVersion := range ctc.crd.Spec.Versions {
		t.Run(fmt.Sprintf("listing objects as %s", listVersion.Name), func(t *testing.T) {
			client := ctc.versionedClient(ns, listVersion.Name)
			obj, err := client.List(metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if len(obj.Items) != len(ctc.crd.Spec.Versions) {
				t.Fatal("unexpected number of items")
			}
			foundNames := sets.String{}
			for _, u := range obj.Items {
				foundNames.Insert(u.GetName())
				verifyMultiVersionObject(t, listVersion.Name, &u)
			}
			if !foundNames.Equal(names) {
				t.Errorf("unexpected set of returned items: %s", foundNames.Difference(names))
			}
		})
	}
}

func validateStoragePruning(t *testing.T, ctc *conversionTestContext) {
	if *ctc.crd.Spec.PreserveUnknownFields {
		return
	}

	ns := ctc.namespace

	for _, createVersion := range ctc.crd.Spec.Versions {
		t.Run(fmt.Sprintf("getting objects created as %s", createVersion.Name), func(t *testing.T) {
			name := "storagepruning-" + createVersion.Name
			client := ctc.versionedClient(ns, createVersion.Name)

			fixture := newConversionMultiVersionFixture(ns, name, createVersion.Name)
			if err := unstructured.SetNestedField(fixture.Object, "foo", "garbage"); err != nil {
				t.Fatal(err)
			}
			_, err := client.Create(fixture, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// verify that the right, pruned version is in storage
			obj, err := ctc.etcdObjectReader.GetStoredCustomResource(ns, name)
			if err != nil {
				t.Fatal(err)
			}
			verifyMultiVersionObject(t, "v1beta1", obj)

			// add garbage and set a label
			if err := unstructured.SetNestedField(obj.Object, "foo", "garbage"); err != nil {
				t.Fatal(err)
			}
			labels := obj.GetLabels()
			if labels == nil {
				labels = map[string]string{}
			}
			labels["mutated"] = "true"
			obj.SetLabels(labels)
			if err := ctc.etcdObjectReader.SetStoredCustomResource(ns, name, obj); err != nil {
				t.Fatal(err)
			}

			for _, getVersion := range ctc.crd.Spec.Versions {
				client := ctc.versionedClient(ns, getVersion.Name)
				obj, err := client.Get(name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}

				// check that the direct mutation in etcd worked
				labels := obj.GetLabels()
				if labels["mutated"] != "true" {
					t.Errorf("expected object %s in version %s to have label 'mutated=true'", name, getVersion.Name)
				}

				verifyMultiVersionObject(t, getVersion.Name, obj)
			}
		})
	}
}

func expectConversionFailureMessage(id, message string) func(t *testing.T, ctc *conversionTestContext) {
	return func(t *testing.T, ctc *conversionTestContext) {
		ns := ctc.namespace
		clients := ctc.versionedClients(ns)
		var err error
		// storage version is v1beta1, so this skips conversion
		obj, err := clients["v1beta1"].Create(newConversionMultiVersionFixture(ns, id, "v1beta1"), metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		for _, verb := range []string{"get", "list", "create", "udpate", "patch", "delete", "deletecollection"} {
			t.Run(verb, func(t *testing.T) {
				switch verb {
				case "get":
					_, err = clients["v1beta2"].Get(obj.GetName(), metav1.GetOptions{})
				case "list":
					_, err = clients["v1beta2"].List(metav1.ListOptions{})
				case "create":
					_, err = clients["v1beta2"].Create(newConversionMultiVersionFixture(ns, id, "v1beta2"), metav1.CreateOptions{})
				case "update":
					_, err = clients["v1beta2"].Update(obj, metav1.UpdateOptions{})
				case "patch":
					_, err = clients["v1beta2"].Patch(obj.GetName(), types.MergePatchType, []byte(`{"metadata":{"annotations":{"patch":"true"}}}`), metav1.PatchOptions{})
				case "delete":
					err = clients["v1beta2"].Delete(obj.GetName(), &metav1.DeleteOptions{})
				case "deletecollection":
					err = clients["v1beta2"].DeleteCollection(&metav1.DeleteOptions{}, metav1.ListOptions{})
				}

				if err == nil {
					t.Errorf("expected error with message %s, but got no error", message)
				} else if !strings.Contains(err.Error(), message) {
					t.Errorf("expected error with message %s, but got %v", message, err)
				}
			})
		}
		for _, subresource := range []string{"status", "scale"} {
			for _, verb := range []string{"get", "udpate", "patch"} {
				t.Run(fmt.Sprintf("%s-%s", subresource, verb), func(t *testing.T) {
					switch verb {
					case "create":
						_, err = clients["v1beta2"].Create(newConversionMultiVersionFixture(ns, id, "v1beta2"), metav1.CreateOptions{}, subresource)
					case "update":
						_, err = clients["v1beta2"].Update(obj, metav1.UpdateOptions{}, subresource)
					case "patch":
						_, err = clients["v1beta2"].Patch(obj.GetName(), types.MergePatchType, []byte(`{"metadata":{"annotations":{"patch":"true"}}}`), metav1.PatchOptions{}, subresource)
					}

					if err == nil {
						t.Errorf("expected error with message %s, but got no error", message)
					} else if !strings.Contains(err.Error(), message) {
						t.Errorf("expected error with message %s, but got %v", message, err)
					}
				})
			}
		}
	}
}

func noopConverter(desiredAPIVersion string, obj runtime.RawExtension) (runtime.RawExtension, error) {
	u := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := json.Unmarshal(obj.Raw, u); err != nil {
		return runtime.RawExtension{}, fmt.Errorf("failed to deserialize object: %s with error: %v", string(obj.Raw), err)
	}
	u.Object["apiVersion"] = desiredAPIVersion
	raw, err := json.Marshal(u)
	if err != nil {
		return runtime.RawExtension{}, fmt.Errorf("failed to serialize object: %v with error: %v", u, err)
	}
	return runtime.RawExtension{Raw: raw}, nil
}

func emptyResponseConverter(review apiextensionsv1beta1.ConversionReview) (apiextensionsv1beta1.ConversionReview, error) {
	review.Response = &apiextensionsv1beta1.ConversionResponse{
		UID:              review.Request.UID,
		ConvertedObjects: []runtime.RawExtension{},
		Result:           metav1.Status{Status: "Success"},
	}
	return review, nil
}

func failureResponseConverter(message string) func(review apiextensionsv1beta1.ConversionReview) (apiextensionsv1beta1.ConversionReview, error) {
	return func(review apiextensionsv1beta1.ConversionReview) (apiextensionsv1beta1.ConversionReview, error) {
		review.Response = &apiextensionsv1beta1.ConversionResponse{
			UID:              review.Request.UID,
			ConvertedObjects: []runtime.RawExtension{},
			Result:           metav1.Status{Message: message, Status: "Failure"},
		}
		return review, nil
	}
}

func nontrivialConverter(desiredAPIVersion string, obj runtime.RawExtension) (runtime.RawExtension, error) {
	u := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := json.Unmarshal(obj.Raw, u); err != nil {
		return runtime.RawExtension{}, fmt.Errorf("failed to deserialize object: %s with error: %v", string(obj.Raw), err)
	}

	currentAPIVersion := u.GetAPIVersion()

	if currentAPIVersion == "stable.example.com/v1beta2" && (desiredAPIVersion == "stable.example.com/v1alpha1" || desiredAPIVersion == "stable.example.com/v1beta1") {
		u.Object["num"] = u.Object["numv2"]
		u.Object["content"] = u.Object["contentv2"]
		delete(u.Object, "numv2")
		delete(u.Object, "contentv2")
	} else if (currentAPIVersion == "stable.example.com/v1alpha1" || currentAPIVersion == "stable.example.com/v1beta1") && desiredAPIVersion == "stable.example.com/v1beta2" {
		u.Object["numv2"] = u.Object["num"]
		u.Object["contentv2"] = u.Object["content"]
		delete(u.Object, "num")
		delete(u.Object, "content")
	} else if currentAPIVersion == "stable.example.com/v1alpha1" && desiredAPIVersion == "stable.example.com/v1beta1" {
		// same schema
	} else if currentAPIVersion == "stable.example.com/v1beta1" && desiredAPIVersion == "stable.example.com/v1alpha1" {
		// same schema
	} else if currentAPIVersion != desiredAPIVersion {
		return runtime.RawExtension{}, fmt.Errorf("cannot convert from %s to %s", currentAPIVersion, desiredAPIVersion)
	}
	u.Object["apiVersion"] = desiredAPIVersion
	raw, err := json.Marshal(u)
	if err != nil {
		return runtime.RawExtension{}, fmt.Errorf("failed to serialize object: %v with error: %v", u, err)
	}
	return runtime.RawExtension{Raw: raw}, nil
}

func newConversionTestContext(t *testing.T, apiExtensionsClient clientset.Interface, dynamicClient dynamic.Interface, etcdObjectReader *storage.EtcdObjectReader, crd *apiextensionsv1beta1.CustomResourceDefinition) (func(), *conversionTestContext) {
	crd, err := fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionsClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	tearDown := func() {
		if err := fixtures.DeleteCustomResourceDefinition(crd, apiExtensionsClient); err != nil {
			t.Fatal(err)
		}
	}

	return tearDown, &conversionTestContext{apiExtensionsClient: apiExtensionsClient, dynamicClient: dynamicClient, crd: crd, etcdObjectReader: etcdObjectReader}
}

type conversionTestContext struct {
	namespace           string
	apiExtensionsClient clientset.Interface
	dynamicClient       dynamic.Interface
	options             *options.CustomResourceDefinitionsServerOptions
	crd                 *apiextensionsv1beta1.CustomResourceDefinition
	etcdObjectReader    *storage.EtcdObjectReader
}

func (c *conversionTestContext) versionedClient(ns string, version string) dynamic.ResourceInterface {
	gvr := schema.GroupVersionResource{Group: c.crd.Spec.Group, Version: version, Resource: c.crd.Spec.Names.Plural}
	if c.crd.Spec.Scope != apiextensionsv1beta1.ClusterScoped {
		return c.dynamicClient.Resource(gvr).Namespace(ns)
	}
	return c.dynamicClient.Resource(gvr)
}

func (c *conversionTestContext) versionedClients(ns string) map[string]dynamic.ResourceInterface {
	ret := map[string]dynamic.ResourceInterface{}
	for _, v := range c.crd.Spec.Versions {
		ret[v.Name] = c.versionedClient(ns, v.Name)
	}
	return ret
}

func (c *conversionTestContext) setConversionWebhook(t *testing.T, webhookClientConfig *apiextensionsv1beta1.WebhookClientConfig) {
	crd, err := c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(c.crd.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	crd.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{
		Strategy:            apiextensionsv1beta1.WebhookConverter,
		WebhookClientConfig: webhookClientConfig,
	}
	crd, err = c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd)
	if err != nil {
		t.Fatal(err)
	}
	c.crd = crd

}

func (c *conversionTestContext) removeConversionWebhook(t *testing.T) {
	crd, err := c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(c.crd.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	crd.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{
		Strategy: apiextensionsv1beta1.NoneConverter,
	}

	crd, err = c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd)
	if err != nil {
		t.Fatal(err)
	}
	c.crd = crd
}

func (c *conversionTestContext) setAndWaitStorageVersion(t *testing.T, version string) {
	c.setStorageVersion(t, version)

	// create probe object. Version should be the default one to avoid webhook calls during test setup.
	client := c.versionedClient("probe", "v1beta1")
	name := fmt.Sprintf("probe-%v", uuid.NewUUID())
	storageProbe, err := client.Create(newConversionMultiVersionFixture("probe", name, "v1beta1"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// update object continuously and wait for etcd to have the target storage version.
	c.waitForStorageVersion(t, version, c.versionedClient(storageProbe.GetNamespace(), "v1beta1"), storageProbe)

	err = client.Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}
}

func (c *conversionTestContext) setStorageVersion(t *testing.T, version string) {
	crd, err := c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(c.crd.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range crd.Spec.Versions {
		crd.Spec.Versions[i].Storage = v.Name == version
	}
	crd, err = c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd)
	if err != nil {
		t.Fatal(err)
	}
	c.crd = crd
}

func (c *conversionTestContext) waitForStorageVersion(t *testing.T, version string, versionedClient dynamic.ResourceInterface, obj *unstructured.Unstructured) *unstructured.Unstructured {
	if err := c.etcdObjectReader.WaitForStorageVersion(version, obj.GetNamespace(), obj.GetName(), 30*time.Second, func() {
		if _, err := versionedClient.Patch(obj.GetName(), types.MergePatchType, []byte(`{}`), metav1.PatchOptions{}); err != nil {
			t.Fatalf("failed to update object: %v", err)
		}
	}); err != nil {
		t.Fatalf("failed waiting for storage version %s: %v", version, err)
	}

	t.Logf("Effective storage version: %s", version)

	return obj
}

func (c *conversionTestContext) setServed(t *testing.T, version string, served bool) {
	crd, err := c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(c.crd.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range crd.Spec.Versions {
		if v.Name == version {
			crd.Spec.Versions[i].Served = served
		}
	}
	crd, err = c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd)
	if err != nil {
		t.Fatal(err)
	}
	c.crd = crd
}

func (c *conversionTestContext) waitForServed(t *testing.T, version string, served bool, versionedClient dynamic.ResourceInterface, obj *unstructured.Unstructured) {
	timeout := 30 * time.Second
	waitCh := time.After(timeout)
	for {
		obj, err := versionedClient.Get(obj.GetName(), metav1.GetOptions{})
		if (err == nil && served) || (errors.IsNotFound(err) && served == false) {
			return
		}
		select {
		case <-waitCh:
			t.Fatalf("Timed out after %v waiting for CRD served=%t for version %s for %v. Last error: %v", timeout, served, version, obj, err)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

var multiVersionFixture = &apiextensionsv1beta1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "multiversion.stable.example.com"},
	Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
		Group:   "stable.example.com",
		Version: "v1beta1",
		Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
			Plural:     "multiversion",
			Singular:   "multiversion",
			Kind:       "MultiVersion",
			ShortNames: []string{"mv"},
			ListKind:   "MultiVersionList",
			Categories: []string{"all"},
		},
		Scope: apiextensionsv1beta1.NamespaceScoped,
		Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{
			{
				// storage version, same schema as v1alpha1
				Name:    "v1beta1",
				Served:  true,
				Storage: true,
				Schema: &apiextensionsv1beta1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
							"content": {
								Type: "object",
								Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
									"key": {Type: "string"},
								},
							},
							"num": {
								Type: "object",
								Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
									"num1": {Type: "integer"},
									"num2": {Type: "integer"},
								},
							},
						},
					},
				},
			},
			{
				// same schema as v1beta1
				Name:    "v1alpha1",
				Served:  true,
				Storage: false,
				Schema: &apiextensionsv1beta1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
							"content": {
								Type: "object",
								Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
									"key": {Type: "string"},
								},
							},
							"num": {
								Type: "object",
								Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
									"num1": {Type: "integer"},
									"num2": {Type: "integer"},
								},
							},
						},
					},
				},
			},
			{
				// different schema than v1beta1 and v1alpha1
				Name:    "v1beta2",
				Served:  true,
				Storage: false,
				Schema: &apiextensionsv1beta1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
							"contentv2": {
								Type: "object",
								Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
									"key": {Type: "string"},
								},
							},
							"numv2": {
								Type: "object",
								Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
									"num1": {Type: "integer"},
									"num2": {Type: "integer"},
								},
							},
						},
					},
				},
			},
		},
		Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
			Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
			Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
				SpecReplicasPath:   ".spec.num.num1",
				StatusReplicasPath: ".status.num.num2",
			},
		},
	},
}

func newConversionMultiVersionFixture(namespace, name, version string) *unstructured.Unstructured {
	u := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "stable.example.com/" + version,
			"kind":       "MultiVersion",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
		},
	}

	switch version {
	case "v1alpha1":
		u.Object["content"] = map[string]interface{}{
			"key": "value",
		}
		u.Object["num"] = map[string]interface{}{
			"num1": int64(1),
			"num2": int64(1000000),
		}
	case "v1beta1":
		u.Object["content"] = map[string]interface{}{
			"key": "value",
		}
		u.Object["num"] = map[string]interface{}{
			"num1": int64(1),
			"num2": int64(1000000),
		}
	case "v1beta2":
		u.Object["contentv2"] = map[string]interface{}{
			"key": "value",
		}
		u.Object["numv2"] = map[string]interface{}{
			"num1": int64(1),
			"num2": int64(1000000),
		}
	default:
		panic(fmt.Sprintf("unknown version %s", version))
	}

	return u
}

func verifyMultiVersionObject(t *testing.T, v string, obj *unstructured.Unstructured) {
	j := runtime.DeepCopyJSON(obj.Object)

	if expected := "stable.example.com/" + v; obj.GetAPIVersion() != expected {
		t.Errorf("unexpected apiVersion %q, expected %q", obj.GetAPIVersion(), expected)
		return
	}

	delete(j, "metadata")

	var expected = map[string]map[string]interface{}{
		"v1alpha1": {
			"apiVersion": "stable.example.com/v1alpha1",
			"kind":       "MultiVersion",
			"content": map[string]interface{}{
				"key": "value",
			},
			"num": map[string]interface{}{
				"num1": int64(1),
				"num2": int64(1000000),
			},
		},
		"v1beta1": {
			"apiVersion": "stable.example.com/v1beta1",
			"kind":       "MultiVersion",
			"content": map[string]interface{}{
				"key": "value",
			},
			"num": map[string]interface{}{
				"num1": int64(1),
				"num2": int64(1000000),
			},
		},
		"v1beta2": {
			"apiVersion": "stable.example.com/v1beta2",
			"kind":       "MultiVersion",
			"contentv2": map[string]interface{}{
				"key": "value",
			},
			"numv2": map[string]interface{}{
				"num1": int64(1),
				"num2": int64(1000000),
			},
		},
	}
	if !reflect.DeepEqual(expected[v], j) {
		t.Errorf("unexpected %s object: %s", v, cmp.Diff(expected[v], j))
	}
}

func closeOnCall(h http.Handler) (chan struct{}, http.Handler) {
	ch := make(chan struct{})
	once := sync.Once{}
	return ch, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		once.Do(func() {
			close(ch)
		})
		h.ServeHTTP(w, r)
	})
}
