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

package integration

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

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
	"k8s.io/apimachinery/pkg/util/uuid"
	etcd3watcher "k8s.io/apiserver/pkg/storage/etcd3"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	featuregatetesting "k8s.io/component-base/featuregate/testing"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/convert"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apiextensions-apiserver/test/integration/storage"
)

type Checker func(t *testing.T, ctc *conversionTestContext)

func checks(checkers ...Checker) []Checker {
	return checkers
}

func TestWebhookConverter(t *testing.T) {
	tests := []struct {
		group   string
		handler http.Handler
		checks  []Checker
	}{
		{
			group:   "noop-converter",
			handler: convert.NewObjectConverterWebhookHandler(t, noopConverter),
			checks:  checks(validateStorageVersion, validateServed, validateMixedStorageVersions),
		},
		{
			group:   "nontrivial-converter",
			handler: convert.NewObjectConverterWebhookHandler(t, nontrivialConverter),
			checks:  checks(validateStorageVersion, validateServed, validateMixedStorageVersions),
		},
		{
			group:   "empty-response",
			handler: convert.NewReviewWebhookHandler(t, emptyResponseConverter),
			checks:  checks(expectConversionFailureMessage("empty-response", "expected 1 converted objects")),
		},
		{
			group:   "failure-message",
			handler: convert.NewReviewWebhookHandler(t, failureResponseConverter("custom webhook conversion error")),
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
	marker, err := ctc.versionedClient("marker", "v1beta2").Create(newConversionMultiVersionFixture("marker", "marker", "v1beta2"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range tests {
		t.Run(test.group, func(t *testing.T) {
			tearDown, webhookClientConfig, webhookWaitReady, err := convert.StartConversionWebhookServerWithWaitReady(test.handler)
			if err != nil {
				t.Fatal(err)
			}
			defer tearDown()

			ctc.setConversionWebhook(t, webhookClientConfig)
			defer ctc.removeConversionWebhook(t)

			err = webhookWaitReady(30*time.Second, func() error {
				// the marker's storage version is v1beta2, so a v1beta1 read always triggers conversion
				_, err := ctc.versionedClient(marker.GetNamespace(), "v1beta1").Get(marker.GetName(), metav1.GetOptions{})
				return err
			})
			if err != nil {
				t.Fatal(err)
			}

			for i, checkFn := range test.checks {
				name := fmt.Sprintf("check-%d", i)
				t.Run(name, func(t *testing.T) {
					ctc.setAndWaitStorageVersion(t, "v1beta2")
					ctc.namespace = fmt.Sprintf("webhook-conversion-%s-%s", test.group, name)
					checkFn(t, ctc)
				})
			}
		})
	}
}

func validateStorageVersion(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace

	for _, version := range []string{"v1beta1", "v1beta2"} {
		t.Run(version, func(t *testing.T) {
			name := "storageversion-" + version
			client := ctc.versionedClient(ns, version)
			obj, err := client.Create(newConversionMultiVersionFixture(ns, name, version), metav1.CreateOptions{})
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
func validateMixedStorageVersions(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace

	v1client := ctc.versionedClient(ns, "v1beta1")
	v2client := ctc.versionedClient(ns, "v1beta2")
	clients := map[string]dynamic.ResourceInterface{"v1beta1": v1client, "v1beta2": v2client}
	versions := []string{"v1beta1", "v1beta2"}

	// Create CRs at all storage versions
	objNames := []string{}
	for _, version := range versions {
		ctc.setAndWaitStorageVersion(t, version)

		name := "stored-at-" + version
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
					t.Errorf("Expected custom resource to be same regardless of which storage version is used but got %+v != %+v", o1, o2)
				}
			}
		})
	}
}

func validateServed(t *testing.T, ctc *conversionTestContext) {
	ns := ctc.namespace

	for _, version := range []string{"v1beta1", "v1beta2"} {
		t.Run(version, func(t *testing.T) {
			name := "served-" + version
			client := ctc.versionedClient(ns, version)
			obj, err := client.Create(newConversionMultiVersionFixture(ns, name, version), metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			ctc.setServed(t, version, false)
			ctc.waitForServed(t, version, false, client, obj)
			ctc.setServed(t, version, true)
			ctc.waitForServed(t, version, true, client, obj)
		})
	}
}

func expectConversionFailureMessage(id, message string) func(t *testing.T, ctc *conversionTestContext) {
	return func(t *testing.T, ctc *conversionTestContext) {
		ns := ctc.namespace
		v1client := ctc.versionedClient(ns, "v1beta1")
		v2client := ctc.versionedClient(ns, "v1beta2")
		var err error
		// storage version is v1beta2, so this skips conversion
		obj, err := v2client.Create(newConversionMultiVersionFixture(ns, id, "v1beta2"), metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		for _, verb := range []string{"get", "list", "create", "udpate", "patch", "delete", "deletecollection"} {
			t.Run(verb, func(t *testing.T) {
				switch verb {
				case "get":
					_, err = v1client.Get(obj.GetName(), metav1.GetOptions{})
				case "list":
					_, err = v1client.List(metav1.ListOptions{})
				case "create":
					_, err = v1client.Create(newConversionMultiVersionFixture(ns, id, "v1beta1"), metav1.CreateOptions{})
				case "update":
					_, err = v1client.Update(obj, metav1.UpdateOptions{})
				case "patch":
					_, err = v1client.Patch(obj.GetName(), types.MergePatchType, []byte(`{"metadata":{"annotations":{"patch":"true"}}}`), metav1.PatchOptions{})
				case "delete":
					err = v1client.Delete(obj.GetName(), &metav1.DeleteOptions{})
				case "deletecollection":
					err = v1client.DeleteCollection(&metav1.DeleteOptions{}, metav1.ListOptions{})
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
						_, err = v1client.Create(newConversionMultiVersionFixture(ns, id, "v1beta1"), metav1.CreateOptions{}, subresource)
					case "update":
						_, err = v1client.Update(obj, metav1.UpdateOptions{}, subresource)
					case "patch":
						_, err = v1client.Patch(obj.GetName(), types.MergePatchType, []byte(`{"metadata":{"annotations":{"patch":"true"}}}`), metav1.PatchOptions{}, subresource)
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
		return runtime.RawExtension{}, fmt.Errorf("Fail to deserialize object: %s with error: %v", string(obj.Raw), err)
	}
	u.Object["apiVersion"] = desiredAPIVersion
	raw, err := json.Marshal(u)
	if err != nil {
		return runtime.RawExtension{}, fmt.Errorf("Fail to serialize object: %v with error: %v", u, err)
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
		return runtime.RawExtension{}, fmt.Errorf("Fail to deserialize object: %s with error: %v", string(obj.Raw), err)
	}

	currentAPIVersion := u.Object["apiVersion"]
	if currentAPIVersion == "v1beta2" && desiredAPIVersion == "v1beta1" {
		u.Object["num"] = u.Object["numv2"]
		u.Object["content"] = u.Object["contentv2"]
		delete(u.Object, "numv2")
		delete(u.Object, "contentv2")
	}
	if currentAPIVersion == "v1beta1" && desiredAPIVersion == "v1beta2" {
		u.Object["numv2"] = u.Object["num"]
		u.Object["contentv2"] = u.Object["content"]
		delete(u.Object, "num")
		delete(u.Object, "content")
	}
	u.Object["apiVersion"] = desiredAPIVersion
	raw, err := json.Marshal(u)
	if err != nil {
		return runtime.RawExtension{}, fmt.Errorf("Fail to serialize object: %v with error: %v", u, err)
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
	return newNamespacedCustomResourceVersionedClient(ns, c.dynamicClient, c.crd, version)
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
	c.setStorageVersion(t, "v1beta2")

	client := c.versionedClient("probe", "v1beta2")
	name := fmt.Sprintf("probe-%v", uuid.NewUUID())
	storageProbe, err := client.Create(newConversionMultiVersionFixture("probe", name, "v1beta2"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	c.waitForStorageVersion(t, "v1beta2", c.versionedClient(storageProbe.GetNamespace(), "v1beta2"), storageProbe)

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
		crd.Spec.Versions[i].Storage = (v.Name == version)
	}
	crd, err = c.apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd)
	if err != nil {
		t.Fatal(err)
	}
	c.crd = crd
}

func (c *conversionTestContext) waitForStorageVersion(t *testing.T, version string, versionedClient dynamic.ResourceInterface, obj *unstructured.Unstructured) *unstructured.Unstructured {
	c.etcdObjectReader.WaitForStorageVersion(version, obj.GetNamespace(), obj.GetName(), 30*time.Second, func() {
		var err error
		obj, err = versionedClient.Update(obj, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to update object: %v", err)
		}
	})
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
				Name:    "v1beta1",
				Served:  true,
				Storage: false,
			},
			{
				Name:    "v1beta2",
				Served:  true,
				Storage: true,
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
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "stable.example.com/" + version,
			"kind":       "MultiVersion",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"content": map[string]interface{}{
				"key": "value",
			},
			"num": map[string]interface{}{
				"num1": 1,
				"num2": 1000000,
			},
		},
	}
}
