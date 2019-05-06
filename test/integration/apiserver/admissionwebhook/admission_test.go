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

package admissionwebhook

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/admission/v1beta1"
	admissionv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	dynamic "k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/etcd"
)

const (
	testNamespace = "webhook-integration"

	mutation   = "mutation"
	validation = "validation"
)

type testContext struct {
	t *testing.T

	admissionHolder *holder

	client    dynamic.Interface
	clientset clientset.Interface
	verb      string
	gvr       schema.GroupVersionResource
	resource  metav1.APIResource
	resources map[schema.GroupVersionResource]metav1.APIResource
}

type testFunc func(*testContext)

var (
	// defaultResourceFuncs holds the default test functions.
	// may be overridden for specific resources by customTestFuncs.
	defaultResourceFuncs = map[string]testFunc{
		"create":           testResourceCreate,
		"update":           testResourceUpdate,
		"patch":            testResourcePatch,
		"delete":           testResourceDelete,
		"deletecollection": testResourceDeletecollection,
	}

	// defaultSubresourceFuncs holds default subresource test functions.
	// may be overridden for specific resources by customTestFuncs.
	defaultSubresourceFuncs = map[string]testFunc{
		"update": testSubresourceUpdate,
		"patch":  testSubresourcePatch,
	}

	// customTestFuncs holds custom test functions by resource and verb.
	customTestFuncs = map[schema.GroupVersionResource]map[string]testFunc{
		gvr("", "v1", "namespaces"): {"delete": testNamespaceDelete},

		gvr("apps", "v1beta1", "deployments/rollback"):       {"create": testDeploymentRollback},
		gvr("extensions", "v1beta1", "deployments/rollback"): {"create": testDeploymentRollback},

		gvr("", "v1", "pods/attach"):      {"create": testPodConnectSubresource},
		gvr("", "v1", "pods/exec"):        {"create": testPodConnectSubresource},
		gvr("", "v1", "pods/portforward"): {"create": testPodConnectSubresource},

		gvr("", "v1", "bindings"):      {"create": testPodBindingEviction},
		gvr("", "v1", "pods/binding"):  {"create": testPodBindingEviction},
		gvr("", "v1", "pods/eviction"): {"create": testPodBindingEviction},

		gvr("", "v1", "nodes/proxy"):    {"*": testSubresourceProxy},
		gvr("", "v1", "pods/proxy"):     {"*": testSubresourceProxy},
		gvr("", "v1", "services/proxy"): {"*": testSubresourceProxy},
	}

	// admissionExemptResources lists objects which are exempt from admission validation/mutation,
	// only resources exempted from admission processing by API server should be listed here.
	admissionExemptResources = map[schema.GroupVersionResource]bool{
		gvr("admissionregistration.k8s.io", "v1beta1", "mutatingwebhookconfigurations"):   true,
		gvr("admissionregistration.k8s.io", "v1beta1", "validatingwebhookconfigurations"): true,
	}

	parentResources = map[schema.GroupVersionResource]schema.GroupVersionResource{
		gvr("extensions", "v1beta1", "replicationcontrollers/scale"): gvr("", "v1", "replicationcontrollers"),
	}

	// stubDataOverrides holds either non persistent resources' definitions or resources where default stub needs to be overridden.
	stubDataOverrides = map[schema.GroupVersionResource]string{
		// Non persistent Reviews resource
		gvr("authentication.k8s.io", "v1", "tokenreviews"):                  `{"metadata": {"name": "tokenreview"}, "spec": {"token": "token", "audience": ["audience1","audience2"]}}`,
		gvr("authentication.k8s.io", "v1beta1", "tokenreviews"):             `{"metadata": {"name": "tokenreview"}, "spec": {"token": "token", "audience": ["audience1","audience2"]}}`,
		gvr("authorization.k8s.io", "v1", "localsubjectaccessreviews"):      `{"metadata": {"name": "", "namespace":"` + testNamespace + `"}, "spec": {"uid": "token", "user": "user1","groups": ["group1","group2"],"resourceAttributes": {"name":"name1","namespace":"` + testNamespace + `"}}}`,
		gvr("authorization.k8s.io", "v1", "subjectaccessreviews"):           `{"metadata": {"name": "", "namespace":""}, "spec": {"user":"user1","resourceAttributes": {"name":"name1", "namespace":"` + testNamespace + `"}}}`,
		gvr("authorization.k8s.io", "v1", "selfsubjectaccessreviews"):       `{"metadata": {"name": "", "namespace":""}, "spec": {"resourceAttributes": {"name":"name1", "namespace":""}}}`,
		gvr("authorization.k8s.io", "v1", "selfsubjectrulesreviews"):        `{"metadata": {"name": "", "namespace":"` + testNamespace + `"}, "spec": {"namespace":"` + testNamespace + `"}}`,
		gvr("authorization.k8s.io", "v1beta1", "localsubjectaccessreviews"): `{"metadata": {"name": "", "namespace":"` + testNamespace + `"}, "spec": {"uid": "token", "user": "user1","groups": ["group1","group2"],"resourceAttributes": {"name":"name1","namespace":"` + testNamespace + `"}}}`,
		gvr("authorization.k8s.io", "v1beta1", "subjectaccessreviews"):      `{"metadata": {"name": "", "namespace":""}, "spec": {"user":"user1","resourceAttributes": {"name":"name1", "namespace":"` + testNamespace + `"}}}`,
		gvr("authorization.k8s.io", "v1beta1", "selfsubjectaccessreviews"):  `{"metadata": {"name": "", "namespace":""}, "spec": {"resourceAttributes": {"name":"name1", "namespace":""}}}`,
		gvr("authorization.k8s.io", "v1beta1", "selfsubjectrulesreviews"):   `{"metadata": {"name": "", "namespace":"` + testNamespace + `"}, "spec": {"namespace":"` + testNamespace + `"}}`,

		// Other Non persistent resources
	}
)

type holder struct {
	lock sync.RWMutex

	t *testing.T

	recordGVR       metav1.GroupVersionResource
	recordOperation v1beta1.Operation
	recordNamespace string
	recordName      string

	expectGVK       schema.GroupVersionKind
	expectObject    bool
	expectOldObject bool

	recorded map[string]*v1beta1.AdmissionRequest
}

func (h *holder) reset(t *testing.T) {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.t = t
	h.recordGVR = metav1.GroupVersionResource{}
	h.expectGVK = schema.GroupVersionKind{}
	h.recordOperation = ""
	h.recordName = ""
	h.recordNamespace = ""
	h.expectObject = false
	h.expectOldObject = false
	h.recorded = map[string]*v1beta1.AdmissionRequest{
		mutation:   nil,
		validation: nil,
	}
}
func (h *holder) expect(gvr schema.GroupVersionResource, gvk schema.GroupVersionKind, operation v1beta1.Operation, name, namespace string, object, oldObject bool) {
	// Special-case namespaces, since the object name shows up in request attributes for update/delete requests
	if len(namespace) == 0 && gvk.Group == "" && gvk.Version == "v1" && gvk.Kind == "Namespace" && operation != v1beta1.Create {
		namespace = name
	}

	h.lock.Lock()
	defer h.lock.Unlock()
	h.recordGVR = metav1.GroupVersionResource{Group: gvr.Group, Version: gvr.Version, Resource: gvr.Resource}
	h.expectGVK = gvk
	h.recordOperation = operation
	h.recordName = name
	h.recordNamespace = namespace
	h.expectObject = object
	h.expectOldObject = oldObject
	h.recorded = map[string]*v1beta1.AdmissionRequest{
		mutation:   nil,
		validation: nil,
	}
}
func (h *holder) record(phase string, request *v1beta1.AdmissionRequest) {
	h.lock.Lock()
	defer h.lock.Unlock()

	// this is useful to turn on if items aren't getting recorded and you need to figure out why
	debug := false
	if debug {
		h.t.Logf("%s %#v %v", request.Operation, request.Resource, request.SubResource)
	}

	resource := request.Resource
	if len(request.SubResource) > 0 {
		resource.Resource += "/" + request.SubResource
	}
	if resource != h.recordGVR {
		if debug {
			h.t.Log(resource, "!=", h.recordGVR)
		}
		return
	}

	if request.Operation != h.recordOperation {
		if debug {
			h.t.Log(request.Operation, "!=", h.recordOperation)
		}
		return
	}
	if request.Namespace != h.recordNamespace {
		if debug {
			h.t.Log(request.Namespace, "!=", h.recordNamespace)
		}
		return
	}

	name := request.Name
	if name == "" && request.Object.Object != nil {
		name = request.Object.Object.(*unstructured.Unstructured).GetName()
	}
	if name != h.recordName {
		if debug {
			h.t.Log(name, "!=", h.recordName)
		}
		return
	}

	h.recorded[phase] = request
}

func (h *holder) verify(t *testing.T) {
	h.lock.Lock()
	defer h.lock.Unlock()

	if err := h.verifyRequest(h.recorded[mutation]); err != nil {
		t.Errorf("mutation error: %v", err)
	}
	if err := h.verifyRequest(h.recorded[validation]); err != nil {
		t.Errorf("validation error: %v", err)
	}
}

func (h *holder) verifyRequest(request *v1beta1.AdmissionRequest) error {
	// Check if current resource should be exempted from Admission processing
	if admissionExemptResources[gvr(h.recordGVR.Group, h.recordGVR.Version, h.recordGVR.Resource)] {
		if request == nil {
			return nil
		}
		return fmt.Errorf("admission webhook was called, but not supposed to")
	}

	if request == nil {
		return fmt.Errorf("no request received")
	}

	if h.expectObject {
		if err := h.verifyObject(request.Object.Object); err != nil {
			return fmt.Errorf("object error: %v", err)
		}
	} else if request.Object.Object != nil {
		return fmt.Errorf("unexpected object: %#v", request.Object.Object)
	}

	if h.expectOldObject {
		if err := h.verifyObject(request.OldObject.Object); err != nil {
			return fmt.Errorf("old object error: %v", err)
		}
	} else if request.OldObject.Object != nil {
		return fmt.Errorf("unexpected old object: %#v", request.OldObject.Object)
	}

	return nil
}

func (h *holder) verifyObject(obj runtime.Object) error {
	if obj == nil {
		return fmt.Errorf("no object sent")
	}
	if obj.GetObjectKind().GroupVersionKind() != h.expectGVK {
		return fmt.Errorf("expected %#v, got %#v", h.expectGVK, obj.GetObjectKind().GroupVersionKind())
	}
	return nil
}

// TestWebhookV1beta1 tests communication between API server and webhook process.
func TestWebhookV1beta1(t *testing.T) {
	// holder communicates expectations to webhooks, and results from webhooks
	holder := &holder{t: t}

	// set up webhook server
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	webhookMux := http.NewServeMux()
	webhookMux.Handle("/"+mutation, newWebhookHandler(t, holder, mutation))
	webhookMux.Handle("/"+validation, newWebhookHandler(t, holder, validation))
	webhookServer := httptest.NewUnstartedServer(webhookMux)
	webhookServer.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	webhookServer.StartTLS()
	defer webhookServer.Close()

	// start API server
	master := etcd.StartRealMasterOrDie(t, func(opts *options.ServerRunOptions) {
		// turn off admission plugins that add finalizers
		opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "StorageObjectInUseProtection"}

		// force enable all resources so we can check storage.
		// TODO: drop these once we stop allowing them to be served.
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/deployments"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/daemonsets"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/replicasets"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/podsecuritypolicies"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/networkpolicies"] = "true"
	})
	defer master.Cleanup()

	if _, err := master.Client.CoreV1().Namespaces().Create(&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}); err != nil {
		t.Fatal(err)
	}
	if err := createV1beta1MutationWebhook(master.Client, webhookServer.URL+"/"+mutation); err != nil {
		t.Fatal(err)
	}
	if err := createV1beta1ValidationWebhook(master.Client, webhookServer.URL+"/"+validation); err != nil {
		t.Fatal(err)
	}

	// gather resources to test
	dynamicClient := master.Dynamic
	_, resources, err := master.Client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("Failed to get ServerGroupsAndResources with error: %+v", err)
	}

	gvrsToTest := []schema.GroupVersionResource{}
	resourcesByGVR := map[schema.GroupVersionResource]metav1.APIResource{}

	for _, list := range resources {
		defaultGroupVersion, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			t.Errorf("Failed to get GroupVersion for: %+v", list)
			continue
		}
		for _, resource := range list.APIResources {
			if resource.Group == "" {
				resource.Group = defaultGroupVersion.Group
			}
			if resource.Version == "" {
				resource.Version = defaultGroupVersion.Version
			}
			gvr := defaultGroupVersion.WithResource(resource.Name)
			resourcesByGVR[gvr] = resource
			if shouldTestResource(gvr, resource) {
				gvrsToTest = append(gvrsToTest, gvr)
			}
		}
	}

	sort.SliceStable(gvrsToTest, func(i, j int) bool {
		if gvrsToTest[i].Group < gvrsToTest[j].Group {
			return true
		}
		if gvrsToTest[i].Group > gvrsToTest[j].Group {
			return false
		}
		if gvrsToTest[i].Version < gvrsToTest[j].Version {
			return true
		}
		if gvrsToTest[i].Version > gvrsToTest[j].Version {
			return false
		}
		if gvrsToTest[i].Resource < gvrsToTest[j].Resource {
			return true
		}
		if gvrsToTest[i].Resource > gvrsToTest[j].Resource {
			return false
		}
		return true
	})

	// Test admission on all resources, subresources, and verbs
	for _, gvr := range gvrsToTest {
		resource := resourcesByGVR[gvr]
		t.Run(gvr.Group+"."+gvr.Version+"."+strings.ReplaceAll(resource.Name, "/", "."), func(t *testing.T) {
			for _, verb := range []string{"create", "update", "patch", "connect", "delete", "deletecollection"} {
				if shouldTestResourceVerb(gvr, resource, verb) {
					t.Run(verb, func(t *testing.T) {
						holder.reset(t)
						testFunc := getTestFunc(gvr, verb)
						testFunc(&testContext{
							t:               t,
							admissionHolder: holder,
							client:          dynamicClient,
							clientset:       master.Client,
							verb:            verb,
							gvr:             gvr,
							resource:        resource,
							resources:       resourcesByGVR,
						})
						holder.verify(t)
					})
				}
			}
		})
	}
}

//
// generic resource testing
//

func testResourceCreate(c *testContext) {
	stubObj, err := getStubObj(c.gvr, c.resource)
	if err != nil {
		c.t.Error(err)
		return
	}
	ns := ""
	if c.resource.Namespaced {
		ns = testNamespace
	}
	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Create, stubObj.GetName(), ns, true, false)
	_, err = c.client.Resource(c.gvr).Namespace(ns).Create(stubObj, metav1.CreateOptions{})
	if err != nil {
		c.t.Error(err)
		return
	}
}

func testResourceUpdate(c *testContext) {
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		obj, err := createOrGetResource(c.client, c.gvr, c.resource)
		if err != nil {
			return err
		}
		obj.SetAnnotations(map[string]string{"update": "true"})
		c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Update, obj.GetName(), obj.GetNamespace(), true, true)
		_, err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Update(obj, metav1.UpdateOptions{})
		return err
	}); err != nil {
		c.t.Error(err)
		return
	}
}

func testResourcePatch(c *testContext) {
	obj, err := createOrGetResource(c.client, c.gvr, c.resource)
	if err != nil {
		c.t.Error(err)
		return
	}
	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Update, obj.GetName(), obj.GetNamespace(), true, true)
	_, err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Patch(
		obj.GetName(),
		types.MergePatchType,
		[]byte(`{"metadata":{"annotations":{"patch":"true"}}}`),
		metav1.PatchOptions{})
	if err != nil {
		c.t.Error(err)
		return
	}
}

func testResourceDelete(c *testContext) {
	obj, err := createOrGetResource(c.client, c.gvr, c.resource)
	if err != nil {
		c.t.Error(err)
		return
	}
	background := metav1.DeletePropagationBackground
	zero := int64(0)
	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Delete, obj.GetName(), obj.GetNamespace(), false, false)
	err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Delete(obj.GetName(), &metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background})
	if err != nil {
		c.t.Error(err)
		return
	}

	// wait for the item to be gone
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		obj, err := c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Get(obj.GetName(), metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		if err == nil {
			c.t.Logf("waiting for %#v to be deleted (name: %s, finalizers: %v)...\n", c.gvr, obj.GetName(), obj.GetFinalizers())
			return false, nil
		}
		return false, err
	})
	if err != nil {
		c.t.Error(err)
		return
	}
}

func testResourceDeletecollection(c *testContext) {
	obj, err := createOrGetResource(c.client, c.gvr, c.resource)
	if err != nil {
		c.t.Error(err)
		return
	}
	background := metav1.DeletePropagationBackground
	zero := int64(0)

	// update the object with a label that matches our selector
	_, err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Patch(
		obj.GetName(),
		types.MergePatchType,
		[]byte(`{"metadata":{"labels":{"webhooktest":"true"}}}`),
		metav1.PatchOptions{})
	if err != nil {
		c.t.Error(err)
		return
	}

	// set expectations
	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Delete, "", obj.GetNamespace(), false, false)

	// delete
	err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).DeleteCollection(&metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background}, metav1.ListOptions{LabelSelector: "webhooktest=true"})
	if err != nil {
		c.t.Error(err)
		return
	}

	// wait for the item to be gone
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		obj, err := c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Get(obj.GetName(), metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		if err == nil {
			c.t.Logf("waiting for %#v to be deleted (name: %s, finalizers: %v)...\n", c.gvr, obj.GetName(), obj.GetFinalizers())
			return false, nil
		}
		return false, err
	})
	if err != nil {
		c.t.Error(err)
		return
	}
}

func getParentGVR(gvr schema.GroupVersionResource) schema.GroupVersionResource {
	parentGVR, found := parentResources[gvr]
	// if no special override is found, just drop the subresource
	if !found {
		parentGVR = gvr
		parentGVR.Resource = strings.Split(parentGVR.Resource, "/")[0]
	}
	return parentGVR
}

func testSubresourceUpdate(c *testContext) {
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		parentGVR := getParentGVR(c.gvr)
		parentResource := c.resources[parentGVR]
		obj, err := createOrGetResource(c.client, parentGVR, parentResource)
		if err != nil {
			return err
		}

		// Save the parent object as what we submit
		submitObj := obj

		gvrWithoutSubresources := c.gvr
		gvrWithoutSubresources.Resource = strings.Split(gvrWithoutSubresources.Resource, "/")[0]
		subresources := strings.Split(c.gvr.Resource, "/")[1:]

		// If the subresource supports get, fetch that as the object to submit (namespaces/finalize, */scale, etc)
		if sets.NewString(c.resource.Verbs...).Has("get") {
			submitObj, err = c.client.Resource(gvrWithoutSubresources).Namespace(obj.GetNamespace()).Get(obj.GetName(), metav1.GetOptions{}, subresources...)
			if err != nil {
				return err
			}
		}

		// Modify the object
		submitObj.SetAnnotations(map[string]string{"subresourceupdate": "true"})

		// set expectations
		c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Update, obj.GetName(), obj.GetNamespace(), true, true)

		_, err = c.client.Resource(gvrWithoutSubresources).Namespace(obj.GetNamespace()).Update(
			submitObj,
			metav1.UpdateOptions{},
			subresources...,
		)
		return err
	}); err != nil {
		c.t.Error(err)
	}
}

func testSubresourcePatch(c *testContext) {
	parentGVR := getParentGVR(c.gvr)
	parentResource := c.resources[parentGVR]
	obj, err := createOrGetResource(c.client, parentGVR, parentResource)
	if err != nil {
		c.t.Error(err)
		return
	}

	gvrWithoutSubresources := c.gvr
	gvrWithoutSubresources.Resource = strings.Split(gvrWithoutSubresources.Resource, "/")[0]
	subresources := strings.Split(c.gvr.Resource, "/")[1:]

	// set expectations
	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Update, obj.GetName(), obj.GetNamespace(), true, true)

	_, err = c.client.Resource(gvrWithoutSubresources).Namespace(obj.GetNamespace()).Patch(
		obj.GetName(),
		types.MergePatchType,
		[]byte(`{"metadata":{"annotations":{"subresourcepatch":"true"}}}`),
		metav1.PatchOptions{},
		subresources...,
	)
	if err != nil {
		c.t.Error(err)
		return
	}
}

func unimplemented(c *testContext) {
	c.t.Errorf("Test function for %+v has not been implemented...", c.gvr)
}

//
// custom methods
//

// testNamespaceDelete verifies namespace-specific delete behavior:
// - ensures admission is called on first delete (which only sets deletionTimestamp and terminating state)
// - removes finalizer from namespace
// - ensures admission is called on final delete once finalizers are removed
func testNamespaceDelete(c *testContext) {
	obj, err := createOrGetResource(c.client, c.gvr, c.resource)
	if err != nil {
		c.t.Error(err)
		return
	}
	background := metav1.DeletePropagationBackground
	zero := int64(0)

	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Delete, obj.GetName(), obj.GetNamespace(), false, false)
	err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Delete(obj.GetName(), &metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background})
	if err != nil {
		c.t.Error(err)
		return
	}
	c.admissionHolder.verify(c.t)

	// do the finalization so the namespace can be deleted
	obj, err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Get(obj.GetName(), metav1.GetOptions{})
	if err != nil {
		c.t.Error(err)
		return
	}
	err = unstructured.SetNestedField(obj.Object, nil, "spec", "finalizers")
	if err != nil {
		c.t.Error(err)
		return
	}
	_, err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Update(obj, metav1.UpdateOptions{}, "finalize")
	if err != nil {
		c.t.Error(err)
		return
	}

	// then run the final delete and make sure admission is called again
	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Delete, obj.GetName(), obj.GetNamespace(), false, false)
	err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Delete(obj.GetName(), &metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background})
	if err != nil {
		c.t.Error(err)
		return
	}
	c.admissionHolder.verify(c.t)

	// verify namespace is gone
	obj, err = c.client.Resource(c.gvr).Namespace(obj.GetNamespace()).Get(obj.GetName(), metav1.GetOptions{})
	if err == nil || !errors.IsNotFound(err) {
		c.t.Errorf("expected namespace to be gone, got %#v, %v", obj, err)
	}
}

// testDeploymentRollback verifies rollback-specific behavior:
// - creates a parent deployment
// - creates a rollback object and posts it
func testDeploymentRollback(c *testContext) {
	deploymentGVR := gvr("apps", "v1", "deployments")
	obj, err := createOrGetResource(c.client, deploymentGVR, c.resources[deploymentGVR])
	if err != nil {
		c.t.Error(err)
		return
	}

	gvrWithoutSubresources := c.gvr
	gvrWithoutSubresources.Resource = strings.Split(gvrWithoutSubresources.Resource, "/")[0]
	subresources := strings.Split(c.gvr.Resource, "/")[1:]

	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Create, obj.GetName(), obj.GetNamespace(), true, false)

	var rollbackObj runtime.Object
	switch c.gvr {
	case gvr("apps", "v1beta1", "deployments/rollback"):
		rollbackObj = &appsv1beta1.DeploymentRollback{
			TypeMeta:   metav1.TypeMeta{APIVersion: "apps/v1beta1", Kind: "DeploymentRollback"},
			Name:       obj.GetName(),
			RollbackTo: appsv1beta1.RollbackConfig{Revision: 0},
		}
	case gvr("extensions", "v1beta1", "deployments/rollback"):
		rollbackObj = &extensionsv1beta1.DeploymentRollback{
			TypeMeta:   metav1.TypeMeta{APIVersion: "extensions/v1beta1", Kind: "DeploymentRollback"},
			Name:       obj.GetName(),
			RollbackTo: extensionsv1beta1.RollbackConfig{Revision: 0},
		}
	default:
		c.t.Errorf("unknown rollback resource %#v", c.gvr)
		return
	}

	rollbackUnstructuredBody, err := runtime.DefaultUnstructuredConverter.ToUnstructured(rollbackObj)
	if err != nil {
		c.t.Errorf("ToUnstructured failed: %v", err)
		return
	}
	rollbackUnstructuredObj := &unstructured.Unstructured{Object: rollbackUnstructuredBody}
	rollbackUnstructuredObj.SetName(obj.GetName())

	_, err = c.client.Resource(gvrWithoutSubresources).Namespace(obj.GetNamespace()).Create(rollbackUnstructuredObj, metav1.CreateOptions{}, subresources...)
	if err != nil {
		c.t.Error(err)
		return
	}
}

// testPodConnectSubresource verifies connect subresources
func testPodConnectSubresource(c *testContext) {
	podGVR := gvr("", "v1", "pods")
	pod, err := createOrGetResource(c.client, podGVR, c.resources[podGVR])
	if err != nil {
		c.t.Error(err)
		return
	}

	// check all upgradeable verbs
	for _, httpMethod := range []string{"GET", "POST"} {
		c.t.Logf("verifying %v", httpMethod)

		c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Connect, pod.GetName(), pod.GetNamespace(), true, false)
		var err error
		switch c.gvr {
		case gvr("", "v1", "pods/exec"):
			err = c.clientset.CoreV1().RESTClient().Verb(httpMethod).Namespace(pod.GetNamespace()).Resource("pods").Name(pod.GetName()).SubResource("exec").Do().Error()
		case gvr("", "v1", "pods/attach"):
			err = c.clientset.CoreV1().RESTClient().Verb(httpMethod).Namespace(pod.GetNamespace()).Resource("pods").Name(pod.GetName()).SubResource("attach").Do().Error()
		case gvr("", "v1", "pods/portforward"):
			err = c.clientset.CoreV1().RESTClient().Verb(httpMethod).Namespace(pod.GetNamespace()).Resource("pods").Name(pod.GetName()).SubResource("portforward").Do().Error()
		default:
			c.t.Errorf("unknown subresource %#v", c.gvr)
			return
		}

		if err != nil {
			c.t.Logf("debug: result of subresource connect: %v", err)
		}
		c.admissionHolder.verify(c.t)

	}
}

// testPodBindingEviction verifies pod binding and eviction admission
func testPodBindingEviction(c *testContext) {
	podGVR := gvr("", "v1", "pods")
	pod, err := createOrGetResource(c.client, podGVR, c.resources[podGVR])
	if err != nil {
		c.t.Error(err)
		return
	}

	background := metav1.DeletePropagationBackground
	zero := int64(0)
	forceDelete := &metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background}
	defer func() {
		err := c.clientset.CoreV1().Pods(pod.GetNamespace()).Delete(pod.GetName(), forceDelete)
		if err != nil && !errors.IsNotFound(err) {
			c.t.Error(err)
			return
		}
	}()

	c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Create, pod.GetName(), pod.GetNamespace(), true, false)

	switch c.gvr {
	case gvr("", "v1", "bindings"):
		err = c.clientset.CoreV1().RESTClient().Post().Namespace(pod.GetNamespace()).Resource("bindings").Body(&corev1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: pod.GetName()},
			Target:     corev1.ObjectReference{Name: "foo", Kind: "Node", APIVersion: "v1"},
		}).Do().Error()

	case gvr("", "v1", "pods/binding"):
		err = c.clientset.CoreV1().RESTClient().Post().Namespace(pod.GetNamespace()).Resource("pods").Name(pod.GetName()).SubResource("binding").Body(&corev1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: pod.GetName()},
			Target:     corev1.ObjectReference{Name: "foo", Kind: "Node", APIVersion: "v1"},
		}).Do().Error()

	case gvr("", "v1", "pods/eviction"):
		err = c.clientset.CoreV1().RESTClient().Post().Namespace(pod.GetNamespace()).Resource("pods").Name(pod.GetName()).SubResource("eviction").Body(&policyv1beta1.Eviction{
			ObjectMeta:    metav1.ObjectMeta{Name: pod.GetName()},
			DeleteOptions: forceDelete,
		}).Do().Error()

	default:
		c.t.Errorf("unhandled resource %#v", c.gvr)
		return
	}

	if err != nil {
		c.t.Error(err)
		return
	}
}

// testSubresourceProxy verifies proxy subresources
func testSubresourceProxy(c *testContext) {
	parentGVR := getParentGVR(c.gvr)
	parentResource := c.resources[parentGVR]
	obj, err := createOrGetResource(c.client, parentGVR, parentResource)
	if err != nil {
		c.t.Error(err)
		return
	}

	gvrWithoutSubresources := c.gvr
	gvrWithoutSubresources.Resource = strings.Split(gvrWithoutSubresources.Resource, "/")[0]
	subresources := strings.Split(c.gvr.Resource, "/")[1:]

	verbToHTTPMethods := map[string][]string{
		"create": {"POST", "GET", "HEAD", "OPTIONS"}, // also test read-only verbs map to Connect admission
		"update": {"PUT"},
		"patch":  {"PATCH"},
		"delete": {"DELETE"},
	}
	httpMethodsToTest, ok := verbToHTTPMethods[c.verb]
	if !ok {
		c.t.Errorf("unknown verb %v", c.verb)
		return
	}

	for _, httpMethod := range httpMethodsToTest {
		c.t.Logf("testing %v", httpMethod)
		request := c.clientset.CoreV1().RESTClient().Verb(httpMethod)

		// add the namespace if required
		if len(obj.GetNamespace()) > 0 {
			request = request.Namespace(obj.GetNamespace())
		}

		// set expectations
		c.admissionHolder.expect(c.gvr, gvk(c.resource.Group, c.resource.Version, c.resource.Kind), v1beta1.Connect, obj.GetName(), obj.GetNamespace(), true, false)
		// run the request. we don't actually care if the request is successful, just that admission gets called as expected
		err = request.Resource(gvrWithoutSubresources.Resource).Name(obj.GetName()).SubResource(subresources...).Do().Error()
		if err != nil {
			c.t.Logf("debug: result of subresource proxy (error expected): %v", err)
		}
		// verify the result
		c.admissionHolder.verify(c.t)
	}
}

//
// utility methods
//

func newWebhookHandler(t *testing.T, holder *holder, phase string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		if contentType := r.Header.Get("Content-Type"); contentType != "application/json" {
			t.Errorf("contentType=%s, expect application/json", contentType)
			return
		}

		review := v1beta1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			t.Errorf("Fail to deserialize object: %s with error: %v", string(data), err)
			http.Error(w, err.Error(), 400)
			return
		}

		if review.GetObjectKind().GroupVersionKind() != gvk("admission.k8s.io", "v1beta1", "AdmissionReview") {
			t.Errorf("Invalid admission review kind: %#v", review.GetObjectKind().GroupVersionKind())
			http.Error(w, err.Error(), 400)
			return
		}

		if len(review.Request.Object.Raw) > 0 {
			u := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := json.Unmarshal(review.Request.Object.Raw, u); err != nil {
				t.Errorf("Fail to deserialize object: %s with error: %v", string(review.Request.Object.Raw), err)
				http.Error(w, err.Error(), 400)
				return
			}
			review.Request.Object.Object = u
		}
		if len(review.Request.OldObject.Raw) > 0 {
			u := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := json.Unmarshal(review.Request.OldObject.Raw, u); err != nil {
				t.Errorf("Fail to deserialize object: %s with error: %v", string(review.Request.OldObject.Raw), err)
				http.Error(w, err.Error(), 400)
				return
			}
			review.Request.OldObject.Object = u
		}
		holder.record(phase, review.Request)

		review.Response = &v1beta1.AdmissionResponse{
			Allowed: true,
			UID:     review.Request.UID,
			Result:  &metav1.Status{Message: "admitted"},
		}
		// If we're mutating, and have an object, return a patch to exercise conversion
		if phase == mutation && len(review.Request.Object.Raw) > 0 {
			review.Response.Patch = []byte(`[{"op":"add","path":"/foo","value":"test"}]`)
			jsonPatch := v1beta1.PatchTypeJSONPatch
			review.Response.PatchType = &jsonPatch
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(review); err != nil {
			t.Errorf("Marshal of response failed with error: %v", err)
		}
	})
}

func getTestFunc(gvr schema.GroupVersionResource, verb string) testFunc {
	if f, found := customTestFuncs[gvr][verb]; found {
		return f
	}
	if f, found := customTestFuncs[gvr]["*"]; found {
		return f
	}
	if strings.Contains(gvr.Resource, "/") {
		if f, found := defaultSubresourceFuncs[verb]; found {
			return f
		}
		return unimplemented
	}
	if f, found := defaultResourceFuncs[verb]; found {
		return f
	}
	return unimplemented
}

func getStubObj(gvr schema.GroupVersionResource, resource metav1.APIResource) (*unstructured.Unstructured, error) {
	stub := ""
	if data, ok := etcd.GetEtcdStorageDataForNamespace(testNamespace)[gvr]; ok {
		stub = data.Stub
	}
	if data, ok := stubDataOverrides[gvr]; ok {
		stub = data
	}
	if len(stub) == 0 {
		return nil, fmt.Errorf("no stub data for %#v", gvr)
	}

	stubObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := json.Unmarshal([]byte(stub), &stubObj.Object); err != nil {
		return nil, fmt.Errorf("error unmarshaling stub for %#v: %v", gvr, err)
	}
	return stubObj, nil
}

func createOrGetResource(client dynamic.Interface, gvr schema.GroupVersionResource, resource metav1.APIResource) (*unstructured.Unstructured, error) {
	stubObj, err := getStubObj(gvr, resource)
	if err != nil {
		return nil, err
	}
	ns := ""
	if resource.Namespaced {
		ns = testNamespace
	}
	obj, err := client.Resource(gvr).Namespace(ns).Get(stubObj.GetName(), metav1.GetOptions{})
	if err == nil {
		return obj, nil
	}
	if !errors.IsNotFound(err) {
		return nil, err
	}
	return client.Resource(gvr).Namespace(ns).Create(stubObj, metav1.CreateOptions{})
}

func gvr(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
}
func gvk(group, version, kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: group, Version: version, Kind: kind}
}

func shouldTestResource(gvr schema.GroupVersionResource, resource metav1.APIResource) bool {
	if !sets.NewString(resource.Verbs...).HasAny("create", "update", "patch", "connect", "delete", "deletecollection") {
		return false
	}
	return true
}

func shouldTestResourceVerb(gvr schema.GroupVersionResource, resource metav1.APIResource, verb string) bool {
	if !sets.NewString(resource.Verbs...).Has(verb) {
		return false
	}
	return true
}

//
// webhook registration helpers
//

func createV1beta1ValidationWebhook(client clientset.Interface, endpoint string) error {
	fail := admissionv1beta1.Fail
	// Attaching Admission webhook to API server
	_, err := client.AdmissionregistrationV1beta1().ValidatingWebhookConfigurations().Create(&admissionv1beta1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "admission.integration.test"},
		Webhooks: []admissionv1beta1.Webhook{{
			Name: "admission.integration.test",
			ClientConfig: admissionv1beta1.WebhookClientConfig{
				URL:      &endpoint,
				CABundle: localhostCert,
			},
			Rules: []admissionv1beta1.RuleWithOperations{{
				Operations: []admissionv1beta1.OperationType{admissionv1beta1.OperationAll},
				Rule:       admissionv1beta1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*/*"}},
			}},
			FailurePolicy:           &fail,
			AdmissionReviewVersions: []string{"v1beta1"},
		}},
	})
	return err
}

func createV1beta1MutationWebhook(client clientset.Interface, endpoint string) error {
	fail := admissionv1beta1.Fail
	// Attaching Mutation webhook to API server
	_, err := client.AdmissionregistrationV1beta1().MutatingWebhookConfigurations().Create(&admissionv1beta1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "mutation.integration.test"},
		Webhooks: []admissionv1beta1.Webhook{{
			Name: "mutation.integration.test",
			ClientConfig: admissionv1beta1.WebhookClientConfig{
				URL:      &endpoint,
				CABundle: localhostCert,
			},
			Rules: []admissionv1beta1.RuleWithOperations{{
				Operations: []admissionv1beta1.OperationType{admissionv1beta1.OperationAll},
				Rule:       admissionv1beta1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*/*"}},
			}},
			FailurePolicy:           &fail,
			AdmissionReviewVersions: []string{"v1beta1"},
		}},
	})
	return err
}

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBjzCCATmgAwIBAgIRAKpi2WmTcFrVjxrl5n5YDUEwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgC
QQC9fEbRszP3t14Gr4oahV7zFObBI4TfA5i7YnlMXeLinb7MnvT4bkfOJzE6zktn
59zP7UiHs3l4YOuqrjiwM413AgMBAAGjaDBmMA4GA1UdDwEB/wQEAwICpDATBgNV
HSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MC4GA1UdEQQnMCWCC2V4
YW1wbGUuY29thwR/AAABhxAAAAAAAAAAAAAAAAAAAAABMA0GCSqGSIb3DQEBCwUA
A0EAUsVE6KMnza/ZbodLlyeMzdo7EM/5nb5ywyOxgIOCf0OOLHsPS9ueGLQX9HEG
//yjTXuhNcUugExIjM/AIwAZPQ==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBAL18RtGzM/e3XgavihqFXvMU5sEjhN8DmLtieUxd4uKdvsye9Phu
R84nMTrOS2fn3M/tSIezeXhg66quOLAzjXcCAwEAAQJBAKcRxH9wuglYLBdI/0OT
BLzfWPZCEw1vZmMR2FF1Fm8nkNOVDPleeVGTWoOEcYYlQbpTmkGSxJ6ya+hqRi6x
goECIQDx3+X49fwpL6B5qpJIJMyZBSCuMhH4B7JevhGGFENi3wIhAMiNJN5Q3UkL
IuSvv03kaPR5XVQ99/UeEetUgGvBcABpAiBJSBzVITIVCGkGc7d+RCf49KTCIklv
bGWObufAR8Ni4QIgWpILjW8dkGg8GOUZ0zaNA6Nvt6TIv2UWGJ4v5PoV98kCIQDx
rIiZs5QbKdycsv9gQJzwQAogC8o04X3Zz3dsoX+h4A==
-----END RSA PRIVATE KEY-----`)
