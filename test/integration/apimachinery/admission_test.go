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

package apimachinery

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io/ioutil"
	"net/http"
	// "strings"
	"testing"

	"k8s.io/api/admission/v1beta1"
	admissionv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	// "k8s.io/apimachinery/pkg/types"
	dynamic "k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/etcd"
)

const testNamespace = "webhook-integration"

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

// TestAdmission tests communication between API server and webhook process.
func TestWebhook(t *testing.T) {
	// holder is map of request key by type of Webhook, Admission or Mutation
	holder := make(map[string]*v1beta1.AdmissionRequest)

	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}
	webhook := &http.Server{
		Addr: ":22443",
		TLSConfig: &tls.Config{
			RootCAs:      roots,
			Certificates: []tls.Certificate{cert},
		},
	}
	handler := func(endpoint string) {
		http.HandleFunc(endpoint, func(w http.ResponseWriter, r *http.Request) {
			var body []byte
			if r.Body != nil {
				if data, err := ioutil.ReadAll(r.Body); err == nil {
					body = data
				}
			}
			contentType := r.Header.Get("Content-Type")
			if contentType != "application/json" {
				t.Errorf("contentType=%s, expect application/json", contentType)
			}
			key := r.URL.Path[1:]
			// The AdmissionReview that was sent to the webhook
			requestedAdmissionReview := v1beta1.AdmissionReview{}

			holder[key] = &v1beta1.AdmissionRequest{}

			// The AdmissionReview that will be returned
			responseAdmissionReview := v1beta1.AdmissionReview{}
			deserializer := codecs.UniversalDeserializer()
			if _, _, err := deserializer.Decode(body, nil, &requestedAdmissionReview); err != nil {
				t.Errorf("Fail to deserialize object: %s with error: %+v", string(body), err)
				responseAdmissionReview.Response = response(false, err.Error())
			} else {
				responseAdmissionReview.Response = response(true, "Admitted by integration testing.")
			}
			holder[key] = requestedAdmissionReview.Request
			t.Logf("%s webhook operation: %s", key, requestedAdmissionReview.Request.Operation)
			// Return the same UID
			responseAdmissionReview.Response.UID = requestedAdmissionReview.Request.UID
			respBytes, err := json.Marshal(responseAdmissionReview)
			if err != nil {
				t.Errorf("Marshal of response failed with error: %+v", err)

			}
			if _, err := w.Write(respBytes); err != nil {
				t.Errorf("Writing response failed with error: %+v", err)
			}
		})
	}
	handler("/admission")
	handler("/mutation")
	// Starting Webhook server
	go webhook.ListenAndServeTLS("", "")

	master := etcd.StartRealMasterOrDie(t, func(opts *options.ServerRunOptions) {
		// force enable all resources so we can check storage.
		// TODO: drop these once we stop allowing them to be served.
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/deployments"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/daemonsets"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/replicasets"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/podsecuritypolicies"] = "true"
		opts.APIEnablement.RuntimeConfig["extensions/v1beta1/networkpolicies"] = "true"
	})
	defer master.Cleanup()

	if _, err := master.Client.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}); err != nil {
		t.Fatal(err)
	}

	if err := createAdmissionWebhook(master.Client, "https://127.0.0.1:22443/admission"); err != nil {
		t.Fatal(err)
	}
	if err := createMutationWebhook(master.Client, "https://127.0.0.1:22443/mutation"); err != nil {
		t.Fatal(err)
	}

	dynamicClient := master.Dynamic
	// testData := getAdmisssionTestData()

	_, resources, err := master.Client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("Failed to get ServerGroupsAndResources with error: %+v", err)
	}
	for _, list := range resources {
		defaultGroupVersion, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			t.Errorf("Failed to get GroupVersion for: %+v", list)
			continue
		}
		for _, resource := range list.APIResources {
			gvr := getResourceGVR(defaultGroupVersion, &resource)
			t.Logf("Resource GVR: %+v", gvr)
			if !shouldResourceTest(gvr) {
				t.Logf("Resource %s should not be tested, skipping...", resource.Name)
				continue
			}
			t.Run(gvr.Group+"/"+gvr.Version+resource.Name, func(t *testing.T) {
				if shouldResourceVerbTest(gvr, "create") {
					getTestFunc(gvr, "create")(t, dynamicClient, gvr)
				}
				if shouldResourceVerbTest(gvr, "update") {
					getTestFunc(gvr, "update")(t, dynamicClient, gvr)
				}
				if shouldResourceVerbTest(gvr, "patch") {
					getTestFunc(gvr, "patch")(t, dynamicClient, gvr)
				}
				if shouldResourceVerbTest(gvr, "connect") {
					getTestFunc(gvr, "connect")(t, dynamicClient, gvr)
				}
				if shouldResourceVerbTest(gvr, "delete") {
					getTestFunc(gvr, "delete")(t, dynamicClient, gvr)
				}
				if shouldResourceVerbTest(gvr, "deletecollection") {
					getTestFunc(gvr, "deletecollection")(t, dynamicClient, gvr)
				}
				/*				createObj, ok := testData[ae]
								if !ok {
									t.Logf("Object: %+v has no test data, skipping", ae)
									t.Skip()
								}
								obj, err := createObj()
								if err != nil {
									t.Errorf("Failed to create Unstructured object with error: %+v", err)
								}
								for _, verb := range verbs {
									t.Logf("Testing verb: %s", verb)
									switch verb {
									case "create":
										_, err = client.Resource(schema.GroupVersionResource{Group: ae.gvk.Group, Version: ae.gvk.Version, Resource: ae.resource}).Namespace(testNamespace).Create(obj, metav1.CreateOptions{})
										if err != nil {
											t.Errorf("Failed to create API object: %+v with error: %+v", obj, err)
										}
									case "delete":
										t.Log("Delayed until last")
										continue
									case "update":
										client.Resource(schema.GroupVersionResource{Group: ae.gvk.Group, Version: ae.gvk.Version, Resource: ae.resource}).Namespace(testNamespace).Update(obj, metav1.UpdateOptions{})
									case "patch":
										data, err := obj.MarshalJSON()
										if err != nil {
											t.Errorf("Failer to Marshal unstructured object for Path operation with error: %+v", err)
										}
										client.Resource(schema.GroupVersionResource{Group: ae.gvk.Group, Version: ae.gvk.Version, Resource: ae.resource}).Namespace(testNamespace).Patch(obj.GetName(), types.StrategicMergePatchType, data, metav1.PatchOptions{})
									case "deletecollection":
										t.Log("Not implemented")
										continue
									default:
										continue
									}
									t.Logf("Admission Webhook:")
									t.Logf("Operation: %s", holder["admission"].Operation)
									t.Logf("resource GVK: %+v GVR: %+v", holder["admission"].Kind, holder["admission"].Resource)
									t.Logf("old object: %+v", holder["admission"].OldObject)
									t.Logf("Mutation Webhook:")
									t.Logf("Operation: %s", holder["mutation"].Operation)
									t.Logf("resource GVK: %+v GVR: %+v", holder["mutation"].Kind, holder["mutation"].Resource)
									t.Logf("old object: %+v", holder["mutation"].OldObject)
								}
								// Running last step to delete
								client.Resource(schema.GroupVersionResource{Group: ae.gvk.Group, Version: ae.gvk.Version, Resource: ae.resource}).Namespace(testNamespace).Delete(obj.GetName(), &metav1.DeleteOptions{})
				*/
			})
		}
	}
}

func getResourceGVR(defaultGroupVersion schema.GroupVersion, resource *metav1.APIResource) metav1.GroupVersionResource {
	gvr := metav1.GroupVersionResource{}
	gvr.Resource = resource.Name
	gvk := defaultGroupVersion.WithKind(resource.Kind)
	gvr.Group = gvk.Group
	gvr.Version = gvk.Version
	if len(resource.Version) > 0 {
		gvr.Version = resource.Version
	}
	if len(resource.Group) > 0 {
		gvr.Group = resource.Group
	}
	return gvr
}

// Default functions for each existing verb
func defaultPodCreate(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Default Create for resource: %v", gvr)
	resource := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: testNamespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: "integration-tests:blah",
				},
			},
		},
	}

	data, err := json.Marshal(&resource)
	if err != nil {
		t.Errorf("Error creating object: %+v with error: %+v", gvr, err)
	}
	createObject(data, t, client, gvr, resource.Kind)
}

func createObject(data []byte, t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource, kind string) {
	gvk := schema.GroupVersionKind{
		Group:   gvr.Group,
		Version: gvr.Version,
		Kind:    "Pod",
	}
	obj, err := JSONToUnstructured(data, gvk)
	if err != nil {
		t.Errorf("Error creating object: %+v with error: %+v", gvr, err)
	}
	_, err = client.Resource(schema.GroupVersionResource{Group: gvr.Group, Version: gvr.Version, Resource: gvr.Resource}).Namespace(testNamespace).Create(obj, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create API object: %+v with error: %+v", obj, err)
	}
}

func defaultPodUpdate(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Default Update for resource: %v", gvr)
}

func defaultPodPatch(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Default Patch for resource: %v", gvr)
}

func defaultPodConnect(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Default Connect for resource: %v", gvr)
}

func defaultPodDelete(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Default Delete for resource: %v", gvr)
}

func defaultPodDeletecollection(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Default Deletecollection for resource: %v", gvr)
}

func getTestFunc(gvr metav1.GroupVersionResource, verb string) func(*testing.T, dynamic.Interface, metav1.GroupVersionResource) {
	defaultFuncs := map[metav1.GroupVersionResource]map[string]func(t *testing.T, client dynamic.Interface, resource metav1.GroupVersionResource){
		{
			Group:    "",
			Version:  "v1",
			Resource: "pods",
		}: {
			"create":           defaultPodCreate,
			"update":           defaultPodUpdate,
			"patch":            defaultPodPatch,
			"connect":          defaultPodConnect,
			"delete":           defaultPodDelete,
			"deletecollection": defaultPodDeletecollection,
		},
	}
	if f, found := getResourceCustomTestFunc(gvr, verb); found {
		return f
	}
	if f, found := defaultFuncs[gvr][verb]; found {
		return f
	}
	return defaultNoop
}

func defaultNoop(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Test function for %+v has not been implemented...", gvr)
}

func getResourceCustomTestFunc(gvr metav1.GroupVersionResource, verb string) (func(*testing.T, dynamic.Interface, metav1.GroupVersionResource), bool) {
	// Initialize Custom functions map
	resourceCustomTestFunc := make(map[metav1.GroupVersionResource]map[string]func(*testing.T, dynamic.Interface, metav1.GroupVersionResource))

	// Populating map with custom functions per verb.
	resourceCustomTestFunc = map[metav1.GroupVersionResource]map[string]func(*testing.T, dynamic.Interface, metav1.GroupVersionResource){
		{
			Group:    "",
			Version:  "v1",
			Resource: "pods",
		}: {
			"create": createCustomPod,
		},
	}

	f, ok := resourceCustomTestFunc[gvr][verb]
	return f, ok
}

func createCustomPod(t *testing.T, client dynamic.Interface, gvr metav1.GroupVersionResource) {
	t.Logf("Custom pod create was called")
	resource := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: testNamespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: "integration-tests:blah",
				},
			},
		},
	}

	data, err := json.Marshal(&resource)
	if err != nil {
		t.Errorf("Error creating object: %+v with error: %+v", gvr, err)
	}
	createObject(data, t, client, gvr, resource.Kind)
}

func shouldResourceTest(gvr metav1.GroupVersionResource) bool {
	excludeResource := getExcludedResources()
	if _, ok := excludeResource[gvr]; ok {
		return false
	}
	return true
}

func shouldResourceVerbTest(gvr metav1.GroupVersionResource, verb string) bool {
	excludeResourceVerb := getExcludedResourceVerbs()
	if _, ok := excludeResourceVerb[gvr][verb]; ok {
		return false
	}
	return true
}

func getExcludedResources() map[metav1.GroupVersionResource]bool {
	excludedResources := make(map[metav1.GroupVersionResource]bool, 0)
	return excludedResources
}

func getExcludedResourceVerbs() map[metav1.GroupVersionResource]map[string]bool {
	exludedResourceVerbs := make(map[metav1.GroupVersionResource]map[string]bool, 0)
	return exludedResourceVerbs
}

func createAdmissionWebhook(client clientset.Interface, endpoint string) error {
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
			FailurePolicy: &fail,
		}},
	})
	return err
}

func createMutationWebhook(client clientset.Interface, endpoint string) error {
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
			FailurePolicy: &fail,
		}},
	})
	return err
}

func JSONToUnstructured(object []byte, gvk schema.GroupVersionKind) (*unstructured.Unstructured, error) {
	typeMetaAdder := map[string]interface{}{}
	if err := json.Unmarshal(object, &typeMetaAdder); err != nil {
		return nil, err
	}

	typeMetaAdder["apiVersion"] = gvk.GroupVersion().String()
	typeMetaAdder["kind"] = gvk.Kind

	return &unstructured.Unstructured{Object: typeMetaAdder}, nil
}

func response(allowed bool, msg string) *v1beta1.AdmissionResponse {
	reviewResponse := v1beta1.AdmissionResponse{}
	reviewResponse.Allowed = allowed
	reviewResponse.Result = &metav1.Status{Message: msg}
	return &reviewResponse
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
