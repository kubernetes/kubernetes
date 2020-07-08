/*
Copyright 2017 The Kubernetes Authors.

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

package master

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/go-openapi/spec"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	// testApiextensionsOverlapProbeString is a probe string which identifies whether
	// a CRD change triggers an OpenAPI spec change
	testApiextensionsOverlapProbeString = "testApiextensionsOverlapProbeField"
)

func TestRun(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// test whether the server is really healthy after /healthz told us so
	t.Logf("Creating Deployment directly after being healthy")
	var replicas int32 = 1
	_, err = client.AppsV1().Deployments("default").Create(&appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "test",
			Labels:    map[string]string{"foo": "bar"},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "foo",
							Image: "foo",
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
}

func endpointReturnsStatusOK(client *kubernetes.Clientset, path string) (bool, error) {
	res := client.CoreV1().RESTClient().Get().RequestURI(path).Do()
	var status int
	res.StatusCode(&status)
	_, err := res.Raw()
	if err != nil {
		return false, err
	}
	return status == http.StatusOK, nil
}

func TestLivezAndReadyz(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--livez-grace-period", "0s"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if statusOK, err := endpointReturnsStatusOK(client, "/livez"); err != nil || !statusOK {
		t.Fatalf("livez should be healthy, got %v and error %v", statusOK, err)
	}
	if statusOK, err := endpointReturnsStatusOK(client, "/readyz"); err != nil || !statusOK {
		t.Fatalf("readyz should be healthy, got %v and error %v", statusOK, err)
	}
}

// TestOpenAPIDelegationChainPlumbing is a smoke test that checks for
// the existence of some representative paths from the
// apiextensions-server and the kube-aggregator server, both part of
// the delegation chain in kube-apiserver.
func TestOpenAPIDelegationChainPlumbing(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := kubeclient.RESTClient().Get().AbsPath("/openapi/v2").Do()
	status := 0
	result.StatusCode(&status)
	if status != http.StatusOK {
		t.Fatalf("GET /openapi/v2 failed: expected status=%d, got=%d", http.StatusOK, status)
	}

	raw, err := result.Raw()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	type openAPISchema struct {
		Paths map[string]interface{} `json:"paths"`
	}

	var doc openAPISchema
	err = json.Unmarshal(raw, &doc)
	if err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	matchedExtension := false
	extensionsPrefix := "/apis/" + apiextensions.GroupName

	matchedRegistration := false
	registrationPrefix := "/apis/" + apiregistration.GroupName

	for path := range doc.Paths {
		if strings.HasPrefix(path, extensionsPrefix) {
			matchedExtension = true
		}
		if strings.HasPrefix(path, registrationPrefix) {
			matchedRegistration = true
		}
		if matchedExtension && matchedRegistration {
			return
		}
	}

	if !matchedExtension {
		t.Errorf("missing path: %q", extensionsPrefix)
	}

	if !matchedRegistration {
		t.Errorf("missing path: %q", registrationPrefix)
	}
}

func TestOpenAPIApiextensionsOverlapProtection(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()
	apiextensionsclient, err := apiextensionsclientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	crdPath, exist, err := getOpenAPIPath(apiextensionsclient, `/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions/{name}`)
	if err != nil {
		t.Fatalf("unexpected error getting CRD OpenAPI path: %v", err)
	}
	if !exist {
		t.Fatalf("unexpected error: apiextensions OpenAPI path doesn't exist")
	}

	// Create a CRD that overlaps OpenAPI path with the CRD API
	crd := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "customresourcedefinitions.apiextensions.k8s.io",
			Annotations: map[string]string{"api-approved.kubernetes.io": "unapproved, test-only"},
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "apiextensions.k8s.io",
			Version: "v1beta1",
			Scope:   apiextensionsv1beta1.ClusterScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   "customresourcedefinitions",
				Singular: "customresourcedefinition",
				Kind:     "CustomResourceDefinition",
				ListKind: "CustomResourceDefinitionList",
			},
			Validation: &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
						testApiextensionsOverlapProbeString: {Type: "boolean"},
					},
				},
			},
		},
	}
	etcd.CreateTestCRDs(t, apiextensionsclient, false, crd)

	// Create a probe CRD foo that triggers an OpenAPI spec change
	if err := triggerSpecUpdateWithProbeCRD(t, apiextensionsclient, "foo"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Expect the CRD path to not change
	path, _, err := getOpenAPIPath(apiextensionsclient, `/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions/{name}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathBytes, err := json.Marshal(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	crdPathBytes, err := json.Marshal(crdPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(pathBytes, crdPathBytes) {
		t.Fatalf("expected CRD OpenAPI path to not change, but got different results: want %q, got %q", string(crdPathBytes), string(pathBytes))
	}

	// Expect the orphan definition to be pruned from the spec
	exist, err = specHasProbe(apiextensionsclient, testApiextensionsOverlapProbeString)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exist {
		t.Fatalf("unexpected error: orphan definition isn't pruned")
	}

	// Create a CRD that overlaps OpenAPI definition with the CRD API
	crd = &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "customresourcedefinitions.apiextensions.apis.pkg.apiextensions-apiserver.k8s.io",
			Annotations: map[string]string{"api-approved.kubernetes.io": "unapproved, test-only"},
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "apiextensions.apis.pkg.apiextensions-apiserver.k8s.io",
			Version: "v1beta1",
			Scope:   apiextensionsv1beta1.ClusterScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   "customresourcedefinitions",
				Singular: "customresourcedefinition",
				Kind:     "CustomResourceDefinition",
				ListKind: "CustomResourceDefinitionList",
			},
			Validation: &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
						testApiextensionsOverlapProbeString: {Type: "boolean"},
					},
				},
			},
		},
	}
	etcd.CreateTestCRDs(t, apiextensionsclient, false, crd)

	// Create a probe CRD bar that triggers an OpenAPI spec change
	if err := triggerSpecUpdateWithProbeCRD(t, apiextensionsclient, "bar"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Expect the apiextensions definition to not change, since the overlapping definition will get renamed.
	apiextensionsDefinition, exist, err := getOpenAPIDefinition(apiextensionsclient, `io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1beta1.CustomResourceDefinition`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !exist {
		t.Fatalf("unexpected error: apiextensions definition doesn't exist")
	}
	bytes, err := json.Marshal(apiextensionsDefinition)
	if exist := strings.Contains(string(bytes), testApiextensionsOverlapProbeString); exist {
		t.Fatalf("unexpected error: apiextensions definition gets overlapped")
	}
}

// triggerSpecUpdateWithProbeCRD creates a probe CRD with suffix in name, and waits until
// the path and definition for the probe CRD show up in the OpenAPI spec
func triggerSpecUpdateWithProbeCRD(t *testing.T, apiextensionsclient *apiextensionsclientset.Clientset, suffix string) error {
	// Create a probe CRD that triggers OpenAPI spec change
	name := fmt.Sprintf("integration-test-%s-crd", suffix)
	kind := fmt.Sprintf("Integration-test-%s-crd", suffix)
	group := "probe.test.com"
	crd := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: name + "s." + group},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   group,
			Version: "v1",
			Scope:   apiextensionsv1beta1.ClusterScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   name + "s",
				Singular: name,
				Kind:     kind,
				ListKind: kind + "List",
			},
		},
	}
	etcd.CreateTestCRDs(t, apiextensionsclient, false, crd)

	// Expect the probe CRD path to show up in the OpenAPI spec
	// TODO(roycaihw): expose response header in rest client and utilize etag here
	if err := wait.Poll(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, exist, err := getOpenAPIPath(apiextensionsclient, fmt.Sprintf(`/apis/%s/v1/%ss/{name}`, group, name))
		if err != nil {
			return false, err
		}
		return exist, nil
	}); err != nil {
		return fmt.Errorf("failed to observe probe CRD path in the spec: %v", err)
	}
	return nil
}

func specHasProbe(clientset *apiextensionsclientset.Clientset, probe string) (bool, error) {
	bs, err := clientset.RESTClient().Get().AbsPath("openapi", "v2").DoRaw()
	if err != nil {
		return false, err
	}
	return strings.Contains(string(bs), probe), nil
}

func getOpenAPIPath(clientset *apiextensionsclientset.Clientset, path string) (spec.PathItem, bool, error) {
	bs, err := clientset.RESTClient().Get().AbsPath("openapi", "v2").DoRaw()
	if err != nil {
		return spec.PathItem{}, false, err
	}
	s := spec.Swagger{}
	if err := json.Unmarshal(bs, &s); err != nil {
		return spec.PathItem{}, false, err
	}
	if s.SwaggerProps.Paths == nil {
		return spec.PathItem{}, false, fmt.Errorf("unexpected empty path")
	}
	value, ok := s.SwaggerProps.Paths.Paths[path]
	return value, ok, nil
}

func getOpenAPIDefinition(clientset *apiextensionsclientset.Clientset, definition string) (spec.Schema, bool, error) {
	bs, err := clientset.RESTClient().Get().AbsPath("openapi", "v2").DoRaw()
	if err != nil {
		return spec.Schema{}, false, err
	}
	s := spec.Swagger{}
	if err := json.Unmarshal(bs, &s); err != nil {
		return spec.Schema{}, false, err
	}
	if s.SwaggerProps.Definitions == nil {
		return spec.Schema{}, false, fmt.Errorf("unexpected empty path")
	}
	value, ok := s.SwaggerProps.Definitions[definition]
	return value, ok, nil
}

// return the unique endpoint IPs
func getEndpointIPs(endpoints *corev1.Endpoints) []string {
	endpointMap := make(map[string]bool)
	ips := make([]string, 0)
	for _, subset := range endpoints.Subsets {
		for _, address := range subset.Addresses {
			if _, ok := endpointMap[address.IP]; !ok {
				endpointMap[address.IP] = true
				ips = append(ips, address.IP)
			}
		}
	}
	return ips
}

func verifyEndpointsWithIPs(servers []*kubeapiservertesting.TestServer, ips []string) bool {
	listenAddresses := make([]string, 0)
	for _, server := range servers {
		listenAddresses = append(listenAddresses, server.ServerOpts.GenericServerRunOptions.AdvertiseAddress.String())
	}
	return reflect.DeepEqual(listenAddresses, ips)
}

func testReconcilersMasterLease(t *testing.T, leaseCount int, masterCount int) {
	var leaseServers []*kubeapiservertesting.TestServer
	var masterCountServers []*kubeapiservertesting.TestServer
	etcd := framework.SharedEtcd()

	instanceOptions := &kubeapiservertesting.TestServerInstanceOptions{
		DisableStorageCleanup: true,
	}

	// cleanup the registry storage
	defer registry.CleanupStorage()

	// 1. start masterCount api servers
	for i := 0; i < masterCount; i++ {
		// start master count api server
		server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, []string{
			"--endpoint-reconciler-type", "master-count",
			"--advertise-address", fmt.Sprintf("10.0.1.%v", i+1),
			"--apiserver-count", fmt.Sprintf("%v", masterCount),
		}, etcd)
		masterCountServers = append(masterCountServers, server)
	}

	// 2. verify master count servers have registered
	if err := wait.PollImmediate(3*time.Second, 2*time.Minute, func() (bool, error) {
		client, err := kubernetes.NewForConfig(masterCountServers[0].ClientConfig)
		endpoints, err := client.CoreV1().Endpoints("default").Get("kubernetes", metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return verifyEndpointsWithIPs(masterCountServers, getEndpointIPs(endpoints)), nil
	}); err != nil {
		t.Fatalf("master count endpoints failed to register: %v", err)
	}

	// 3. start lease api servers
	for i := 0; i < leaseCount; i++ {
		options := []string{
			"--endpoint-reconciler-type", "lease",
			"--advertise-address", fmt.Sprintf("10.0.1.%v", i+10),
		}
		server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, options, etcd)
		defer server.TearDownFn()
		leaseServers = append(leaseServers, server)
	}

	time.Sleep(3 * time.Second)

	// 4. Shutdown the masterCount server
	for _, server := range masterCountServers {
		server.TearDownFn()
	}

	// 5. verify only leaseEndpoint servers left
	if err := wait.PollImmediate(3*time.Second, 2*time.Minute, func() (bool, error) {
		client, err := kubernetes.NewForConfig(leaseServers[0].ClientConfig)
		if err != nil {
			t.Logf("create client error: %v", err)
			return false, nil
		}
		endpoints, err := client.CoreV1().Endpoints("default").Get("kubernetes", metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return verifyEndpointsWithIPs(leaseServers, getEndpointIPs(endpoints)), nil
	}); err != nil {
		t.Fatalf("did not find only lease endpoints: %v", err)
	}
}

func TestReconcilerMasterLeaseCombined(t *testing.T) {
	testReconcilersMasterLease(t, 1, 3)
}

func TestReconcilerMasterLeaseMultiMoreMasters(t *testing.T) {
	testReconcilersMasterLease(t, 3, 2)
}

func TestReconcilerMasterLeaseMultiCombined(t *testing.T) {
	testReconcilersMasterLease(t, 3, 3)
}
