/*
Copyright 2015 The Kubernetes Authors.

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
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/group"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest"
	clientsetv1 "k8s.io/client-go/kubernetes"
	clienttypedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	api "k8s.io/kubernetes/pkg/apis/core"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	AliceToken string = "abc123" // username: alice.  Present in token file.
	BobToken   string = "xyz987" // username: bob.  Present in token file.
)

type allowAliceAuthorizer struct{}

func (allowAliceAuthorizer) Authorize(a authorizer.Attributes) (authorizer.Decision, string, error) {
	if a.GetUser() != nil && a.GetUser().GetName() == "alice" {
		return authorizer.DecisionAllow, "", nil
	}
	return authorizer.DecisionNoOpinion, "I can't allow that.  Go ask alice.", nil
}

func testPrefix(t *testing.T, prefix string) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	resp, err := http.Get(s.URL + prefix)
	if err != nil {
		t.Fatalf("unexpected error getting %s prefix: %v", prefix, err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}
}

func TestAutoscalingPrefix(t *testing.T) {
	testPrefix(t, "/apis/autoscaling/")
}

func TestBatchPrefix(t *testing.T) {
	testPrefix(t, "/apis/batch/")
}

func TestAppsPrefix(t *testing.T) {
	testPrefix(t, "/apis/apps/")
}

func TestExtensionsPrefix(t *testing.T) {
	testPrefix(t, "/apis/extensions/")
}

func TestKubernetesService(t *testing.T) {
	config := framework.NewMasterConfig()
	_, _, closeFn := framework.RunAMaster(config)
	defer closeFn()
	coreClient := clientset.NewForConfigOrDie(config.GenericConfig.LoopbackClientConfig)
	if _, err := coreClient.Core().Services(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{}); err != nil {
		t.Fatalf("Expected kubernetes service to exists, got: %v", err)
	}
}

func TestEmptyList(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	u := s.URL + "/api/v1/namespaces/default/pods"
	resp, err := http.Get(u)
	if err != nil {
		t.Fatalf("unexpected error getting %s: %v", u, err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}
	defer resp.Body.Close()
	data, _ := ioutil.ReadAll(resp.Body)
	decodedData := map[string]interface{}{}
	if err := json.Unmarshal(data, &decodedData); err != nil {
		t.Logf("body: %s", string(data))
		t.Fatalf("got error decoding data: %v", err)
	}
	if items, ok := decodedData["items"]; !ok {
		t.Logf("body: %s", string(data))
		t.Fatalf("missing items field in empty list (all lists should return an items field)")
	} else if items == nil {
		t.Logf("body: %s", string(data))
		t.Fatalf("nil items field from empty list (all lists should return non-nil empty items lists)")
	}
}

func initStatusForbiddenMasterCongfig() *master.Config {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authorization.Authorizer = authorizerfactory.NewAlwaysDenyAuthorizer()
	return masterConfig
}

func initUnauthorizedMasterCongfig() *master.Config {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	tokenAuthenticator := tokentest.New()
	tokenAuthenticator.Tokens[AliceToken] = &user.DefaultInfo{Name: "alice", UID: "1"}
	tokenAuthenticator.Tokens[BobToken] = &user.DefaultInfo{Name: "bob", UID: "2"}
	masterConfig.GenericConfig.Authentication.Authenticator = group.NewGroupAdder(bearertoken.New(tokenAuthenticator), []string{user.AllAuthenticated})
	masterConfig.GenericConfig.Authorization.Authorizer = allowAliceAuthorizer{}
	return masterConfig
}

func TestStatus(t *testing.T) {
	testCases := []struct {
		name         string
		masterConfig *master.Config
		statusCode   int
		reqPath      string
		reason       string
		message      string
	}{
		{
			name:         "404",
			masterConfig: nil,
			statusCode:   http.StatusNotFound,
			reqPath:      "/apis/batch/v1/namespaces/default/jobs/foo",
			reason:       "NotFound",
			message:      `jobs.batch "foo" not found`,
		},
		{
			name:         "403",
			masterConfig: initStatusForbiddenMasterCongfig(),
			statusCode:   http.StatusForbidden,
			reqPath:      "/apis",
			reason:       "Forbidden",
			message:      `forbidden: User "" cannot get path "/apis": Everything is forbidden.`,
		},
		{
			name:         "401",
			masterConfig: initUnauthorizedMasterCongfig(),
			statusCode:   http.StatusUnauthorized,
			reqPath:      "/apis",
			reason:       "Unauthorized",
			message:      `Unauthorized`,
		},
	}

	for _, tc := range testCases {
		_, s, closeFn := framework.RunAMaster(tc.masterConfig)
		defer closeFn()

		u := s.URL + tc.reqPath
		resp, err := http.Get(u)
		if err != nil {
			t.Fatalf("unexpected error getting %s: %v", u, err)
		}
		if resp.StatusCode != tc.statusCode {
			t.Fatalf("got status %v instead of %s", resp.StatusCode, tc.name)
		}
		defer resp.Body.Close()
		data, _ := ioutil.ReadAll(resp.Body)
		decodedData := map[string]interface{}{}
		if err := json.Unmarshal(data, &decodedData); err != nil {
			t.Logf("body: %s", string(data))
			t.Fatalf("got error decoding data: %v", err)
		}
		t.Logf("body: %s", string(data))

		if got, expected := decodedData["apiVersion"], "v1"; got != expected {
			t.Errorf("unexpected apiVersion %q, expected %q", got, expected)
		}
		if got, expected := decodedData["kind"], "Status"; got != expected {
			t.Errorf("unexpected kind %q, expected %q", got, expected)
		}
		if got, expected := decodedData["status"], "Failure"; got != expected {
			t.Errorf("unexpected status %q, expected %q", got, expected)
		}
		if got, expected := decodedData["code"], float64(tc.statusCode); got != expected {
			t.Errorf("unexpected code %v, expected %v", got, expected)
		}
		if got, expected := decodedData["reason"], tc.reason; got != expected {
			t.Errorf("unexpected reason %v, expected %v", got, expected)
		}
		if got, expected := decodedData["message"], tc.message; got != expected {
			t.Errorf("unexpected message %v, expected %v", got, expected)
		}
	}
}

func constructBody(val string, size int, field string, t *testing.T) *appsv1.Deployment {
	var replicas int32 = 1
	deploymentObject := &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "test",
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"foo": "bar",
				},
			},
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
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
	}

	switch field {
	case "labels":
		labelsMap := map[string]string{}
		for i := 0; i < size; i++ {
			key := val + strconv.Itoa(i)
			labelsMap[key] = val
		}
		deploymentObject.ObjectMeta.Labels = labelsMap
	case "annotations":
		annotationsMap := map[string]string{}
		for i := 0; i < size; i++ {
			key := val + strconv.Itoa(i)
			annotationsMap[key] = val
		}
		deploymentObject.ObjectMeta.Annotations = annotationsMap
	case "finalizers":
		finalizerString := []string{}
		for i := 0; i < size; i++ {
			finalizerString = append(finalizerString, val)
		}
		deploymentObject.ObjectMeta.Finalizers = finalizerString
	default:
		t.Fatalf("Unexpected field: %s used for making large deployment object value", field)
	}

	return deploymentObject
}

func TestObjectSizeResponses(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	client := clientsetv1.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	const DeploymentMegabyteSize = 100000
	const DeploymentTwoMegabyteSize = 1000000

	expectedMsgFor1MB := `etcdserver: request is too large`
	expectedMsgFor2MB := `rpc error: code = ResourceExhausted desc = trying to send message larger than max`
	expectedMsgForLargeAnnotation := `metadata.annotations: Too long: must have at most 262144 characters`

	deployment1 := constructBody("a", DeploymentMegabyteSize, "labels", t)    // >1 MB file
	deployment2 := constructBody("a", DeploymentTwoMegabyteSize, "labels", t) // >2 MB file

	deployment3 := constructBody("a", DeploymentMegabyteSize, "annotations", t)

	deployment4 := constructBody("sample/sample", DeploymentMegabyteSize, "finalizers", t)    // >1 MB file
	deployment5 := constructBody("sample/sample", DeploymentTwoMegabyteSize, "finalizers", t) // >2 MB file

	requests := []struct {
		size             string
		deploymentObject *appsv1.Deployment
		expectedMessage  string
	}{
		{"1 MB", deployment1, expectedMsgFor1MB},
		{"2 MB", deployment2, expectedMsgFor2MB},
		{"1 MB", deployment3, expectedMsgForLargeAnnotation},
		{"1 MB", deployment4, expectedMsgFor1MB},
		{"2 MB", deployment5, expectedMsgFor2MB},
	}

	for _, r := range requests {
		t.Run(r.size, func(t *testing.T) {
			_, err := client.AppsV1().Deployments(metav1.NamespaceDefault).Create(r.deploymentObject)
			if err != nil {
				if !strings.Contains(err.Error(), r.expectedMessage) {
					t.Errorf("got: %s;want: %s", err.Error(), r.expectedMessage)
				}
			}
		})
	}
}

func TestWatchSucceedsWithoutArgs(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	resp, err := http.Get(s.URL + "/api/v1/namespaces?watch=1")
	if err != nil {
		t.Fatalf("unexpected error getting experimental prefix: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}
	resp.Body.Close()
}

var hpaV1 string = `
{
  "apiVersion": "autoscaling/v1",
  "kind": "HorizontalPodAutoscaler",
  "metadata": {
    "name": "test-hpa",
    "namespace": "default"
  },
  "spec": {
    "scaleTargetRef": {
      "kind": "ReplicationController",
      "name": "test-hpa",
      "namespace": "default"
    },
    "minReplicas": 1,
    "maxReplicas": 10,
    "targetCPUUtilizationPercentage": 50
  }
}
`

var deploymentExtensions string = `
{
  "apiVersion": "extensions/v1beta1",
  "kind": "Deployment",
  "metadata": {
     "name": "test-deployment1",
     "namespace": "default"
  },
  "spec": {
    "replicas": 1,
    "template": {
      "metadata": {
        "labels": {
          "app": "nginx0"
        }
      },
      "spec": {
        "containers": [{
          "name": "nginx",
          "image": "k8s.gcr.io/nginx:1.7.9"
        }]
      }
    }
  }
}
`

var deploymentApps string = `
{
  "apiVersion": "apps/v1beta1",
  "kind": "Deployment",
  "metadata": {
     "name": "test-deployment2",
     "namespace": "default"
  },
  "spec": {
    "replicas": 1,
    "template": {
      "metadata": {
        "labels": {
          "app": "nginx0"
        }
      },
      "spec": {
        "containers": [{
          "name": "nginx",
          "image": "k8s.gcr.io/nginx:1.7.9"
        }]
      }
    }
  }
}
`

func autoscalingPath(resource, namespace, name string) string {
	return testapi.Autoscaling.ResourcePath(resource, namespace, name)
}

func batchPath(resource, namespace, name string) string {
	return testapi.Batch.ResourcePath(resource, namespace, name)
}

func extensionsPath(resource, namespace, name string) string {
	return testapi.Extensions.ResourcePath(resource, namespace, name)
}

func appsPath(resource, namespace, name string) string {
	return testapi.Apps.ResourcePath(resource, namespace, name)
}

func TestAutoscalingGroupBackwardCompatibility(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()
	transport := http.DefaultTransport

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		{"POST", autoscalingPath("horizontalpodautoscalers", metav1.NamespaceDefault, ""), hpaV1, integration.Code201, ""},
		{"GET", autoscalingPath("horizontalpodautoscalers", metav1.NamespaceDefault, ""), "", integration.Code200, testapi.Autoscaling.GroupVersion().String()},
	}

	for _, r := range requests {
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			b, _ := ioutil.ReadAll(resp.Body)
			body := string(b)
			if _, ok := r.expectedStatusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.expectedStatusCodes, resp.StatusCode)
				t.Errorf("Body: %v", body)
			}
			if !strings.Contains(body, "\"apiVersion\":\""+r.expectedVersion) {
				t.Logf("case %v", r)
				t.Errorf("Expected version %v, got body %v", r.expectedVersion, body)
			}
		}()
	}
}

func TestAppsGroupBackwardCompatibility(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()
	transport := http.DefaultTransport

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		// Post to extensions endpoint and get back from both: extensions and apps
		{"POST", extensionsPath("deployments", metav1.NamespaceDefault, ""), deploymentExtensions, integration.Code201, ""},
		{"GET", extensionsPath("deployments", metav1.NamespaceDefault, "test-deployment1"), "", integration.Code200, testapi.Extensions.GroupVersion().String()},
		{"GET", appsPath("deployments", metav1.NamespaceDefault, "test-deployment1"), "", integration.Code200, testapi.Apps.GroupVersion().String()},
		{"DELETE", extensionsPath("deployments", metav1.NamespaceDefault, "test-deployment1"), "", integration.Code200, testapi.Extensions.GroupVersion().String()},
		// Post to apps endpoint and get back from both: apps and extensions
		{"POST", appsPath("deployments", metav1.NamespaceDefault, ""), deploymentApps, integration.Code201, ""},
		{"GET", appsPath("deployments", metav1.NamespaceDefault, "test-deployment2"), "", integration.Code200, testapi.Apps.GroupVersion().String()},
		{"GET", extensionsPath("deployments", metav1.NamespaceDefault, "test-deployment2"), "", integration.Code200, testapi.Extensions.GroupVersion().String()},
		{"DELETE", appsPath("deployments", metav1.NamespaceDefault, "test-deployment2"), "", integration.Code200, testapi.Apps.GroupVersion().String()},
	}

	for _, r := range requests {
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			b, _ := ioutil.ReadAll(resp.Body)
			body := string(b)
			if _, ok := r.expectedStatusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.expectedStatusCodes, resp.StatusCode)
				t.Errorf("Body: %v", body)
			}
			if !strings.Contains(body, "\"apiVersion\":\""+r.expectedVersion) {
				t.Logf("case %v", r)
				t.Errorf("Expected version %v, got body %v", r.expectedVersion, body)
			}
		}()
	}
}

func TestAccept(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	resp, err := http.Get(s.URL + "/api/")
	if err != nil {
		t.Fatalf("unexpected error getting api: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}

	body, _ := ioutil.ReadAll(resp.Body)
	if resp.Header.Get("Content-Type") != "application/json" {
		t.Errorf("unexpected content: %s", body)
	}
	if err := json.Unmarshal(body, &map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("GET", s.URL+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application/yaml")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ = ioutil.ReadAll(resp.Body)
	if resp.Header.Get("Content-Type") != "application/yaml" {
		t.Errorf("unexpected content: %s", body)
	}
	t.Logf("body: %s", body)
	if err := yaml.Unmarshal(body, &map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("GET", s.URL+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application/json, application/yaml")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ = ioutil.ReadAll(resp.Body)
	if resp.Header.Get("Content-Type") != "application/json" {
		t.Errorf("unexpected content: %s", body)
	}
	t.Logf("body: %s", body)
	if err := yaml.Unmarshal(body, &map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("GET", s.URL+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application") // not a valid media type
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNotAcceptable {
		t.Errorf("unexpected error from the server")
	}
}

func countEndpoints(eps *api.Endpoints) int {
	count := 0
	for i := range eps.Subsets {
		count += len(eps.Subsets[i].Addresses) * len(eps.Subsets[i].Ports)
	}
	return count
}

func TestMasterService(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer closeFn()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
		svcList, err := client.Core().Services(metav1.NamespaceDefault).List(metav1.ListOptions{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return false, nil
		}
		found := false
		for i := range svcList.Items {
			if svcList.Items[i].Name == "kubernetes" {
				found = true
				break
			}
		}
		if found {
			ep, err := client.Core().Endpoints(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			if countEndpoints(ep) == 0 {
				return false, fmt.Errorf("no endpoints for kubernetes service: %v", ep)
			}
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestServiceAlloc(t *testing.T) {
	cfg := framework.NewIntegrationTestMasterConfig()
	_, cidr, err := net.ParseCIDR("192.168.0.0/29")
	if err != nil {
		t.Fatalf("bad cidr: %v", err)
	}
	cfg.ExtraConfig.ServiceIPRange = *cidr
	_, s, closeFn := framework.RunAMaster(cfg)
	defer closeFn()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	svc := func(i int) *api.Service {
		return &api.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("svc-%v", i),
			},
			Spec: api.ServiceSpec{
				Type: api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{
					{Port: 80},
				},
			},
		}
	}

	// Wait until the default "kubernetes" service is created.
	if err = wait.Poll(250*time.Millisecond, time.Minute, func() (bool, error) {
		_, err := client.Core().Services(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
		if err != nil && !errors.IsNotFound(err) {
			return false, err
		}
		return !errors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("creating kubernetes service timed out")
	}

	// make 5 more services to take up all IPs
	for i := 0; i < 5; i++ {
		if _, err := client.Core().Services(metav1.NamespaceDefault).Create(svc(i)); err != nil {
			t.Error(err)
		}
	}

	// Make another service. It will fail because we're out of cluster IPs
	if _, err := client.Core().Services(metav1.NamespaceDefault).Create(svc(8)); err != nil {
		if !strings.Contains(err.Error(), "range is full") {
			t.Errorf("unexpected error text: %v", err)
		}
	} else {
		svcs, err := client.Core().Services(metav1.NamespaceAll).List(metav1.ListOptions{})
		if err != nil {
			t.Fatalf("unexpected success, and error getting the services: %v", err)
		}
		allIPs := []string{}
		for _, s := range svcs.Items {
			allIPs = append(allIPs, s.Spec.ClusterIP)
		}
		t.Fatalf("unexpected creation success. The following IPs exist: %#v. It should only be possible to allocate 2 IP addresses in this cluster.\n\n%#v", allIPs, svcs)
	}

	// Delete the first service.
	if err := client.Core().Services(metav1.NamespaceDefault).Delete(svc(1).ObjectMeta.Name, nil); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// This time creating the second service should work.
	if _, err := client.Core().Services(metav1.NamespaceDefault).Create(svc(8)); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}
}

// TestUpdateNodeObjects represents a simple version of the behavior of node checkins at steady
// state. This test allows for easy profiling of a realistic master scenario for baseline CPU
// in very large clusters. It is disabled by default - start a kube-apiserver and pass
// UPDATE_NODE_APISERVER as the host value.
func TestUpdateNodeObjects(t *testing.T) {
	server := os.Getenv("UPDATE_NODE_APISERVER")
	if len(server) == 0 {
		t.Skip("UPDATE_NODE_APISERVER is not set")
	}
	c := clienttypedv1.NewForConfigOrDie(&restclient.Config{
		QPS:  10000,
		Host: server,
		ContentConfig: restclient.ContentConfig{
			AcceptContentTypes: "application/vnd.kubernetes.protobuf",
			ContentType:        "application/vnd.kubernetes.protobuf",
		},
	})

	nodes := 400
	listers := 5
	watchers := 50
	iterations := 10000

	for i := 0; i < nodes*6; i++ {
		c.Nodes().Delete(fmt.Sprintf("node-%d", i), nil)
		_, err := c.Nodes().Create(&v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("node-%d", i),
			},
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	for k := 0; k < listers; k++ {
		go func(lister int) {
			for i := 0; i < iterations; i++ {
				_, err := c.Nodes().List(metav1.ListOptions{})
				if err != nil {
					fmt.Printf("[list:%d] error after %d: %v\n", lister, i, err)
					break
				}
				time.Sleep(time.Duration(lister)*10*time.Millisecond + 1500*time.Millisecond)
			}
		}(k)
	}

	for k := 0; k < watchers; k++ {
		go func(lister int) {
			w, err := c.Nodes().Watch(metav1.ListOptions{})
			if err != nil {
				fmt.Printf("[watch:%d] error: %v", lister, err)
				return
			}
			i := 0
			for r := range w.ResultChan() {
				i++
				if _, ok := r.Object.(*v1.Node); !ok {
					fmt.Printf("[watch:%d] unexpected object after %d: %#v\n", lister, i, r)
				}
				if i%100 == 0 {
					fmt.Printf("[watch:%d] iteration %d ...\n", lister, i)
				}
			}
			fmt.Printf("[watch:%d] done\n", lister)
		}(k)
	}

	var wg sync.WaitGroup
	wg.Add(nodes - listers)

	for j := 0; j < nodes; j++ {
		go func(node int) {
			var lastCount int
			for i := 0; i < iterations; i++ {
				if i%100 == 0 {
					fmt.Printf("[%d] iteration %d ...\n", node, i)
				}
				if i%20 == 0 {
					_, err := c.Nodes().List(metav1.ListOptions{})
					if err != nil {
						fmt.Printf("[%d] error after %d: %v\n", node, i, err)
						break
					}
				}

				r, err := c.Nodes().List(metav1.ListOptions{
					FieldSelector:   fmt.Sprintf("metadata.name=node-%d", node),
					ResourceVersion: "0",
				})
				if err != nil {
					fmt.Printf("[%d] error after %d: %v\n", node, i, err)
					break
				}
				if len(r.Items) != 1 {
					fmt.Printf("[%d] error after %d: unexpected list count\n", node, i)
					break
				}

				n, err := c.Nodes().Get(fmt.Sprintf("node-%d", node), metav1.GetOptions{})
				if err != nil {
					fmt.Printf("[%d] error after %d: %v\n", node, i, err)
					break
				}
				if len(n.Status.Conditions) != lastCount {
					fmt.Printf("[%d] worker set %d, read %d conditions\n", node, lastCount, len(n.Status.Conditions))
					break
				}
				previousCount := lastCount
				switch {
				case i%4 == 0:
					lastCount = 1
					n.Status.Conditions = []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "foo",
						},
					}
				case i%4 == 1:
					lastCount = 2
					n.Status.Conditions = []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionFalse,
							Reason: "foo",
						},
						{
							Type:   v1.NodeDiskPressure,
							Status: v1.ConditionTrue,
							Reason: "bar",
						},
					}
				case i%4 == 2:
					lastCount = 0
					n.Status.Conditions = nil
				}
				if _, err := c.Nodes().UpdateStatus(n); err != nil {
					if !errors.IsConflict(err) {
						fmt.Printf("[%d] error after %d: %v\n", node, i, err)
						break
					}
					lastCount = previousCount
				}
			}
			wg.Done()
			fmt.Printf("[%d] done\n", node)
		}(j)
	}
	wg.Wait()
}
