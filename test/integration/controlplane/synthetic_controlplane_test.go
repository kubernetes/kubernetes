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

package controlplane

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	clienttypedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	// Fake values for testing.
	AliceToken string = "abc123" // username: alice.  Present in token file.
	BobToken   string = "xyz987" // username: bob.  Present in token file.
)

func testPrefix(t *testing.T, prefix string) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("GET", server.ClientConfig.Host+prefix, nil)
	if err != nil {
		t.Fatalf("couldn't create a request: %v", err)
	}

	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error getting %s prefix: %v", prefix, err)
	}
	defer resp.Body.Close()
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

func TestKubernetesService(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--advertise-address=10.1.1.1"}, framework.SharedEtcd())
	defer server.TearDownFn()

	coreClient := clientset.NewForConfigOrDie(server.ClientConfig)
	err := wait.PollImmediate(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
		if _, err := coreClient.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{}); err != nil && apierrors.IsNotFound(err) {
			return false, nil
		} else if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Expected kubernetes service to exist, got: %v", err)
	}
}

func TestEmptyList(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	u := server.ClientConfig.Host + "/api/v1/namespaces/default/pods"
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		t.Fatalf("couldn't create a request: %v", err)
	}

	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error getting response: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
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

func initStatusForbiddenControlPlaneConfig(options *options.ServerRunOptions) {
	options.Authorization.Modes = []string{"AlwaysDeny"}
}

func initUnauthorizedControlPlaneConfig(options *options.ServerRunOptions) {
	options.Authentication.Anonymous.Allow = false
}

func TestStatus(t *testing.T) {
	testCases := []struct {
		name          string
		modifyOptions func(*options.ServerRunOptions)
		statusCode    int
		reqPath       string
		reason        string
		message       string
	}{
		{
			name:       "404",
			statusCode: http.StatusNotFound,
			reqPath:    "/apis/batch/v1/namespaces/default/jobs/foo",
			reason:     "NotFound",
			message:    `jobs.batch "foo" not found`,
		},
		{
			name:          "403",
			modifyOptions: initStatusForbiddenControlPlaneConfig,
			statusCode:    http.StatusForbidden,
			reqPath:       "/apis",
			reason:        "Forbidden",
			message:       `forbidden: User "system:anonymous" cannot get path "/apis": Everything is forbidden.`,
		},
		{
			name:          "401",
			modifyOptions: initUnauthorizedControlPlaneConfig,
			statusCode:    http.StatusUnauthorized,
			reqPath:       "/apis",
			reason:        "Unauthorized",
			message:       `Unauthorized`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			_, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerRunOptions: func(options *options.ServerRunOptions) {
					if tc.modifyOptions != nil {
						tc.modifyOptions(options)
					}
				},
			})
			defer tearDownFn()

			// When modifying authenticator and authorizer, don't use
			// bearer token than will be always authorized.
			if tc.modifyOptions != nil {
				kubeConfig.BearerToken = ""
			}
			transport, err := restclient.TransportFor(kubeConfig)
			if err != nil {
				t.Fatal(err)
			}

			req, err := http.NewRequest("GET", kubeConfig.Host+tc.reqPath, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != tc.statusCode {
				t.Fatalf("got status %v instead of %s", resp.StatusCode, tc.name)
			}
			data, _ := io.ReadAll(resp.Body)
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
		})
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
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--storage-media-type=application/json"}, framework.SharedEtcd())
	defer server.TearDownFn()

	server.ClientConfig.ContentType = runtime.ContentTypeJSON
	client := clientset.NewForConfigOrDie(server.ClientConfig)

	// Computing ManagedFields is extremely inefficient for large object, e.g.
	// it may take 10s+ to just compute it if we have ~100k very small labels or
	// annotations.  This in turn may lead to timing out requests,
	// which have hardcoded timeout of 34 seconds.
	// As a result, we're using slightly larger individual labels/annotations
	// to reduce the number of those.
	const DeploymentMegabyteSize = 25000
	const DeploymentTwoMegabyteSize = 30000
	const DeploymentThreeMegabyteSize = 45000

	expectedMsgFor1MB := `etcdserver: request is too large`
	expectedMsgFor2MB := `rpc error: code = ResourceExhausted desc = trying to send message larger than max`
	expectedMsgFor3MB := `Request entity too large: limit is 3145728`
	expectedMsgForLargeAnnotation := `metadata.annotations: Too long: must have at most 262144 bytes`

	deployment1 := constructBody("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", DeploymentMegabyteSize, "labels", t)      // >1.5 MB file
	deployment2 := constructBody("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", DeploymentTwoMegabyteSize, "labels", t)   // >2 MB file
	deployment3 := constructBody("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", DeploymentThreeMegabyteSize, "labels", t) // >3 MB file

	deployment4 := constructBody("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", DeploymentMegabyteSize, "annotations", t)

	deployment5 := constructBody("sample0123456789/sample0123456789", 2*DeploymentMegabyteSize, "finalizers", t)      // >1.5 MB file
	deployment6 := constructBody("sample0123456789/sample0123456789", 2*DeploymentTwoMegabyteSize, "finalizers", t)   // >2 MB file
	deployment7 := constructBody("sample0123456789/sample0123456789", 2*DeploymentThreeMegabyteSize, "finalizers", t) // >3 MB file

	requests := []struct {
		size             string
		deploymentObject *appsv1.Deployment
		expectedMessage  string
	}{
		{"1 MB labels", deployment1, expectedMsgFor1MB},
		{"2 MB labels", deployment2, expectedMsgFor2MB},
		{"3 MB labels", deployment3, expectedMsgFor3MB},
		{"1 MB annotations", deployment4, expectedMsgForLargeAnnotation},
		{"1 MB finalizers", deployment5, expectedMsgFor1MB},
		{"2 MB finalizers", deployment6, expectedMsgFor2MB},
		{"3 MB finalizers", deployment7, expectedMsgFor3MB},
	}

	for _, r := range requests {
		t.Run(r.size, func(t *testing.T) {
			_, err := client.AppsV1().Deployments(metav1.NamespaceDefault).Create(context.TODO(), r.deploymentObject, metav1.CreateOptions{})
			if err == nil {
				t.Errorf("got: <nil>;want: %s", r.expectedMessage)
			}
			if err != nil {
				if !strings.Contains(err.Error(), r.expectedMessage) {
					t.Errorf("got: %s;want: %s", err.Error(), r.expectedMessage)
				}
			}
		})
	}
}

func TestWatchSucceedsWithoutArgs(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("GET", server.ClientConfig.Host+"/api/v1/namespaces?watch=1", nil)
	if err != nil {
		t.Fatalf("couldn't create a request: %v", err)
	}

	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error getting response: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}
}

var hpaV1 = `
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

var deploymentApps = `
{
  "apiVersion": "apps/v1",
  "kind": "Deployment",
  "metadata": {
     "name": "test-deployment2",
     "namespace": "default"
  },
  "spec": {
    "replicas": 1,
    "selector": {
      "matchLabels": {
        "app": "nginx0"
      }
    },
    "template": {
      "metadata": {
        "labels": {
          "app": "nginx0"
        }
      },
      "spec": {
        "containers": [{
          "name": "nginx",
          "image": "registry.k8s.io/nginx:1.7.9"
        }]
      }
    }
  }
}
`

func autoscalingPath(resource, namespace, name string) string {
	if namespace != "" {
		namespace = path.Join("namespaces", namespace)
	}
	return path.Join("/apis/autoscaling/v1", namespace, resource, name)
}

func appsPath(resource, namespace, name string) string {
	if namespace != "" {
		namespace = path.Join("namespaces", namespace)
	}
	return path.Join("/apis/apps/v1", namespace, resource, name)
}

func TestAutoscalingGroupBackwardCompatibility(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		{"POST", autoscalingPath("horizontalpodautoscalers", metav1.NamespaceDefault, ""), hpaV1, integration.Code201, ""},
		{"GET", autoscalingPath("horizontalpodautoscalers", metav1.NamespaceDefault, ""), "", integration.Code200, "autoscaling/v1"},
	}

	for _, r := range requests {
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, server.ClientConfig.Host+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			defer resp.Body.Close()
			b, _ := io.ReadAll(resp.Body)
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
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		// Post to apps endpoint and get back from apps
		{"POST", appsPath("deployments", metav1.NamespaceDefault, ""), deploymentApps, integration.Code201, ""},
		{"GET", appsPath("deployments", metav1.NamespaceDefault, "test-deployment2"), "", integration.Code200, "apps/v1"},
		// set propagationPolicy=Orphan to force the object to be returned so we can check the apiVersion (otherwise, we just get a status object back)
		{"DELETE", appsPath("deployments", metav1.NamespaceDefault, "test-deployment2") + "?propagationPolicy=Orphan", "", integration.Code200, "apps/v1"},
	}

	for _, r := range requests {
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, server.ClientConfig.Host+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			defer resp.Body.Close()
			b, _ := io.ReadAll(resp.Body)
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
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("GET", server.ClientConfig.Host+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error getting api: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %v instead of 200 OK", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if resp.Header.Get("Content-Type") != "application/json" {
		t.Errorf("unexpected content: %s", body)
	}
	if err := json.Unmarshal(body, &map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("GET", server.ClientConfig.Host+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application/yaml")
	resp, err = transport.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ = io.ReadAll(resp.Body)
	if resp.Header.Get("Content-Type") != "application/yaml" {
		t.Errorf("unexpected content: %s", body)
	}
	t.Logf("body: %s", body)
	if err := yaml.Unmarshal(body, &map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("GET", server.ClientConfig.Host+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application/json, application/yaml")
	resp, err = transport.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ = io.ReadAll(resp.Body)
	if resp.Header.Get("Content-Type") != "application/json" {
		t.Errorf("unexpected content: %s", body)
	}
	t.Logf("body: %s", body)
	if err := yaml.Unmarshal(body, &map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("GET", server.ClientConfig.Host+"/api/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application") // not a valid media type
	resp, err = transport.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotAcceptable {
		t.Errorf("unexpected error from the server")
	}
}

func countEndpoints(eps *corev1.Endpoints) int {
	count := 0
	for i := range eps.Subsets {
		count += len(eps.Subsets[i].Addresses) * len(eps.Subsets[i].Ports)
	}
	return count
}

func TestAPIServerService(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--advertise-address=10.1.1.1"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
		svcList, err := client.CoreV1().Services(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{})
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
			ep, err := client.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
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

// TestUpdateNodeObjects represents a simple version of the behavior of node checkins at steady
// state. This test allows for easy profiling of a realistic primary scenario for baseline CPU
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
		c.Nodes().Delete(context.TODO(), fmt.Sprintf("node-%d", i), metav1.DeleteOptions{})
		_, err := c.Nodes().Create(context.TODO(), &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("node-%d", i),
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}

	for k := 0; k < listers; k++ {
		go func(lister int) {
			for i := 0; i < iterations; i++ {
				_, err := c.Nodes().List(context.TODO(), metav1.ListOptions{})
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
			w, err := c.Nodes().Watch(context.TODO(), metav1.ListOptions{})
			if err != nil {
				fmt.Printf("[watch:%d] error: %v", lister, err)
				return
			}
			i := 0
			for r := range w.ResultChan() {
				i++
				if _, ok := r.Object.(*corev1.Node); !ok {
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
					_, err := c.Nodes().List(context.TODO(), metav1.ListOptions{})
					if err != nil {
						fmt.Printf("[%d] error after %d: %v\n", node, i, err)
						break
					}
				}

				r, err := c.Nodes().List(context.TODO(), metav1.ListOptions{
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

				n, err := c.Nodes().Get(context.TODO(), fmt.Sprintf("node-%d", node), metav1.GetOptions{})
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
					n.Status.Conditions = []corev1.NodeCondition{
						{
							Type:   corev1.NodeReady,
							Status: corev1.ConditionTrue,
							Reason: "foo",
						},
					}
				case i%4 == 1:
					lastCount = 2
					n.Status.Conditions = []corev1.NodeCondition{
						{
							Type:   corev1.NodeReady,
							Status: corev1.ConditionFalse,
							Reason: "foo",
						},
						{
							Type:   corev1.NodeDiskPressure,
							Status: corev1.ConditionTrue,
							Reason: "bar",
						},
					}
				case i%4 == 2:
					lastCount = 0
					n.Status.Conditions = nil
				}
				if _, err := c.Nodes().UpdateStatus(context.TODO(), n, metav1.UpdateOptions{}); err != nil {
					if !apierrors.IsConflict(err) {
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
