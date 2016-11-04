// +build integration,!no-etcd

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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	clienttypedv1 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func testPrefix(t *testing.T, prefix string) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

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

func TestWatchSucceedsWithoutArgs(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

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

func autoscalingPath(resource, namespace, name string) string {
	return testapi.Autoscaling.ResourcePath(resource, namespace, name)
}

func batchPath(resource, namespace, name string) string {
	return testapi.Batch.ResourcePath(resource, namespace, name)
}

func extensionsPath(resource, namespace, name string) string {
	return testapi.Extensions.ResourcePath(resource, namespace, name)
}

func TestAutoscalingGroupBackwardCompatibility(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()
	transport := http.DefaultTransport

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		{"POST", autoscalingPath("horizontalpodautoscalers", api.NamespaceDefault, ""), hpaV1, integration.Code201, ""},
		{"GET", autoscalingPath("horizontalpodautoscalers", api.NamespaceDefault, ""), "", integration.Code200, testapi.Autoscaling.GroupVersion().String()},
		{"GET", extensionsPath("horizontalpodautoscalers", api.NamespaceDefault, ""), "", integration.Code200, testapi.Extensions.GroupVersion().String()},
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

var jobV1beta1 string = `
{
    "kind": "Job",
    "apiVersion": "extensions/v1beta1",
    "metadata": {
        "name": "pi",
        "labels": {
            "app": "pi"
        }
    },
    "spec": {
        "parallelism": 1,
        "completions": 1,
        "selector": {
            "matchLabels": {
                "app": "pi"
            }
        },
        "template": {
            "metadata": {
                "name": "pi",
                "creationTimestamp": null,
                "labels": {
                    "app": "pi"
                }
            },
            "spec": {
                "containers": [
                    {
                        "name": "pi",
                        "image": "perl",
                        "command": [
                            "perl",
                            "-Mbignum=bpi",
                            "-wle",
                            "print bpi(2000)"
                        ]
                    }
                ],
                "restartPolicy": "Never"
            }
        }
    }
}
`

var jobV1 string = `
{
    "kind": "Job",
    "apiVersion": "batch/v1",
    "metadata": {
        "name": "pi"
    },
    "spec": {
        "parallelism": 1,
        "completions": 1,
        "template": {
            "metadata": {
                "name": "pi",
                "creationTimestamp": null
            },
            "spec": {
                "containers": [
                    {
                        "name": "pi",
                        "image": "perl",
                        "command": [
                            "perl",
                            "-Mbignum=bpi",
                            "-wle",
                            "print bpi(2000)"
                        ]
                    }
                ],
                "restartPolicy": "Never"
            }
        }
    }
}
`

// TestBatchGroupBackwardCompatibility is testing that batch/v1 and ext/v1beta1
// Job share storage.  This test can be deleted when Jobs is removed from ext/v1beta1,
// (expected to happen in 1.4).
func TestBatchGroupBackwardCompatibility(t *testing.T) {
	if *testapi.Batch.GroupVersion() == v2alpha1.SchemeGroupVersion {
		t.Skip("Shared job storage is not required for batch/v2alpha1.")
	}
	_, s := framework.RunAMaster(nil)
	defer s.Close()
	transport := http.DefaultTransport

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		// Post a v1 and get back both as v1beta1 and as v1.
		{"POST", batchPath("jobs", api.NamespaceDefault, ""), jobV1, integration.Code201, ""},
		{"GET", batchPath("jobs", api.NamespaceDefault, "pi"), "", integration.Code200, testapi.Batch.GroupVersion().String()},
		{"GET", extensionsPath("jobs", api.NamespaceDefault, "pi"), "", integration.Code200, testapi.Extensions.GroupVersion().String()},
		{"DELETE", batchPath("jobs", api.NamespaceDefault, "pi"), "", integration.Code200, registered.GroupOrDie(api.GroupName).GroupVersion.String()}, // status response
		// Post a v1beta1 and get back both as v1beta1 and as v1.
		{"POST", extensionsPath("jobs", api.NamespaceDefault, ""), jobV1beta1, integration.Code201, ""},
		{"GET", batchPath("jobs", api.NamespaceDefault, "pi"), "", integration.Code200, testapi.Batch.GroupVersion().String()},
		{"GET", extensionsPath("jobs", api.NamespaceDefault, "pi"), "", integration.Code200, testapi.Extensions.GroupVersion().String()},
		{"DELETE", extensionsPath("jobs", api.NamespaceDefault, "pi"), "", integration.Code200, registered.GroupOrDie(api.GroupName).GroupVersion.String()}, //status response
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
	_, s := framework.RunAMaster(nil)
	defer s.Close()

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
	_, s := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer s.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})

	err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
		svcList, err := client.Core().Services(api.NamespaceDefault).List(api.ListOptions{})
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
			ep, err := client.Core().Endpoints(api.NamespaceDefault).Get("kubernetes")
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
	cfg.ServiceIPRange = *cidr
	_, s := framework.RunAMaster(cfg)
	defer s.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})

	svc := func(i int) *api.Service {
		return &api.Service{
			ObjectMeta: api.ObjectMeta{
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
		_, err := client.Core().Services(api.NamespaceDefault).Get("kubernetes")
		if err != nil && !errors.IsNotFound(err) {
			return false, err
		}
		return !errors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("creating kubernetes service timed out")
	}

	// make 5 more services to take up all IPs
	for i := 0; i < 5; i++ {
		if _, err := client.Core().Services(api.NamespaceDefault).Create(svc(i)); err != nil {
			t.Error(err)
		}
	}

	// Make another service. It will fail because we're out of cluster IPs
	if _, err := client.Core().Services(api.NamespaceDefault).Create(svc(8)); err != nil {
		if !strings.Contains(err.Error(), "range is full") {
			t.Errorf("unexpected error text: %v", err)
		}
	} else {
		svcs, err := client.Core().Services(api.NamespaceAll).List(api.ListOptions{})
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
	if err := client.Core().Services(api.NamespaceDefault).Delete(svc(1).ObjectMeta.Name, nil); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// This time creating the second service should work.
	if _, err := client.Core().Services(api.NamespaceDefault).Create(svc(8)); err != nil {
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
			ObjectMeta: v1.ObjectMeta{
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
				_, err := c.Nodes().List(v1.ListOptions{})
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
			w, err := c.Nodes().Watch(v1.ListOptions{})
			if err != nil {
				fmt.Printf("[watch:%d] error: %v", k, err)
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
					_, err := c.Nodes().List(v1.ListOptions{})
					if err != nil {
						fmt.Printf("[%d] error after %d: %v\n", node, i, err)
						break
					}
				}

				r, err := c.Nodes().List(v1.ListOptions{
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

				n, err := c.Nodes().Get(fmt.Sprintf("node-%d", node))
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
				case i%4 == 1:
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
