// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/test/integration/framework"
)

func testPrefix(t *testing.T, prefix string) {
	_, s := framework.RunAMaster(t)
	// TODO: Uncomment when fix #19254
	// defer s.Close()

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

func TestExtensionsPrefix(t *testing.T) {
	testPrefix(t, "/apis/extensions/")
}

func TestWatchSucceedsWithoutArgs(t *testing.T) {
	_, s := framework.RunAMaster(t)
	// TODO: Uncomment when fix #19254
	// defer s.Close()

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

func extensionsPath(resource, namespace, name string) string {
	return testapi.Extensions.ResourcePath(resource, namespace, name)
}

func TestAutoscalingGroupBackwardCompatibility(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()
	transport := http.DefaultTransport

	requests := []struct {
		verb                string
		URL                 string
		body                string
		expectedStatusCodes map[int]bool
		expectedVersion     string
	}{
		{"POST", autoscalingPath("horizontalpodautoscalers", api.NamespaceDefault, ""), hpaV1, code201, ""},
		{"GET", autoscalingPath("horizontalpodautoscalers", api.NamespaceDefault, ""), "", code200, testapi.Autoscaling.GroupVersion().String()},
		{"GET", extensionsPath("horizontalpodautoscalers", api.NamespaceDefault, ""), "", code200, testapi.Extensions.GroupVersion().String()},
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
	_, s := framework.RunAMaster(t)
	// TODO: Uncomment when fix #19254
	// defer s.Close()

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
