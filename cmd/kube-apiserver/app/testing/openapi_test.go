// +build !race

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

// We disable the race detector here because we cross a 5 minute test
// time out threshold on CI systems; after 5 minutes the tests are
// deemed to have failed.
//
// A `make test` will use -race whereas `make bazel-test` will not. We
// want this test executed at least once and the simplest way is to
// ensure that this test is not executed when -race is in effect.

package testing

import (
	"encoding/json"
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
)

// TestOpenAPIDelegationChainPlumbing is a smoke test that checks for
// the existence of some representative paths from the
// apiextensions-server and the kube-aggregator server, both part of
// the delegation chain in kube-apiserver.
func TestOpenAPIDelegationChainPlumbing(t *testing.T) {
	config, tearDown := StartTestServerOrDie(t)
	defer tearDown()

	kubeclient, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	result := kubeclient.RESTClient().Get().AbsPath("/swagger.json").Do()
	status := 0
	result.StatusCode(&status)
	if status != 200 {
		t.Fatalf("GET /swagger.json failed: expected status=%d, got=%d", 200, status)
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
