/*
Copyright 2024 The Kubernetes Authors.

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

package cel

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/discovery"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/exclusion"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestExcludedResources is an open-box test to ensure that a resource that is not persisted
// is either in the allow-list or block-list of the CEL admission.
//
// see staging/src/k8s.io/apiserver/pkg/admission/exclusion/exclusion.go
// for the lists that the test is about.
func TestExcludedResources(t *testing.T) {
	server, err := apiservertesting.StartTestServer(t,
		nil, // no extra options
		// enable all APIs to cover all defined resources
		// note that it does not require feature flags enabled to
		// discovery GVRs of disabled features.
		[]string{"--runtime-config=api/all=true"},
		framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	_, resourceLists, _, err := discoveryClient.GroupsAndMaybeResources()
	if err != nil {
		t.Fatal(err)
	}
	interestedGRs := sets.New[schema.GroupResource]()
	interestedVerbCombinations := []metav1.Verbs{
		{"create"},
		{"create", "get"},
	}
	for _, rl := range resourceLists {
		for _, r := range rl.APIResources {
			slices.Sort(r.Verbs)
			for _, c := range interestedVerbCombinations {
				if slices.Equal(r.Verbs, c) {
					gv, err := schema.ParseGroupVersion(rl.GroupVersion)
					if err != nil {
						t.Fatalf("internal error: cannot parse GV from %q: %v", rl.GroupVersion, err)
					}
					interestedGRs.Insert(gv.WithResource(r.Name).GroupResource())
				}
			}
		}
	}
	existing := sets.New[schema.GroupResource]()
	existing.Insert(exclusion.Included()...)
	existing.Insert(exclusion.Excluded()...)
	shouldAdd, shouldRemove := interestedGRs.Difference(existing), existing.Difference(interestedGRs)
	if shouldAdd.Len() > 0 {
		t.Errorf("the following resources should either be in Included or Excluded in\n"+
			"pkg/kubeapiserver/admission/exclusion/resources.go\n%s",
			formatGRs(shouldAdd.UnsortedList()),
		)
	}
	if shouldRemove.Len() > 0 {
		t.Errorf("the following resources are in pkg/kubeapiserver/admission/exclusion/resources.go\n"+
			"but does not seem to be transient.\n%s",
			formatGRs(shouldRemove.UnsortedList()))
	}
}

func formatGRs(grs []schema.GroupResource) string {
	lines := make([]string, 0, len(grs))
	for _, gvr := range grs {
		item := fmt.Sprintf("%#v,", gvr)
		lines = append(lines, item)
	}
	slices.Sort(lines)
	return strings.Join(lines, "\n")
}
