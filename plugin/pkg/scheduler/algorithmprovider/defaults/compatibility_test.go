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

package defaults

import (
	"fmt"
	"reflect"
	"testing"

	"net/http/httptest"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
)

func TestCompatibility_v1_Scheduler(t *testing.T) {
	// Add serialized versions of scheduler config that exercise available options to ensure compatibility between releases
	schedulerFiles := map[string]struct {
		JSON           string
		ExpectedPolicy schedulerapi.Policy
	}{
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		"1.0": {
			JSON: `{
  "kind": "Policy",
  "apiVersion": "v1",
  "predicates": [
    {"name": "MatchNodeSelector"},
    {"name": "PodFitsResources"},
    {"name": "PodFitsPorts"},
    {"name": "NoDiskConflict"},
    {"name": "TestServiceAffinity", "argument": {"serviceAffinity" : {"labels" : ["region"]}}},
    {"name": "TestLabelsPresence",  "argument": {"labelsPresence"  : {"labels" : ["foo"], "presence":true}}}
  ],"priorities": [
    {"name": "LeastRequestedPriority",   "weight": 1},
    {"name": "ServiceSpreadingPriority", "weight": 2},
    {"name": "TestServiceAntiAffinity",  "weight": 3, "argument": {"serviceAntiAffinity": {"label": "zone"}}},
    {"name": "TestLabelPreference",      "weight": 4, "argument": {"labelPreference": {"label": "bar", "presence":true}}}
  ]
}`,
			ExpectedPolicy: schedulerapi.Policy{
				Predicates: []schedulerapi.PredicatePolicy{
					{Name: "MatchNodeSelector"},
					{Name: "PodFitsResources"},
					{Name: "PodFitsPorts"},
					{Name: "NoDiskConflict"},
					{Name: "TestServiceAffinity", Argument: &schedulerapi.PredicateArgument{ServiceAffinity: &schedulerapi.ServiceAffinity{Labels: []string{"region"}}}},
					{Name: "TestLabelsPresence", Argument: &schedulerapi.PredicateArgument{LabelsPresence: &schedulerapi.LabelsPresence{Labels: []string{"foo"}, Presence: true}}},
				},
				Priorities: []schedulerapi.PriorityPolicy{
					{Name: "LeastRequestedPriority", Weight: 1},
					{Name: "ServiceSpreadingPriority", Weight: 2},
					{Name: "TestServiceAntiAffinity", Weight: 3, Argument: &schedulerapi.PriorityArgument{ServiceAntiAffinity: &schedulerapi.ServiceAntiAffinity{Label: "zone"}}},
					{Name: "TestLabelPreference", Weight: 4, Argument: &schedulerapi.PriorityArgument{LabelPreference: &schedulerapi.LabelPreference{Label: "bar", Presence: true}}},
				},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		"1.1": {
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
			{"name": "MatchNodeSelector"},
			{"name": "PodFitsHostPorts"},
			{"name": "PodFitsResources"},
			{"name": "NoDiskConflict"},
			{"name": "HostName"},
			{"name": "TestServiceAffinity", "argument": {"serviceAffinity" : {"labels" : ["region"]}}},
			{"name": "TestLabelsPresence",  "argument": {"labelsPresence"  : {"labels" : ["foo"], "presence":true}}}
		  ],"priorities": [
			{"name": "EqualPriority",   "weight": 2},
			{"name": "LeastRequestedPriority",   "weight": 2},
			{"name": "BalancedResourceAllocation",   "weight": 2},
			{"name": "SelectorSpreadPriority",   "weight": 2},
			{"name": "TestServiceAntiAffinity",  "weight": 3, "argument": {"serviceAntiAffinity": {"label": "zone"}}},
			{"name": "TestLabelPreference",      "weight": 4, "argument": {"labelPreference": {"label": "bar", "presence":true}}}
		  ]
		}`,
			ExpectedPolicy: schedulerapi.Policy{
				Predicates: []schedulerapi.PredicatePolicy{
					{Name: "MatchNodeSelector"},
					{Name: "PodFitsHostPorts"},
					{Name: "PodFitsResources"},
					{Name: "NoDiskConflict"},
					{Name: "HostName"},
					{Name: "TestServiceAffinity", Argument: &schedulerapi.PredicateArgument{ServiceAffinity: &schedulerapi.ServiceAffinity{Labels: []string{"region"}}}},
					{Name: "TestLabelsPresence", Argument: &schedulerapi.PredicateArgument{LabelsPresence: &schedulerapi.LabelsPresence{Labels: []string{"foo"}, Presence: true}}},
				},
				Priorities: []schedulerapi.PriorityPolicy{
					{Name: "EqualPriority", Weight: 2},
					{Name: "LeastRequestedPriority", Weight: 2},
					{Name: "BalancedResourceAllocation", Weight: 2},
					{Name: "SelectorSpreadPriority", Weight: 2},
					{Name: "TestServiceAntiAffinity", Weight: 3, Argument: &schedulerapi.PriorityArgument{ServiceAntiAffinity: &schedulerapi.ServiceAntiAffinity{Label: "zone"}}},
					{Name: "TestLabelPreference", Weight: 4, Argument: &schedulerapi.PriorityArgument{LabelPreference: &schedulerapi.LabelPreference{Label: "bar", Presence: true}}},
				},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		"1.2": {
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
			{"name": "MatchNodeSelector"},
			{"name": "PodFitsResources"},
			{"name": "PodFitsHostPorts"},
			{"name": "HostName"},
			{"name": "NoDiskConflict"},
			{"name": "NoVolumeZoneConflict"},
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "TestServiceAffinity", "argument": {"serviceAffinity" : {"labels" : ["region"]}}},
			{"name": "TestLabelsPresence",  "argument": {"labelsPresence"  : {"labels" : ["foo"], "presence":true}}}
		  ],"priorities": [
			{"name": "EqualPriority",   "weight": 2},
			{"name": "NodeAffinityPriority",   "weight": 2},
			{"name": "ImageLocalityPriority",   "weight": 2},
			{"name": "LeastRequestedPriority",   "weight": 2},
			{"name": "BalancedResourceAllocation",   "weight": 2},
			{"name": "SelectorSpreadPriority",   "weight": 2},
			{"name": "TestServiceAntiAffinity",  "weight": 3, "argument": {"serviceAntiAffinity": {"label": "zone"}}},
			{"name": "TestLabelPreference",      "weight": 4, "argument": {"labelPreference": {"label": "bar", "presence":true}}}
		  ]
		}`,
			ExpectedPolicy: schedulerapi.Policy{
				Predicates: []schedulerapi.PredicatePolicy{
					{Name: "MatchNodeSelector"},
					{Name: "PodFitsResources"},
					{Name: "PodFitsHostPorts"},
					{Name: "HostName"},
					{Name: "NoDiskConflict"},
					{Name: "NoVolumeZoneConflict"},
					{Name: "MaxEBSVolumeCount"},
					{Name: "MaxGCEPDVolumeCount"},
					{Name: "TestServiceAffinity", Argument: &schedulerapi.PredicateArgument{ServiceAffinity: &schedulerapi.ServiceAffinity{Labels: []string{"region"}}}},
					{Name: "TestLabelsPresence", Argument: &schedulerapi.PredicateArgument{LabelsPresence: &schedulerapi.LabelsPresence{Labels: []string{"foo"}, Presence: true}}},
				},
				Priorities: []schedulerapi.PriorityPolicy{
					{Name: "EqualPriority", Weight: 2},
					{Name: "NodeAffinityPriority", Weight: 2},
					{Name: "ImageLocalityPriority", Weight: 2},
					{Name: "LeastRequestedPriority", Weight: 2},
					{Name: "BalancedResourceAllocation", Weight: 2},
					{Name: "SelectorSpreadPriority", Weight: 2},
					{Name: "TestServiceAntiAffinity", Weight: 3, Argument: &schedulerapi.PriorityArgument{ServiceAntiAffinity: &schedulerapi.ServiceAntiAffinity{Label: "zone"}}},
					{Name: "TestLabelPreference", Weight: 4, Argument: &schedulerapi.PriorityArgument{LabelPreference: &schedulerapi.LabelPreference{Label: "bar", Presence: true}}},
				},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		"1.3": {
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
			{"name": "MatchNodeSelector"},
			{"name": "PodFitsResources"},
			{"name": "PodFitsHostPorts"},
			{"name": "HostName"},
			{"name": "NoDiskConflict"},
			{"name": "NoVolumeZoneConflict"},
			{"name": "PodToleratesNodeTaints"},
			{"name": "CheckNodeMemoryPressure"},
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "GeneralPredicates"},
			{"name": "TestServiceAffinity", "argument": {"serviceAffinity" : {"labels" : ["region"]}}},
			{"name": "TestLabelsPresence",  "argument": {"labelsPresence"  : {"labels" : ["foo"], "presence":true}}}
		  ],"priorities": [
			{"name": "EqualPriority",   "weight": 2},
			{"name": "ImageLocalityPriority",   "weight": 2},
			{"name": "LeastRequestedPriority",   "weight": 2},
			{"name": "BalancedResourceAllocation",   "weight": 2},
			{"name": "SelectorSpreadPriority",   "weight": 2},
			{"name": "NodeAffinityPriority",   "weight": 2},
			{"name": "TaintTolerationPriority",   "weight": 2},
			{"name": "InterPodAffinityPriority",   "weight": 2}
		  ]
		}`,
			ExpectedPolicy: schedulerapi.Policy{
				Predicates: []schedulerapi.PredicatePolicy{
					{Name: "MatchNodeSelector"},
					{Name: "PodFitsResources"},
					{Name: "PodFitsHostPorts"},
					{Name: "HostName"},
					{Name: "NoDiskConflict"},
					{Name: "NoVolumeZoneConflict"},
					{Name: "PodToleratesNodeTaints"},
					{Name: "CheckNodeMemoryPressure"},
					{Name: "MaxEBSVolumeCount"},
					{Name: "MaxGCEPDVolumeCount"},
					{Name: "MatchInterPodAffinity"},
					{Name: "GeneralPredicates"},
					{Name: "TestServiceAffinity", Argument: &schedulerapi.PredicateArgument{ServiceAffinity: &schedulerapi.ServiceAffinity{Labels: []string{"region"}}}},
					{Name: "TestLabelsPresence", Argument: &schedulerapi.PredicateArgument{LabelsPresence: &schedulerapi.LabelsPresence{Labels: []string{"foo"}, Presence: true}}},
				},
				Priorities: []schedulerapi.PriorityPolicy{
					{Name: "EqualPriority", Weight: 2},
					{Name: "ImageLocalityPriority", Weight: 2},
					{Name: "LeastRequestedPriority", Weight: 2},
					{Name: "BalancedResourceAllocation", Weight: 2},
					{Name: "SelectorSpreadPriority", Weight: 2},
					{Name: "NodeAffinityPriority", Weight: 2},
					{Name: "TaintTolerationPriority", Weight: 2},
					{Name: "InterPodAffinityPriority", Weight: 2},
				},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		"1.4": {
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
			{"name": "MatchNodeSelector"},
			{"name": "PodFitsResources"},
			{"name": "PodFitsHostPorts"},
			{"name": "HostName"},
			{"name": "NoDiskConflict"},
			{"name": "NoVolumeZoneConflict"},
			{"name": "PodToleratesNodeTaints"},
			{"name": "CheckNodeMemoryPressure"},
			{"name": "CheckNodeDiskPressure"},
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "GeneralPredicates"},
			{"name": "TestServiceAffinity", "argument": {"serviceAffinity" : {"labels" : ["region"]}}},
			{"name": "TestLabelsPresence",  "argument": {"labelsPresence"  : {"labels" : ["foo"], "presence":true}}}
		  ],"priorities": [
			{"name": "EqualPriority",   "weight": 2},
			{"name": "ImageLocalityPriority",   "weight": 2},
			{"name": "LeastRequestedPriority",   "weight": 2},
			{"name": "BalancedResourceAllocation",   "weight": 2},
			{"name": "SelectorSpreadPriority",   "weight": 2},
			{"name": "NodePreferAvoidPodsPriority",   "weight": 2},
			{"name": "NodeAffinityPriority",   "weight": 2},
			{"name": "TaintTolerationPriority",   "weight": 2},
			{"name": "InterPodAffinityPriority",   "weight": 2},
			{"name": "MostRequestedPriority",   "weight": 2}
		  ]
		}`,
			ExpectedPolicy: schedulerapi.Policy{
				Predicates: []schedulerapi.PredicatePolicy{
					{Name: "MatchNodeSelector"},
					{Name: "PodFitsResources"},
					{Name: "PodFitsHostPorts"},
					{Name: "HostName"},
					{Name: "NoDiskConflict"},
					{Name: "NoVolumeZoneConflict"},
					{Name: "PodToleratesNodeTaints"},
					{Name: "CheckNodeMemoryPressure"},
					{Name: "CheckNodeDiskPressure"},
					{Name: "MaxEBSVolumeCount"},
					{Name: "MaxGCEPDVolumeCount"},
					{Name: "MatchInterPodAffinity"},
					{Name: "GeneralPredicates"},
					{Name: "TestServiceAffinity", Argument: &schedulerapi.PredicateArgument{ServiceAffinity: &schedulerapi.ServiceAffinity{Labels: []string{"region"}}}},
					{Name: "TestLabelsPresence", Argument: &schedulerapi.PredicateArgument{LabelsPresence: &schedulerapi.LabelsPresence{Labels: []string{"foo"}, Presence: true}}},
				},
				Priorities: []schedulerapi.PriorityPolicy{
					{Name: "EqualPriority", Weight: 2},
					{Name: "ImageLocalityPriority", Weight: 2},
					{Name: "LeastRequestedPriority", Weight: 2},
					{Name: "BalancedResourceAllocation", Weight: 2},
					{Name: "SelectorSpreadPriority", Weight: 2},
					{Name: "NodePreferAvoidPodsPriority", Weight: 2},
					{Name: "NodeAffinityPriority", Weight: 2},
					{Name: "TaintTolerationPriority", Weight: 2},
					{Name: "InterPodAffinityPriority", Weight: 2},
					{Name: "MostRequestedPriority", Weight: 2},
				},
			},
		},
	}

	registeredPredicates := sets.NewString(factory.ListRegisteredFitPredicates()...)
	registeredPriorities := sets.NewString(factory.ListRegisteredPriorityFunctions()...)
	seenPredicates := sets.NewString()
	seenPriorities := sets.NewString()

	for v, tc := range schedulerFiles {
		fmt.Printf("%s: Testing scheduler config\n", v)

		policy := schedulerapi.Policy{}
		if err := runtime.DecodeInto(latestschedulerapi.Codec, []byte(tc.JSON), &policy); err != nil {
			t.Errorf("%s: Error decoding: %v", v, err)
			continue
		}
		for _, predicate := range policy.Predicates {
			seenPredicates.Insert(predicate.Name)
		}
		for _, priority := range policy.Priorities {
			seenPriorities.Insert(priority.Name)
		}
		if !reflect.DeepEqual(policy, tc.ExpectedPolicy) {
			t.Errorf("%s: Expected:\n\t%#v\nGot:\n\t%#v", v, tc.ExpectedPolicy, policy)
		}

		handler := utiltesting.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})

		if _, err := factory.NewConfigFactory(client, "some-scheduler-name", v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains).CreateFromConfig(policy); err != nil {
			t.Errorf("%s: Error constructing: %v", v, err)
			continue
		}
	}

	if !seenPredicates.HasAll(registeredPredicates.List()...) {
		t.Errorf("Registered predicates are missing from compatibility test (add to test stanza for version currently in development): %#v", registeredPredicates.Difference(seenPredicates).List())
	}
	if !seenPriorities.HasAll(registeredPriorities.List()...) {
		t.Errorf("Registered priorities are missing from compatibility test (add to test stanza for version currently in development): %#v", registeredPriorities.Difference(seenPriorities).List())
	}
}
