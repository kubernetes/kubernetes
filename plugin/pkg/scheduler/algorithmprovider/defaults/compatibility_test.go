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

package defaults

import (
	"reflect"
	"testing"

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
		// Do not change this JSON. A failure indicates backwards compatibility with 1.0 was broken.
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

		// Do not change this JSON after 1.1 is tagged. A failure indicates backwards compatibility with 1.1 was broken.
		"1.1": {
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
		    {"name": "PodFitsHostPorts"}
		  ],"priorities": [
		    {"name": "SelectorSpreadPriority",   "weight": 2}
		  ]
		}`,
			ExpectedPolicy: schedulerapi.Policy{
				Predicates: []schedulerapi.PredicatePolicy{
					{Name: "PodFitsHostPorts"},
				},
				Priorities: []schedulerapi.PriorityPolicy{
					{Name: "SelectorSpreadPriority", Weight: 2},
				},
			},
		},
	}

	for v, tc := range schedulerFiles {
		policy := schedulerapi.Policy{}
		err := latestschedulerapi.Codec.DecodeInto([]byte(tc.JSON), &policy)
		if err != nil {
			t.Errorf("%s: Error decoding: %v", v, err)
			continue
		}
		if !reflect.DeepEqual(policy, tc.ExpectedPolicy) {
			t.Errorf("%s: Expected:\n\t%#v\nGot:\n\t%#v", v, tc.ExpectedPolicy, policy)
		}
		_, err = factory.NewConfigFactory(nil, nil).CreateFromConfig(policy)
		if err != nil {
			t.Errorf("%s: Error constructing: %v", v, err)
			continue
		}
	}
}
