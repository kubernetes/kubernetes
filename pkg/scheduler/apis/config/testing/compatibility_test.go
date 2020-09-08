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

package testing

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler/profile"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
)

type testCase struct {
	name          string
	JSON          string
	featureGates  map[featuregate.Feature]bool
	wantPlugins   map[string][]config.Plugin
	wantExtenders []config.Extender
}

func TestCompatibility_v1_Scheduler(t *testing.T) {
	// Add serialized versions of scheduler config that exercise available options to ensure compatibility between releases
	testcases := []testCase{
		// This is a special test for the "composite" predicate "GeneralPredicate". GeneralPredicate is a combination
		// of predicates, and here we test that if given, it is mapped to the set of plugins that should be executed.
		{
			name: "GeneralPredicate",
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
			{"name": "GeneralPredicates"}
                  ],
		  "priorities": [
                  ]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "TaintToleration"},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		// This is a special test for the case where a policy is specified without specifying any filters.
		{
			name: "MandatoryFilters",
			JSON: `{
				"kind": "Policy",
				"apiVersion": "v1",
				"predicates": [
				],
				"priorities": [
				]
			}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.0",
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
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
				},
				"PreScorePlugin": {{Name: "SelectorSpread"}},
				"ScorePlugin": {
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeLabel", Weight: 4},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "ServiceAffinity", Weight: 3},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.1",
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
			{"name": "TestServiceAntiAffinity1",  "weight": 3, "argument": {"serviceAntiAffinity": {"label": "zone"}}},
			{"name": "TestServiceAntiAffinity2",  "weight": 3, "argument": {"serviceAntiAffinity": {"label": "region"}}},
			{"name": "TestLabelPreference1",      "weight": 4, "argument": {"labelPreference": {"label": "bar", "presence":true}}},
			{"name": "TestLabelPreference2",      "weight": 4, "argument": {"labelPreference": {"label": "foo", "presence":false}}}
		  ]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
				},
				"PreScorePlugin": {{Name: "SelectorSpread"}},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeLabel", Weight: 8}, // Weight is 4 * number of LabelPreference priorities
					{Name: "SelectorSpread", Weight: 2},
					{Name: "ServiceAffinity", Weight: 6}, // Weight is the 3 * number of custom ServiceAntiAffinity priorities
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.2",
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
			{"name": "MaxAzureDiskVolumeCount"},
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
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeZone"},
				},
				"PreScorePlugin": {{Name: "SelectorSpread"}},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodeLabel", Weight: 4},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "ServiceAffinity", Weight: 3},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.3",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
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
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.4",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
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
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.7",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
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
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"BindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.7 was missing json tags on the BindVerb field and required "BindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
			}},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.8",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
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
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.8 became case-insensitive and tolerated "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
			}},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.9",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "CheckVolumeBinding"},
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
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin":    {{Name: "DefaultBinder"}},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.9 was case-insensitive and tolerated "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
			}},
		},

		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.10",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "CheckVolumeBinding"},
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
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true,
			"managedResources": [{"name":"example.com/foo","ignoredByScheduler":true}],
			"ignorable":true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin":    {{Name: "DefaultBinder"}},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.10 was case-insensitive and tolerated "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{{Name: "example.com/foo", IgnoredByScheduler: true}},
				Ignorable:        true,
			}},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.11",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "CheckVolumeBinding"},
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
			{"name": "MostRequestedPriority",   "weight": 2},
			{
				"name": "RequestedToCapacityRatioPriority",
				"weight": 2,
				"argument": {
				"requestedToCapacityRatioArguments": {
					"shape": [
						{"utilization": 0,  "score": 0},
						{"utilization": 50, "score": 7}
					]
				}
			}}
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true,
			"managedResources": [{"name":"example.com/foo","ignoredByScheduler":true}],
			"ignorable":true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "RequestedToCapacityRatio", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin":    {{Name: "DefaultBinder"}},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{{Name: "example.com/foo", IgnoredByScheduler: true}},
				Ignorable:        true,
			}},
		},
		// Do not change this JSON after the corresponding release has been tagged.
		// A failure indicates backwards compatibility with the specified release was broken.
		{
			name: "1.12",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MaxCSIVolumeCountPred"},
			{"name": "MatchInterPodAffinity"},
			{"name": "CheckVolumeBinding"},
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
			{"name": "MostRequestedPriority",   "weight": 2},
			{
				"name": "RequestedToCapacityRatioPriority",
				"weight": 2,
				"argument": {
				"requestedToCapacityRatioArguments": {
					"shape": [
						{"utilization": 0,  "score": 0},
						{"utilization": 50, "score": 7}
					]
				}
			}}
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true,
			"managedResources": [{"name":"example.com/foo","ignoredByScheduler":true}],
			"ignorable":true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "RequestedToCapacityRatio", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin":    {{Name: "DefaultBinder"}},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{{Name: "example.com/foo", IgnoredByScheduler: true}},
				Ignorable:        true,
			}},
		},
		{
			name: "1.14",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MaxCSIVolumeCountPred"},
                        {"name": "MaxCinderVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "CheckVolumeBinding"},
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
			{"name": "MostRequestedPriority",   "weight": 2},
			{
				"name": "RequestedToCapacityRatioPriority",
				"weight": 2,
				"argument": {
				"requestedToCapacityRatioArguments": {
					"shape": [
						{"utilization": 0,  "score": 0},
						{"utilization": 50, "score": 7}
					]
				}
			}}
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true,
			"managedResources": [{"name":"example.com/foo","ignoredByScheduler":true}],
			"ignorable":true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "CinderLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "RequestedToCapacityRatio", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin":    {{Name: "DefaultBinder"}},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{{Name: "example.com/foo", IgnoredByScheduler: true}},
				Ignorable:        true,
			}},
		},
		{
			name: "1.16",
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
			{"name": "MaxEBSVolumeCount"},
			{"name": "MaxGCEPDVolumeCount"},
			{"name": "MaxAzureDiskVolumeCount"},
			{"name": "MaxCSIVolumeCountPred"},
                        {"name": "MaxCinderVolumeCount"},
			{"name": "MatchInterPodAffinity"},
			{"name": "CheckVolumeBinding"},
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
			{"name": "MostRequestedPriority",   "weight": 2},
			{
				"name": "RequestedToCapacityRatioPriority",
				"weight": 2,
				"argument": {
				"requestedToCapacityRatioArguments": {
					"shape": [
						{"utilization": 0,  "score": 0},
						{"utilization": 50, "score": 7}
					],
					"resources": [
						{"name": "intel.com/foo", "weight": 3},
						{"name": "intel.com/bar", "weight": 5}
					]
				}
			}}
		  ],"extenders": [{
			"urlPrefix":        "/prefix",
			"filterVerb":       "filter",
			"prioritizeVerb":   "prioritize",
			"weight":           1,
			"bindVerb":         "bind",
			"enableHttps":      true,
			"tlsConfig":        {"Insecure":true},
			"httpTimeout":      1,
			"nodeCacheCapable": true,
			"managedResources": [{"name":"example.com/foo","ignoredByScheduler":true}],
			"ignorable":true
		  }]
		}`,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "CinderLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "SelectorSpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "RequestedToCapacityRatio", Weight: 2},
					{Name: "SelectorSpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin":    {{Name: "DefaultBinder"}},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      metav1.Duration{Duration: 1},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{{Name: "example.com/foo", IgnoredByScheduler: true}},
				Ignorable:        true,
			}},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			for feature, value := range tc.featureGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, value)()
			}

			policyConfigMap := v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: "scheduler-custom-policy-config"},
				Data:       map[string]string{config.SchedulerPolicyConfigMapKey: tc.JSON},
			}
			client := fake.NewSimpleClientset(&policyConfigMap)
			algorithmSrc := config.SchedulerAlgorithmSource{
				Policy: &config.SchedulerPolicySource{
					ConfigMap: &config.SchedulerPolicyConfigMapSource{
						Namespace: policyConfigMap.Namespace,
						Name:      policyConfigMap.Name,
					},
				},
			}
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			recorderFactory := profile.NewRecorderFactory(events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()}))

			sched, err := scheduler.New(
				client,
				informerFactory,
				recorderFactory,
				make(chan struct{}),
				scheduler.WithAlgorithmSource(algorithmSrc),
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			defProf := sched.Profiles["default-scheduler"]
			gotPlugins := defProf.Framework.ListPlugins()
			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}

			gotExtenders := sched.Algorithm.Extenders()
			var wantExtenders []*core.HTTPExtender
			for _, e := range tc.wantExtenders {
				extender, err := core.NewHTTPExtender(&e)
				if err != nil {
					t.Errorf("Error transforming extender: %+v", e)
				}
				wantExtenders = append(wantExtenders, extender.(*core.HTTPExtender))
			}
			for i := range gotExtenders {
				if !core.Equal(wantExtenders[i], gotExtenders[i].(*core.HTTPExtender)) {
					t.Errorf("Got extender #%d %+v, want %+v", i, gotExtenders[i], wantExtenders[i])
				}
			}
		})
	}
}

func TestAlgorithmProviderCompatibility(t *testing.T) {
	// Add serialized versions of scheduler config that exercise available options to ensure compatibility between releases
	defaultPlugins := map[string][]config.Plugin{
		"QueueSortPlugin": {
			{Name: "PrioritySort"},
		},
		"PreFilterPlugin": {
			{Name: "NodeResourcesFit"},
			{Name: "NodePorts"},
			{Name: "PodTopologySpread"},
			{Name: "InterPodAffinity"},
			{Name: "VolumeBinding"},
		},
		"FilterPlugin": {
			{Name: "NodeUnschedulable"},
			{Name: "NodeResourcesFit"},
			{Name: "NodeName"},
			{Name: "NodePorts"},
			{Name: "NodeAffinity"},
			{Name: "VolumeRestrictions"},
			{Name: "TaintToleration"},
			{Name: "EBSLimits"},
			{Name: "GCEPDLimits"},
			{Name: "NodeVolumeLimits"},
			{Name: "AzureDiskLimits"},
			{Name: "VolumeBinding"},
			{Name: "VolumeZone"},
			{Name: "PodTopologySpread"},
			{Name: "InterPodAffinity"},
		},
		"PostFilterPlugin": {
			{Name: "DefaultPreemption"},
		},
		"PreScorePlugin": {
			{Name: "InterPodAffinity"},
			{Name: "PodTopologySpread"},
			{Name: "TaintToleration"},
			{Name: "SelectorSpread"},
		},
		"ScorePlugin": {
			{Name: "NodeResourcesBalancedAllocation", Weight: 1},
			{Name: "ImageLocality", Weight: 1},
			{Name: "InterPodAffinity", Weight: 1},
			{Name: "NodeResourcesLeastAllocated", Weight: 1},
			{Name: "NodeAffinity", Weight: 1},
			{Name: "NodePreferAvoidPods", Weight: 10000},
			{Name: "PodTopologySpread", Weight: 2},
			{Name: "TaintToleration", Weight: 1},
			{Name: "SelectorSpread", Weight: 1},
		},
		"BindPlugin":    {{Name: "DefaultBinder"}},
		"ReservePlugin": {{Name: "VolumeBinding"}},
		"PreBindPlugin": {{Name: "VolumeBinding"}},
	}

	testcases := []struct {
		name        string
		provider    string
		wantPlugins map[string][]config.Plugin
	}{
		{
			name:        "No Provider specified",
			wantPlugins: defaultPlugins,
		},
		{
			name:        "DefaultProvider",
			provider:    config.SchedulerDefaultProviderName,
			wantPlugins: defaultPlugins,
		},
		{
			name:     "ClusterAutoscalerProvider",
			provider: algorithmprovider.ClusterAutoscalerProvider,
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {
					{Name: "PrioritySort"},
				},
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
					{Name: "VolumeBinding"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
				},
				"PostFilterPlugin": {
					{Name: "DefaultPreemption"},
				},
				"PreScorePlugin": {
					{Name: "InterPodAffinity"},
					{Name: "PodTopologySpread"},
					{Name: "TaintToleration"},
					{Name: "SelectorSpread"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesMostAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "PodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 1},
					{Name: "SelectorSpread", Weight: 1},
				},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
				"BindPlugin":    {{Name: "DefaultBinder"}},
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var opts []scheduler.Option
			if len(tc.provider) != 0 {
				opts = append(opts, scheduler.WithAlgorithmSource(config.SchedulerAlgorithmSource{
					Provider: &tc.provider,
				}))
			}

			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			recorderFactory := profile.NewRecorderFactory(events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()}))

			sched, err := scheduler.New(
				client,
				informerFactory,
				recorderFactory,
				make(chan struct{}),
				opts...,
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			defProf := sched.Profiles["default-scheduler"]
			gotPlugins := defProf.ListPlugins()
			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}
}

func TestPluginsConfigurationCompatibility(t *testing.T) {
	defaultPlugins := map[string][]config.Plugin{
		"QueueSortPlugin": {
			{Name: "PrioritySort"},
		},
		"PreFilterPlugin": {
			{Name: "NodeResourcesFit"},
			{Name: "NodePorts"},
			{Name: "PodTopologySpread"},
			{Name: "InterPodAffinity"},
			{Name: "VolumeBinding"},
		},
		"FilterPlugin": {
			{Name: "NodeUnschedulable"},
			{Name: "NodeResourcesFit"},
			{Name: "NodeName"},
			{Name: "NodePorts"},
			{Name: "NodeAffinity"},
			{Name: "VolumeRestrictions"},
			{Name: "TaintToleration"},
			{Name: "EBSLimits"},
			{Name: "GCEPDLimits"},
			{Name: "NodeVolumeLimits"},
			{Name: "AzureDiskLimits"},
			{Name: "VolumeBinding"},
			{Name: "VolumeZone"},
			{Name: "PodTopologySpread"},
			{Name: "InterPodAffinity"},
		},
		"PostFilterPlugin": {
			{Name: "DefaultPreemption"},
		},
		"PreScorePlugin": {
			{Name: "InterPodAffinity"},
			{Name: "PodTopologySpread"},
			{Name: "TaintToleration"},
			{Name: "SelectorSpread"},
		},
		"ScorePlugin": {
			{Name: "NodeResourcesBalancedAllocation", Weight: 1},
			{Name: "ImageLocality", Weight: 1},
			{Name: "InterPodAffinity", Weight: 1},
			{Name: "NodeResourcesLeastAllocated", Weight: 1},
			{Name: "NodeAffinity", Weight: 1},
			{Name: "NodePreferAvoidPods", Weight: 10000},
			{Name: "PodTopologySpread", Weight: 2},
			{Name: "TaintToleration", Weight: 1},
			{Name: "SelectorSpread", Weight: 1},
		},
		"ReservePlugin": {{Name: "VolumeBinding"}},
		"PreBindPlugin": {{Name: "VolumeBinding"}},
		"BindPlugin":    {{Name: "DefaultBinder"}},
	}

	testcases := []struct {
		name             string
		plugins          config.Plugins
		wantPlugins      map[string][]config.Plugin
		pluginConfig     []config.PluginConfig
		wantPluginConfig []config.PluginConfig
	}{
		{
			name:             "default plugins",
			wantPlugins:      defaultPlugins,
			wantPluginConfig: nil,
		},
		{
			name:        "default plugins with customized plugin config",
			wantPlugins: defaultPlugins,
			pluginConfig: []config.PluginConfig{
				{
					Name: "NodeResourcesFit",
					Args: &config.NodeResourcesFitArgs{
						IgnoredResources: []string{"foo", "bar"},
					},
				},
				{
					Name: "PodTopologySpread",
					Args: &config.PodTopologySpreadArgs{
						DefaultConstraints: []v1.TopologySpreadConstraint{
							{
								MaxSkew:           1,
								TopologyKey:       "foo",
								WhenUnsatisfiable: v1.DoNotSchedule,
							},
							{
								MaxSkew:           10,
								TopologyKey:       "bar",
								WhenUnsatisfiable: v1.ScheduleAnyway,
							},
						},
					},
				},
				{
					Name: "RequestedToCapacityRatio",
					Args: &config.RequestedToCapacityRatioArgs{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 5, Score: 5},
						},
						Resources: []config.ResourceSpec{
							{Name: "cpu", Weight: 10},
						},
					},
				},
				{
					Name: "InterPodAffinity",
					Args: &config.InterPodAffinityArgs{
						HardPodAffinityWeight: 100,
					},
				},
				{
					Name: "NodeLabel",
					Args: &config.NodeLabelArgs{
						PresentLabels:           []string{"foo", "bar"},
						AbsentLabels:            []string{"apple"},
						PresentLabelsPreference: []string{"dog"},
						AbsentLabelsPreference:  []string{"cat"},
					},
				},
				{
					Name: "ServiceAffinity",
					Args: &config.ServiceAffinityArgs{
						AffinityLabels:               []string{"foo", "bar"},
						AntiAffinityLabelsPreference: []string{"disk", "flash"},
					},
				},
				{
					Name: "VolumeBinding",
					Args: &config.VolumeBindingArgs{
						BindTimeoutSeconds: 300,
					},
				},
			},
			wantPluginConfig: []config.PluginConfig{
				{
					Name: "NodeResourcesFit",
					Args: &config.NodeResourcesFitArgs{
						IgnoredResources: []string{"foo", "bar"},
					},
				},
				{
					Name: "PodTopologySpread",
					Args: &config.PodTopologySpreadArgs{
						DefaultConstraints: []v1.TopologySpreadConstraint{
							{
								MaxSkew:           1,
								TopologyKey:       "foo",
								WhenUnsatisfiable: v1.DoNotSchedule,
							},
							{
								MaxSkew:           10,
								TopologyKey:       "bar",
								WhenUnsatisfiable: v1.ScheduleAnyway,
							},
						},
					},
				},
				{
					Name: "RequestedToCapacityRatio",
					Args: &config.RequestedToCapacityRatioArgs{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 5, Score: 5},
						},
						Resources: []config.ResourceSpec{
							{Name: "cpu", Weight: 10},
						},
					},
				},
				{
					Name: "InterPodAffinity",
					Args: &config.InterPodAffinityArgs{
						HardPodAffinityWeight: 100,
					},
				},
				{
					Name: "NodeLabel",
					Args: &config.NodeLabelArgs{
						PresentLabels:           []string{"foo", "bar"},
						AbsentLabels:            []string{"apple"},
						PresentLabelsPreference: []string{"dog"},
						AbsentLabelsPreference:  []string{"cat"},
					},
				},
				{
					Name: "ServiceAffinity",
					Args: &config.ServiceAffinityArgs{
						AffinityLabels:               []string{"foo", "bar"},
						AntiAffinityLabelsPreference: []string{"disk", "flash"},
					},
				},
				{
					Name: "VolumeBinding",
					Args: &config.VolumeBindingArgs{
						BindTimeoutSeconds: 300,
					},
				},
			},
		},
		{
			name: "disable some default plugins",
			plugins: config.Plugins{
				PreFilter: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "NodeResourcesFit"},
						{Name: "NodePorts"},
						{Name: "InterPodAffinity"},
						{Name: "PodTopologySpread"},
						{Name: "VolumeBinding"},
					},
				},
				Filter: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "NodeUnschedulable"},
						{Name: "NodeResourcesFit"},
						{Name: "NodeName"},
						{Name: "NodePorts"},
						{Name: "NodeAffinity"},
						{Name: "VolumeRestrictions"},
						{Name: "TaintToleration"},
						{Name: "EBSLimits"},
						{Name: "GCEPDLimits"},
						{Name: "NodeVolumeLimits"},
						{Name: "AzureDiskLimits"},
						{Name: "VolumeBinding"},
						{Name: "VolumeZone"},
						{Name: "InterPodAffinity"},
						{Name: "PodTopologySpread"},
					},
				},
				PostFilter: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "DefaultPreemption"},
					},
				},
				PreScore: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "InterPodAffinity"},
						{Name: "SelectorSpread"},
						{Name: "TaintToleration"},
						{Name: "PodTopologySpread"},
					},
				},
				Score: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "NodeResourcesBalancedAllocation"},
						{Name: "ImageLocality"},
						{Name: "InterPodAffinity"},
						{Name: "NodeResourcesLeastAllocated"},
						{Name: "NodeAffinity"},
						{Name: "NodePreferAvoidPods"},
						{Name: "SelectorSpread"},
						{Name: "TaintToleration"},
						{Name: "PodTopologySpread"},
					},
				},
				PreBind: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "VolumeBinding"},
					},
				},
				PostBind: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "VolumeBinding"},
					},
				},
				Reserve: &config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "VolumeBinding"},
					},
				},
			},
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {
					{Name: "PrioritySort"},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		{
			name: "reverse default plugins order with changing score weight",
			plugins: config.Plugins{
				QueueSort: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: "PrioritySort"},
					},
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
				PreFilter: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: "InterPodAffinity"},
						{Name: "NodePorts"},
						{Name: "NodeResourcesFit"},
					},
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
				Filter: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: "InterPodAffinity"},
						{Name: "VolumeZone"},
						{Name: "VolumeBinding"},
						{Name: "AzureDiskLimits"},
						{Name: "NodeVolumeLimits"},
						{Name: "GCEPDLimits"},
						{Name: "EBSLimits"},
						{Name: "TaintToleration"},
						{Name: "VolumeRestrictions"},
						{Name: "NodeAffinity"},
						{Name: "NodePorts"},
						{Name: "NodeName"},
						{Name: "NodeResourcesFit"},
						{Name: "NodeUnschedulable"},
					},
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
				PreScore: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: "TaintToleration"},
						{Name: "SelectorSpread"},
						{Name: "InterPodAffinity"},
					},
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
				Score: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: "TaintToleration", Weight: 24},
						{Name: "SelectorSpread", Weight: 24},
						{Name: "NodePreferAvoidPods", Weight: 24},
						{Name: "NodeAffinity", Weight: 24},
						{Name: "NodeResourcesLeastAllocated", Weight: 24},
						{Name: "InterPodAffinity", Weight: 24},
						{Name: "ImageLocality", Weight: 24},
						{Name: "NodeResourcesBalancedAllocation", Weight: 24},
					},
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
				Bind: &config.PluginSet{
					Enabled:  []config.Plugin{{Name: "DefaultBinder"}},
					Disabled: []config.Plugin{{Name: "*"}},
				},
			},
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {
					{Name: "PrioritySort"},
				},
				"PreFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
				},
				"FilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "VolumeZone"},
					{Name: "VolumeBinding"},
					{Name: "AzureDiskLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "GCEPDLimits"},
					{Name: "EBSLimits"},
					{Name: "TaintToleration"},
					{Name: "VolumeRestrictions"},
					{Name: "NodeAffinity"},
					{Name: "NodePorts"},
					{Name: "NodeName"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeUnschedulable"},
				},
				"PostFilterPlugin": {
					{Name: "DefaultPreemption"},
				},
				"PreScorePlugin": {
					{Name: "TaintToleration"},
					{Name: "SelectorSpread"},
					{Name: "InterPodAffinity"},
				},
				"ScorePlugin": {
					{Name: "TaintToleration", Weight: 24},
					{Name: "SelectorSpread", Weight: 24},
					{Name: "NodePreferAvoidPods", Weight: 24},
					{Name: "NodeAffinity", Weight: 24},
					{Name: "NodeResourcesLeastAllocated", Weight: 24},
					{Name: "InterPodAffinity", Weight: 24},
					{Name: "ImageLocality", Weight: 24},
					{Name: "NodeResourcesBalancedAllocation", Weight: 24},
				},
				"ReservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin": {{Name: "VolumeBinding"}},
				"BindPlugin":    {{Name: "DefaultBinder"}},
			},
			wantPluginConfig: nil,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {

			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			recorderFactory := profile.NewRecorderFactory(events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()}))

			sched, err := scheduler.New(
				client,
				informerFactory,
				recorderFactory,
				make(chan struct{}),
				scheduler.WithProfiles(config.KubeSchedulerProfile{
					SchedulerName: v1.DefaultSchedulerName,
					Plugins:       &tc.plugins,
					PluginConfig:  tc.pluginConfig,
				}),
				scheduler.WithBuildFrameworkCapturer(func(p config.KubeSchedulerProfile) {
					if p.SchedulerName != v1.DefaultSchedulerName {
						t.Errorf("unexpected scheduler name (want %q, got %q)", v1.DefaultSchedulerName, p.SchedulerName)
					}
					if diff := cmp.Diff(tc.wantPluginConfig, p.PluginConfig); diff != "" {
						t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
					}
				}),
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			defProf := sched.Profiles[v1.DefaultSchedulerName]
			gotPlugins := defProf.ListPlugins()
			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}
}
