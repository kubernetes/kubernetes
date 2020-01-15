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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
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
				"PostFilterPlugin": {{Name: "DefaultPodTopologySpread"}},
				"ScorePlugin": {
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeLabel", Weight: 4},
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				"PostFilterPlugin": {{Name: "DefaultPodTopologySpread"}},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeLabel", Weight: 8}, // Weight is 4 * number of LabelPreference priorities
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				"PostFilterPlugin": {{Name: "DefaultPodTopologySpread"}},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodeLabel", Weight: 4},
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
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
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.9 was case-insensitive and tolerated "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.10 was case-insensitive and tolerated "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      1,
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
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
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
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
			wantExtenders: []config.Extender{{
				URLPrefix:        "/prefix",
				FilterVerb:       "filter",
				PrioritizeVerb:   "prioritize",
				Weight:           1,
				BindVerb:         "bind", // 1.11 restored case-sensitivity, but allowed either "BindVerb" or "bindVerb"
				EnableHTTPS:      true,
				TLSConfig:        &config.ExtenderTLSConfig{Insecure: true},
				HTTPTimeout:      1,
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{{Name: "example.com/foo", IgnoredByScheduler: true}},
				Ignorable:        true,
			}},
		},
		{
			name: "enable alpha feature EvenPodsSpread",
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [
			{"name": "EvenPodsSpread"}
		  ],
		  "priorities": [
			{"name": "EvenPodsSpreadPriority",   "weight": 2}
		  ]
		}`,
			featureGates: map[featuregate.Feature]bool{
				features.EvenPodsSpread: true,
			},
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "PodTopologySpread"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
					{Name: "PodTopologySpread"},
				},
				"PostFilterPlugin": {
					{Name: "PodTopologySpread"},
				},
				"ScorePlugin": {
					{Name: "PodTopologySpread", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		{
			name: "enable alpha feature ResourceLimitsPriorityFunction",
			JSON: `{
		  "kind": "Policy",
		  "apiVersion": "v1",
		  "predicates": [],
		  "priorities": [
			{"name": "ResourceLimitsPriority",   "weight": 2}
		  ]
		}`,
			featureGates: map[featuregate.Feature]bool{
				features.ResourceLimitsPriorityFunction: true,
			},
			wantPlugins: map[string][]config.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PostFilterPlugin": {
					{Name: "NodeResourceLimits"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourceLimits", Weight: 2},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
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

			sched, err := scheduler.New(
				client,
				informerFactory,
				informerFactory.Core().V1().Pods(),
				nil,
				make(chan struct{}),
				scheduler.WithAlgorithmSource(algorithmSrc),
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			gotPlugins := sched.Framework.ListPlugins()
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
			{Name: "InterPodAffinity"},
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
			{Name: "InterPodAffinity"},
		},
		"PostFilterPlugin": {
			{Name: "InterPodAffinity"},
			{Name: "DefaultPodTopologySpread"},
			{Name: "TaintToleration"},
		},
		"ScorePlugin": {
			{Name: "NodeResourcesBalancedAllocation", Weight: 1},
			{Name: "ImageLocality", Weight: 1},
			{Name: "InterPodAffinity", Weight: 1},
			{Name: "NodeResourcesLeastAllocated", Weight: 1},
			{Name: "NodeAffinity", Weight: 1},
			{Name: "NodePreferAvoidPods", Weight: 10000},
			{Name: "DefaultPodTopologySpread", Weight: 1},
			{Name: "TaintToleration", Weight: 1},
		},
		"BindPlugin": {{Name: "DefaultBinder"}},
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
					{Name: "InterPodAffinity"},
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
					{Name: "InterPodAffinity"},
				},
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesMostAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "DefaultPodTopologySpread", Weight: 1},
					{Name: "TaintToleration", Weight: 1},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
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

			sched, err := scheduler.New(
				client,
				informerFactory,
				informerFactory.Core().V1().Pods(),
				nil,
				make(chan struct{}),
				opts...,
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			gotPlugins := sched.Framework.ListPlugins()
			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}

}
