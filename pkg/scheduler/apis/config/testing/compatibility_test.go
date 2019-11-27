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
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	_ "k8s.io/kubernetes/pkg/scheduler/algorithmprovider/defaults"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
)

type testCase struct {
	name             string
	JSON             string
	featureGates     map[featuregate.Feature]bool
	wantPredicates   sets.String
	wantPrioritizers sets.String
	wantPlugins      map[string][]config.Plugin
	wantExtenders    []config.Extender
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
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "TaintToleration"},
				},
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
			wantPredicates: sets.NewString(
				"PodFitsPorts",
			),
			wantPrioritizers: sets.NewString(
				"ServiceSpreadingPriority",
			),
			wantPlugins: map[string][]config.Plugin{
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "NodeLabel"},
					{Name: "ServiceAffinity"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeLabel", Weight: 4},
					{Name: "ServiceAffinity", Weight: 3},
				},
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
			{"name": "TestServiceAntiAffinity",  "weight": 3, "argument": {"serviceAntiAffinity": {"label": "zone"}}},
			{"name": "TestLabelPreference",      "weight": 4, "argument": {"labelPreference": {"label": "bar", "presence":true}}}
		  ]
		}`,
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeLabel", Weight: 4},
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "ServiceAffinity", Weight: 3},
				},
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodeLabel", Weight: 4},
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "ServiceAffinity", Weight: 3},
				},
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "DefaultPodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				},
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
			wantPredicates:   sets.NewString(),
			wantPrioritizers: sets.NewString(),
			wantPlugins: map[string][]config.Plugin{
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
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
					{Name: "PodTopologySpread"},
				},
				"ScorePlugin": {
					{Name: "PodTopologySpread", Weight: 2},
				},
			},
		},
	}
	registeredPredicates := sets.NewString(scheduler.ListRegisteredFitPredicates()...)
	registeredPriorities := sets.NewString(scheduler.ListRegisteredPriorityFunctions()...)
	seenPredicates := sets.NewString()
	seenPriorities := sets.NewString()
	mandatoryPredicates := sets.NewString()
	generalPredicateFilters := []string{"NodeResourcesFit", "NodeName", "NodePorts", "NodeAffinity"}
	filterToPredicateMap := map[string]string{
		"NodeUnschedulable":  "CheckNodeUnschedulable",
		"TaintToleration":    "PodToleratesNodeTaints",
		"NodeName":           "HostName",
		"NodePorts":          "PodFitsHostPorts",
		"NodeResourcesFit":   "PodFitsResources",
		"NodeAffinity":       "MatchNodeSelector",
		"VolumeBinding":      "CheckVolumeBinding",
		"VolumeRestrictions": "NoDiskConflict",
		"VolumeZone":         "NoVolumeZoneConflict",
		"NodeVolumeLimits":   "MaxCSIVolumeCountPred",
		"EBSLimits":          "MaxEBSVolumeCount",
		"GCEPDLimits":        "MaxGCEPDVolumeCount",
		"AzureDiskLimits":    "MaxAzureDiskVolumeCount",
		"CinderLimits":       "MaxCinderVolumeCount",
		"InterPodAffinity":   "MatchInterPodAffinity",
		"PodTopologySpread":  "EvenPodsSpread",
	}
	scoreToPriorityMap := map[string]string{
		"DefaultPodTopologySpread":        "SelectorSpreadPriority",
		"ImageLocality":                   "ImageLocalityPriority",
		"InterPodAffinity":                "InterPodAffinityPriority",
		"NodeAffinity":                    "NodeAffinityPriority",
		"NodePreferAvoidPods":             "NodePreferAvoidPodsPriority",
		"TaintToleration":                 "TaintTolerationPriority",
		"NodeResourcesLeastAllocated":     "LeastRequestedPriority",
		"NodeResourcesBalancedAllocation": "BalancedResourceAllocation",
		"NodeResourcesMostAllocated":      "MostRequestedPriority",
		"RequestedToCapacityRatio":        "RequestedToCapacityRatioPriority",
		"NodeLabel":                       "TestLabelPreference",
		"ServiceAffinity":                 "TestServiceAntiAffinity",
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			for feature, value := range tc.featureGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, value)()
			}
			defer algorithmprovider.ApplyFeatureGates()()
			if len(tc.featureGates) > 0 {
				// The enabled featuregate can register more predicates
				registeredPredicates = registeredPredicates.Union(sets.NewString(scheduler.ListRegisteredFitPredicates()...))
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
			gotPredicates := sets.NewString()
			for p := range sched.Algorithm.Predicates() {
				gotPredicates.Insert(p)
			}
			wantPredicates := tc.wantPredicates.Union(mandatoryPredicates)
			if !gotPredicates.Equal(wantPredicates) {
				t.Errorf("Got predicates %v, want %v", gotPredicates, wantPredicates)
			}

			gotPrioritizers := sets.NewString()
			for _, p := range sched.Algorithm.Prioritizers() {
				gotPrioritizers.Insert(p.Name)
			}
			if !gotPrioritizers.Equal(tc.wantPrioritizers) {
				t.Errorf("Got prioritizers %v, want %v", gotPrioritizers, tc.wantPrioritizers)
			}

			gotPlugins := sched.Framework.ListPlugins()
			for _, p := range gotPlugins["FilterPlugin"] {
				seenPredicates.Insert(filterToPredicateMap[p.Name])

			}
			if pluginsToStringSet(gotPlugins["FilterPlugin"]).HasAll(generalPredicateFilters...) {
				seenPredicates.Insert("GeneralPredicates")
			}
			for _, p := range gotPlugins["ScorePlugin"] {
				seenPriorities.Insert(scoreToPriorityMap[p.Name])

			}

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

			seenPredicates = seenPredicates.Union(gotPredicates)
			seenPriorities = seenPriorities.Union(gotPrioritizers)
		})
	}

	if !seenPredicates.HasAll(registeredPredicates.List()...) {
		t.Errorf("Registered predicates are missing from compatibility test (add to test stanza for version currently in development): %#v", registeredPredicates.Difference(seenPredicates).List())
	}
	if !seenPriorities.HasAll(registeredPriorities.List()...) {
		t.Errorf("Registered priorities are missing from compatibility test (add to test stanza for version currently in development): %#v", registeredPriorities.Difference(seenPriorities).List())
	}
}

func pluginsToStringSet(plugins []config.Plugin) sets.String {
	s := sets.NewString()
	for _, p := range plugins {
		s.Insert(p.Name)
	}
	return s
}
