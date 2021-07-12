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
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

type testCase struct {
	name          string
	JSON          string
	featureGates  map[featuregate.Feature]bool
	wantPlugins   config.Plugins
	wantExtenders []config.Extender
}

func TestPolicyCompatibility(t *testing.T) {
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
			wantPlugins: config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "PrioritySort"}}},
				PreFilter: config.PluginSet{Enabled: []config.Plugin{
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
				}},
				Filter: config.PluginSet{Enabled: []config.Plugin{
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "TaintToleration"},
				}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultPreemption"}}},
				Bind:       config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultBinder"}}},
			},
		},
		// This is a special test for the case where a policy is specified without specifying any filters.
		{
			name: "default config",
			JSON: `{
				"kind": "Policy",
				"apiVersion": "v1",
				"predicates": [
				],
				"priorities": [
				]
			}`,
			wantPlugins: config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "PrioritySort"}}},
				Filter: config.PluginSet{Enabled: []config.Plugin{
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
				}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultPreemption"}}},
				Bind:       config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultBinder"}}},
			},
		},
		{
			name: "all predicates and priorities",
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
			wantPlugins: config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "PrioritySort"}}},
				PreFilter: config.PluginSet{Enabled: []config.Plugin{
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
					{Name: "VolumeBinding"},
					{Name: "InterPodAffinity"},
				}},
				Filter: config.PluginSet{Enabled: []config.Plugin{
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
				}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultPreemption"}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: "InterPodAffinity"},
					{Name: "NodeAffinity"},
					{Name: "PodTopologySpread"},
					{Name: "TaintToleration"},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: "NodeResourcesBalancedAllocation", Weight: 2},
					{Name: "ImageLocality", Weight: 2},
					{Name: "InterPodAffinity", Weight: 2},
					{Name: "NodeResourcesLeastAllocated", Weight: 2},
					{Name: "NodeResourcesMostAllocated", Weight: 2},
					{Name: "NodeAffinity", Weight: 2},
					{Name: "NodePreferAvoidPods", Weight: 2},
					{Name: "RequestedToCapacityRatio", Weight: 2},
					{Name: "PodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 2},
				}},
				Bind:    config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultBinder"}}},
				Reserve: config.PluginSet{Enabled: []config.Plugin{{Name: "VolumeBinding"}}},
				PreBind: config.PluginSet{Enabled: []config.Plugin{{Name: "VolumeBinding"}}},
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
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			recorderFactory := profile.NewRecorderFactory(events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()}))

			sched, err := scheduler.New(
				client,
				informerFactory,
				recorderFactory,
				make(chan struct{}),
				scheduler.WithProfiles([]config.KubeSchedulerProfile(nil)...),
				scheduler.WithLegacyPolicySource(&config.SchedulerPolicySource{
					ConfigMap: &config.SchedulerPolicyConfigMapSource{
						Namespace: policyConfigMap.Namespace,
						Name:      policyConfigMap.Name,
					},
				}),
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			defProf := sched.Profiles["default-scheduler"]
			gotPlugins := defProf.ListPlugins()
			if diff := cmp.Diff(&tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}

			gotExtenders := sched.Extenders
			var wantExtenders []*scheduler.HTTPExtender
			for _, e := range tc.wantExtenders {
				extender, err := scheduler.NewHTTPExtender(&e)
				if err != nil {
					t.Errorf("Error transforming extender: %+v", e)
				}
				wantExtenders = append(wantExtenders, extender.(*scheduler.HTTPExtender))
			}
			for i := range gotExtenders {
				if !scheduler.Equal(wantExtenders[i], gotExtenders[i].(*scheduler.HTTPExtender)) {
					t.Errorf("Got extender #%d %+v, want %+v", i, gotExtenders[i], wantExtenders[i])
				}
			}
		})
	}
}
