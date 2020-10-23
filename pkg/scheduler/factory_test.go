/*
Copyright 2014 The Kubernetes Authors.

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

package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	apicore "k8s.io/kubernetes/pkg/apis/core"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkplugins "k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
)

const (
	podInitialBackoffDurationSeconds = 1
	podMaxBackoffDurationSeconds     = 10
	testSchedulerName                = "test-scheduler"
)

func TestCreate(t *testing.T) {
	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, stopCh)
	if _, err := factory.createFromProvider(schedulerapi.SchedulerDefaultProviderName); err != nil {
		t.Error(err)
	}
}

// createAlgorithmSourceFromPolicy creates the schedulerAlgorithmSource from policy string
func createAlgorithmSourceFromPolicy(configData []byte, clientSet clientset.Interface) schedulerapi.SchedulerAlgorithmSource {
	configPolicyName := "scheduler-custom-policy-config"
	policyConfigMap := v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: configPolicyName},
		Data:       map[string]string{schedulerapi.SchedulerPolicyConfigMapKey: string(configData)},
	}

	clientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &policyConfigMap, metav1.CreateOptions{})

	return schedulerapi.SchedulerAlgorithmSource{
		Policy: &schedulerapi.SchedulerPolicySource{
			ConfigMap: &schedulerapi.SchedulerPolicyConfigMapSource{
				Namespace: policyConfigMap.Namespace,
				Name:      policyConfigMap.Name,
			},
		},
	}
}

// TestCreateFromConfig configures a scheduler from policies defined in a configMap.
// It combines some configurable predicate/priorities with some pre-defined ones
func TestCreateFromConfig(t *testing.T) {
	testcases := []struct {
		name             string
		configData       []byte
		wantPluginConfig []schedulerapi.PluginConfig
		wantPlugins      *schedulerapi.Plugins
	}{

		{
			name: "policy with unspecified predicates or priorities uses default",
			configData: []byte(`{
				"kind" : "Policy",
				"apiVersion" : "v1"
			}`),
			wantPluginConfig: []schedulerapi.PluginConfig{
				{
					Name: podtopologyspread.Name,
					Args: &schedulerapi.PodTopologySpreadArgs{DefaultingType: schedulerapi.SystemDefaulting},
				},
			},
			wantPlugins: &schedulerapi.Plugins{
				QueueSort: &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
				PreFilter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "NodeResourcesFit"},
						{Name: "NodePorts"},
						{Name: "VolumeBinding"},
						{Name: "PodTopologySpread"},
						{Name: "InterPodAffinity"},
					},
				},
				Filter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
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
				},
				PostFilter: &schedulerapi.PluginSet{},
				PreScore: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "PodTopologySpread"},
						{Name: "InterPodAffinity"},
						{Name: "TaintToleration"},
					},
				},
				Score: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "NodeResourcesBalancedAllocation", Weight: 1},
						{Name: "PodTopologySpread", Weight: 2},
						{Name: "ImageLocality", Weight: 1},
						{Name: "InterPodAffinity", Weight: 1},
						{Name: "NodeResourcesLeastAllocated", Weight: 1},
						{Name: "NodeAffinity", Weight: 1},
						{Name: "NodePreferAvoidPods", Weight: 10000},
						{Name: "TaintToleration", Weight: 1},
					},
				},
				Reserve:  &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "VolumeBinding"}}},
				Permit:   &schedulerapi.PluginSet{},
				PreBind:  &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "VolumeBinding"}}},
				Bind:     &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				PostBind: &schedulerapi.PluginSet{},
			},
		},
		{
			name: "policy with arguments",
			configData: []byte(`{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [
					{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["zone"]}}},
					{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["foo"]}}},
					{"name" : "TestRequireZone", "argument" : {"labelsPresence" : {"labels" : ["zone"], "presence" : true}}},
					{"name" : "TestNoFooLabel", "argument" : {"labelsPresence" : {"labels" : ["foo"], "presence" : false}}}
				],
				"priorities" : [
					{"name" : "RackSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "rack"}}},
					{"name" : "ZoneSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "zone"}}},
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
						}
					},
					{"name" : "LabelPreference1", "weight" : 3, "argument" : {"labelPreference" : {"label" : "l1", "presence": true}}},
					{"name" : "LabelPreference2", "weight" : 3, "argument" : {"labelPreference" : {"label" : "l2", "presence": false}}},
					{"name" : "NodeAffinityPriority", "weight" : 2},
					{"name" : "InterPodAffinityPriority", "weight" : 1}
				]
			}`),
			wantPluginConfig: []schedulerapi.PluginConfig{
				{
					Name: nodelabel.Name,
					Args: &schedulerapi.NodeLabelArgs{
						PresentLabels:           []string{"zone"},
						AbsentLabels:            []string{"foo"},
						PresentLabelsPreference: []string{"l1"},
						AbsentLabelsPreference:  []string{"l2"},
					},
				},
				{
					Name: serviceaffinity.Name,
					Args: &schedulerapi.ServiceAffinityArgs{
						AffinityLabels:               []string{"zone", "foo"},
						AntiAffinityLabelsPreference: []string{"rack", "zone"},
					},
				},
				{
					Name: noderesources.RequestedToCapacityRatioName,
					Args: &schedulerapi.RequestedToCapacityRatioArgs{
						Shape: []schedulerapi.UtilizationShapePoint{
							{Utilization: 0, Score: 0},
							{Utilization: 50, Score: 7},
						},
						Resources: []schedulerapi.ResourceSpec{},
					},
				},
			},
			wantPlugins: &schedulerapi.Plugins{
				QueueSort: &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
				PreFilter: &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "ServiceAffinity"}}},
				Filter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "NodeUnschedulable"},
						{Name: "TaintToleration"},
						{Name: "NodeLabel"},
						{Name: "ServiceAffinity"},
					},
				},
				PostFilter: &schedulerapi.PluginSet{},
				PreScore:   &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "InterPodAffinity"}}},
				Score: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "InterPodAffinity", Weight: 1},
						{Name: "NodeAffinity", Weight: 2},
						{Name: "NodeLabel", Weight: 6},
						{Name: "RequestedToCapacityRatio", Weight: 2},
						{Name: "ServiceAffinity", Weight: 6},
					},
				},
				Reserve:  &schedulerapi.PluginSet{},
				Permit:   &schedulerapi.PluginSet{},
				PreBind:  &schedulerapi.PluginSet{},
				Bind:     &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				PostBind: &schedulerapi.PluginSet{},
			},
		},
		{
			name: "policy with HardPodAffinitySymmetricWeight argument",
			configData: []byte(`{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [
					{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["zone"]}}},
					{"name" : "TestRequireZone", "argument" : {"labelsPresence" : {"labels" : ["zone"], "presence" : true}}},
					{"name" : "PodFitsResources"},
					{"name" : "PodFitsHostPorts"}
				],
				"priorities" : [
					{"name" : "RackSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "rack"}}},
					{"name" : "NodeAffinityPriority", "weight" : 2},
					{"name" : "ImageLocalityPriority", "weight" : 1},
					{"name" : "InterPodAffinityPriority", "weight" : 1}
				],
				"hardPodAffinitySymmetricWeight" : 10
			}`),
			wantPluginConfig: []schedulerapi.PluginConfig{
				{
					Name: nodelabel.Name,
					Args: &schedulerapi.NodeLabelArgs{
						PresentLabels: []string{"zone"},
					},
				},
				{
					Name: serviceaffinity.Name,
					Args: &schedulerapi.ServiceAffinityArgs{
						AffinityLabels:               []string{"zone"},
						AntiAffinityLabelsPreference: []string{"rack"},
					},
				},
				{
					Name: interpodaffinity.Name,
					Args: &schedulerapi.InterPodAffinityArgs{
						HardPodAffinityWeight: 10,
					},
				},
			},
			wantPlugins: &schedulerapi.Plugins{
				QueueSort: &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
				PreFilter: &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "ServiceAffinity"},
				}},
				Filter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "NodeUnschedulable"},
						{Name: "NodePorts"},
						{Name: "NodeResourcesFit"},
						{Name: "TaintToleration"},
						{Name: "NodeLabel"},
						{Name: "ServiceAffinity"},
					},
				},
				PostFilter: &schedulerapi.PluginSet{},
				PreScore:   &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "InterPodAffinity"}}},
				Score: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: "ImageLocality", Weight: 1},
						{Name: "InterPodAffinity", Weight: 1},
						{Name: "NodeAffinity", Weight: 2},
						{Name: "ServiceAffinity", Weight: 3},
					},
				},
				Reserve:  &schedulerapi.PluginSet{},
				Permit:   &schedulerapi.PluginSet{},
				PreBind:  &schedulerapi.PluginSet{},
				Bind:     &schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				PostBind: &schedulerapi.PluginSet{},
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()

			informerFactory := informers.NewSharedInformerFactory(client, 0)
			recorderFactory := profile.NewRecorderFactory(events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()}))

			_, err := New(
				client,
				informerFactory,
				recorderFactory,
				make(chan struct{}),
				WithAlgorithmSource(createAlgorithmSourceFromPolicy(tc.configData, client)),
				WithBuildFrameworkCapturer(func(p schedulerapi.KubeSchedulerProfile) {
					if p.SchedulerName != v1.DefaultSchedulerName {
						t.Errorf("unexpected scheduler name: want %q, got %q", v1.DefaultSchedulerName, p.SchedulerName)
					}

					if diff := cmp.Diff(tc.wantPluginConfig, p.PluginConfig); diff != "" {
						t.Errorf("unexpected plugins config diff (-want, +got): %s", diff)
					}

					if diff := cmp.Diff(tc.wantPlugins, p.Plugins); diff != "" {
						t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
					}
				}),
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}
		})
	}
}

func TestDefaultErrorFunc(t *testing.T) {
	testPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"}}
	testPodUpdated := testPod.DeepCopy()
	testPodUpdated.Labels = map[string]string{"foo": ""}

	tests := []struct {
		name                       string
		injectErr                  error
		podUpdatedDuringScheduling bool // pod is updated during a scheduling cycle
		podDeletedDuringScheduling bool // pod is deleted during a scheduling cycle
		expect                     *v1.Pod
	}{
		{
			name:                       "pod is updated during a scheduling cycle",
			injectErr:                  nil,
			podUpdatedDuringScheduling: true,
			expect:                     testPodUpdated,
		},
		{
			name:      "pod is not updated during a scheduling cycle",
			injectErr: nil,
			expect:    testPod,
		},
		{
			name:                       "pod is deleted during a scheduling cycle",
			injectErr:                  nil,
			podDeletedDuringScheduling: true,
			expect:                     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stopCh := make(chan struct{})
			defer close(stopCh)

			client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}})
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			podInformer := informerFactory.Core().V1().Pods()
			// Need to add/update/delete testPod to the store.
			podInformer.Informer().GetStore().Add(testPod)

			queue := internalqueue.NewPriorityQueue(nil, internalqueue.WithClock(clock.NewFakeClock(time.Now())))
			schedulerCache := internalcache.New(30*time.Second, stopCh)

			queue.Add(testPod)
			queue.Pop()

			if tt.podUpdatedDuringScheduling {
				podInformer.Informer().GetStore().Update(testPodUpdated)
				queue.Update(testPod, testPodUpdated)
			}
			if tt.podDeletedDuringScheduling {
				podInformer.Informer().GetStore().Delete(testPod)
				queue.Delete(testPod)
			}

			testPodInfo := &framework.QueuedPodInfo{Pod: testPod}
			errFunc := MakeDefaultErrorFunc(client, podInformer.Lister(), queue, schedulerCache)
			errFunc(testPodInfo, tt.injectErr)

			var got *v1.Pod
			if tt.podUpdatedDuringScheduling {
				head, e := queue.Pop()
				if e != nil {
					t.Fatalf("Cannot pop pod from the activeQ: %v", e)
				}
				got = head.Pod
			} else {
				got = getPodFromPriorityQueue(queue, testPod)
			}

			if diff := cmp.Diff(tt.expect, got); diff != "" {
				t.Errorf("Unexpected pod (-want, +got): %s", diff)
			}
		})
	}
}

func TestDefaultErrorFunc_NodeNotFound(t *testing.T) {
	nodeFoo := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	nodeBar := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}
	testPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"}}
	tests := []struct {
		name             string
		nodes            []v1.Node
		nodeNameToDelete string
		injectErr        error
		expectNodeNames  sets.String
	}{
		{
			name:             "node is deleted during a scheduling cycle",
			nodes:            []v1.Node{*nodeFoo, *nodeBar},
			nodeNameToDelete: "foo",
			injectErr:        apierrors.NewNotFound(apicore.Resource("node"), nodeFoo.Name),
			expectNodeNames:  sets.NewString("bar"),
		},
		{
			name:            "node is not deleted but NodeNotFound is received incorrectly",
			nodes:           []v1.Node{*nodeFoo, *nodeBar},
			injectErr:       apierrors.NewNotFound(apicore.Resource("node"), nodeFoo.Name),
			expectNodeNames: sets.NewString("foo", "bar"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stopCh := make(chan struct{})
			defer close(stopCh)

			client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}}, &v1.NodeList{Items: tt.nodes})
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			podInformer := informerFactory.Core().V1().Pods()
			// Need to add testPod to the store.
			podInformer.Informer().GetStore().Add(testPod)

			queue := internalqueue.NewPriorityQueue(nil, internalqueue.WithClock(clock.NewFakeClock(time.Now())))
			schedulerCache := internalcache.New(30*time.Second, stopCh)

			for i := range tt.nodes {
				node := tt.nodes[i]
				// Add node to schedulerCache no matter it's deleted in API server or not.
				schedulerCache.AddNode(&node)
				if node.Name == tt.nodeNameToDelete {
					client.CoreV1().Nodes().Delete(context.TODO(), node.Name, metav1.DeleteOptions{})
				}
			}

			testPodInfo := &framework.QueuedPodInfo{Pod: testPod}
			errFunc := MakeDefaultErrorFunc(client, podInformer.Lister(), queue, schedulerCache)
			errFunc(testPodInfo, tt.injectErr)

			gotNodes := schedulerCache.Dump().Nodes
			gotNodeNames := sets.NewString()
			for _, nodeInfo := range gotNodes {
				gotNodeNames.Insert(nodeInfo.Node().Name)
			}
			if diff := cmp.Diff(tt.expectNodeNames, gotNodeNames); diff != "" {
				t.Errorf("Unexpected nodes (-want, +got): %s", diff)
			}
		})
	}
}

// getPodFromPriorityQueue is the function used in the TestDefaultErrorFunc test to get
// the specific pod from the given priority queue. It returns the found pod in the priority queue.
func getPodFromPriorityQueue(queue *internalqueue.PriorityQueue, pod *v1.Pod) *v1.Pod {
	podList := queue.PendingPods()
	if len(podList) == 0 {
		return nil
	}

	queryPodKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		return nil
	}

	for _, foundPod := range podList {
		foundPodKey, err := cache.MetaNamespaceKeyFunc(foundPod)
		if err != nil {
			return nil
		}

		if foundPodKey == queryPodKey {
			return foundPod
		}
	}

	return nil
}

func newConfigFactoryWithFrameworkRegistry(
	client clientset.Interface, stopCh <-chan struct{},
	registry frameworkruntime.Registry) *Configurator {
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	snapshot := internalcache.NewEmptySnapshot()
	recorderFactory := profile.NewRecorderFactory(events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()}))
	return &Configurator{
		client:                   client,
		informerFactory:          informerFactory,
		percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
		podInitialBackoffSeconds: podInitialBackoffDurationSeconds,
		podMaxBackoffSeconds:     podMaxBackoffDurationSeconds,
		StopEverything:           stopCh,
		registry:                 registry,
		profiles: []schedulerapi.KubeSchedulerProfile{
			{SchedulerName: testSchedulerName},
		},
		recorderFactory:  recorderFactory,
		nodeInfoSnapshot: snapshot,
	}
}

func newConfigFactory(client clientset.Interface, stopCh <-chan struct{}) *Configurator {
	return newConfigFactoryWithFrameworkRegistry(client, stopCh,
		frameworkplugins.NewInTreeRegistry())
}

type fakeExtender struct {
	isBinder          bool
	interestedPodName string
	ignorable         bool
	gotBind           bool
}

func (f *fakeExtender) Name() string {
	return "fakeExtender"
}

func (f *fakeExtender) IsIgnorable() bool {
	return f.ignorable
}

func (f *fakeExtender) ProcessPreemption(
	_ *v1.Pod,
	_ map[string]*extenderv1.Victims,
	_ framework.NodeInfoLister,
) (map[string]*extenderv1.Victims, error) {
	return nil, nil
}

func (f *fakeExtender) SupportsPreemption() bool {
	return false
}

func (f *fakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node) (filteredNodes []*v1.Node, failedNodesMap extenderv1.FailedNodesMap, err error) {
	return nil, nil, nil
}

func (f *fakeExtender) Prioritize(
	_ *v1.Pod,
	_ []*v1.Node,
) (hostPriorities *extenderv1.HostPriorityList, weight int64, err error) {
	return nil, 0, nil
}

func (f *fakeExtender) Bind(binding *v1.Binding) error {
	if f.isBinder {
		f.gotBind = true
		return nil
	}
	return errors.New("not a binder")
}

func (f *fakeExtender) IsBinder() bool {
	return f.isBinder
}

func (f *fakeExtender) IsInterested(pod *v1.Pod) bool {
	return pod != nil && pod.Name == f.interestedPodName
}

type TestPlugin struct {
	name string
}

var _ framework.ScorePlugin = &TestPlugin{}
var _ framework.FilterPlugin = &TestPlugin{}

func (t *TestPlugin) Name() string {
	return t.name
}

func (t *TestPlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	return 1, nil
}

func (t *TestPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (t *TestPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	return nil
}
