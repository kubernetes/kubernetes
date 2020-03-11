/*
Copyright 2020 The Kubernetes Authors.

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

package clusterautoscalerintegrationtest

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	volume_scheduling "k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	scheduler_apis_config "k8s.io/kubernetes/pkg/scheduler/apis/config"
	scheduler_plugins "k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	scheduler_framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestCreateSchedulerFramework(t *testing.T) {
	// Mimics the way Cluster-Autoscaler creates scheduler framework

	kubeClient := clientsetfake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)
	providerRegistry := algorithmprovider.NewRegistry()
	plugins := providerRegistry[scheduler_apis_config.SchedulerDefaultProviderName]
	sharedLister := &DelegatingSchedulerSharedLister{}

	volumeBinder := volume_scheduling.NewVolumeBinder(
		kubeClient,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Storage().V1().CSINodes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Storage().V1().StorageClasses(),
		time.Duration(10)*time.Second,
	)

	framework, err := scheduler_framework.NewFramework(
		scheduler_plugins.NewInTreeRegistry(),
		plugins,
		nil, // This is fine.
		scheduler_framework.WithInformerFactory(informerFactory),
		scheduler_framework.WithSnapshotSharedLister(sharedLister),
		scheduler_framework.WithVolumeBinder(volumeBinder),
	)

	if err != nil {
		t.Errorf("Could not create scheduler framework; %s", err)
	}

	if _, ok := framework.(scheduler_framework.Framework); !ok {
		t.Errorf("Expected NewFramework to return scheduler_framework.Framework interface")
	}
}

func TestSchedulerFrameworkInterface(t *testing.T) {
	// This is strictly compile time check; verifying if methods on Framework
	// called by cluster autoscaler still exist and have unchanged signature.

	var framework scheduler_framework.Framework
	var pod *v1.Pod
	var status *scheduler_framework.Status
	var statuses map[string]*scheduler_framework.Status
	var nodeInfo *nodeinfo.NodeInfo

	if false {
		// framework interface
		state := scheduler_framework.NewCycleState()
		status = framework.RunPreFilterPlugins(context.TODO(), state, pod)
		statuses = framework.RunFilterPlugins(context.TODO(), state, pod, nodeInfo)

		// status interface
		status.Message()
		status.Reasons()
	}

	// just trick GO so it does not complain about unused variables
	if status == nil {}
	if len(statuses) == 0 {}
}

func TestInTreePluginSet(t *testing.T) {
	kubeClient := clientsetfake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)
	providerRegistry := algorithmprovider.NewRegistry()
	plugins := providerRegistry[scheduler_apis_config.SchedulerDefaultProviderName]
	sharedLister := &DelegatingSchedulerSharedLister{}

	framework, err := scheduler_framework.NewFramework(
		scheduler_plugins.NewInTreeRegistry(),
		plugins,
		nil, // This is fine.
		scheduler_framework.WithInformerFactory(informerFactory),
		scheduler_framework.WithSnapshotSharedLister(sharedLister),
	)

	if err != nil {
		t.Errorf("Could not create scheduler framework; %s", err)
	}

	preFilterPluginNames := getPluginNames(framework, "PreFilterPlugin")
	filterPluginNames := getPluginNames(framework, "FilterPlugin")

	expectedPreFilterPluginNames := []string{
		"NodeResourcesFit",
		"NodePorts",
		"InterPodAffinity",
		"PodTopologySpread",
	}

	expectedFilterPluginNames := []string{
		"NodeUnschedulable",
		"NodeResourcesFit",
		"NodeName",
		"NodePorts",
		"NodeAffinity",
		"VolumeRestrictions",
		"TaintToleration",
		"EBSLimits",
		"GCEPDLimits",
		"NodeVolumeLimits",
		"AzureDiskLimits",
		"VolumeBinding",
		"VolumeZone",
		"InterPodAffinity",
		"PodTopologySpread",
	}

	if !stringSliceEqual(expectedPreFilterPluginNames, preFilterPluginNames) {
		t.Errorf("Expected preFilterPlugins to be %#v; got %#v", expectedFilterPluginNames, preFilterPluginNames)
	}

	if !stringSliceEqual(expectedFilterPluginNames, filterPluginNames) {
		t.Errorf("Expected filterPlugins to be %#v; got %#v", expectedFilterPluginNames, filterPluginNames)
	}
}

func getPluginNames(framework scheduler_framework.Framework, extensionPoint string) []string {
	listedPlugins := framework.ListPlugins()
	var pluginNames []string
	for _, pluginConfig := range listedPlugins[extensionPoint] {
		pluginNames = append(pluginNames, pluginConfig.Name)
	}
	return pluginNames
}

func stringSliceEqual(a []string, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

