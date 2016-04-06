/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
//TODO: to be changed later
package predicates

import (
	"fmt"
	"os/exec"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/codeinspector"
	"k8s.io/kubernetes/federation/apis/federation"

	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"
)

type FakeClusterInfo federation.Cluster

func (n FakeClusterInfo) GetClusterInfo(clusterName string) (*federation.Cluster, error) {
	cluster := federation.Cluster(n)
	return &cluster, nil
}

type FakeClusterListInfo []federation.Cluster

func (clusters FakeClusterListInfo) GetClusterInfo(clusterName string) (*federation.Cluster, error) {
	for _, cluster := range clusters {
		if cluster.Name == clusterName {
			return &cluster, nil
		}
	}
	return nil, fmt.Errorf("Unable to find cluster: %s", clusterName)
}

type FakePersistentVolumeClaimInfo []api.PersistentVolumeClaim

func (pvcs FakePersistentVolumeClaimInfo) GetPersistentVolumeClaimInfo(namespace string, pvcID string) (*api.PersistentVolumeClaim, error) {
	for _, pvc := range pvcs {
		if pvc.Name == pvcID && pvc.Namespace == namespace {
			return &pvc, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume claim: %s/%s", namespace, pvcID)
}

type FakePersistentVolumeInfo []api.PersistentVolume

func (pvs FakePersistentVolumeInfo) GetPersistentVolumeInfo(pvID string) (*api.PersistentVolume, error) {
	for _, pv := range pvs {
		if pv.Name == pvID {
			return &pv, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume: %s", pvID)
}

func makeResources(milliCPU int64, memory int64) federation.ClusterResources {
	return federation.ClusterResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
		},
	}
}

func makeAllocatableResources(milliCPU int64, memory int64) api.ResourceList {
	return api.ResourceList{
		api.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		api.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
	}
}

func newResourceReplicaSet(usage ...resourceRequest) *extensions.ReplicaSet {
	containers := []api.Container{}
	for _, req := range usage {
		containers = append(containers, api.Container{
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceCPU:    *resource.NewMilliQuantity(req.milliCPU, resource.DecimalSI),
					api.ResourceMemory: *resource.NewQuantity(req.memory, resource.BinarySI),
				},
			},
		})
	}
	return &extensions.ReplicaSet{
		Spec: extensions.ReplicaSetSpec {
			Replicas : 1,
			Template: api.PodTemplateSpec {
				Spec: api.PodSpec{
					Containers: containers,
				},
			},
		},
	}
}

func TestReplicaSetFitsResources(t *testing.T) {
	replicaSetsTests := []struct {
		replicaSet  *extensions.ReplicaSet
		clusterInfo *schedulercache.ClusterInfo
		fits        bool
		test        string
		wErr        error
	}{
		{
			replicaSet: &extensions.ReplicaSet{},
			clusterInfo: schedulercache.NewClusterInfo(
				newResourceReplicaSet(resourceRequest{milliCPU: 10, memory: 20})),
			fits: true,
			test: "no resources requested always fits",
			wErr: nil,
		},
		{
			replicaSet: newResourceReplicaSet(resourceRequest{milliCPU: 1, memory: 1}),
			clusterInfo: schedulercache.NewClusterInfo(
				newResourceReplicaSet(resourceRequest{milliCPU: 10, memory: 20})),
			fits: false,
			test: "too many resources fails",
			wErr: newInsufficientResourceError(cpuResourceName, 1, 10, 10),
		},
		{
			replicaSet: newResourceReplicaSet(resourceRequest{milliCPU: 1, memory: 1}),
			clusterInfo: schedulercache.NewClusterInfo(
				newResourceReplicaSet(resourceRequest{milliCPU: 5, memory: 5})),
			fits: true,
			test: "both resources fit",
			wErr: nil,
		},
		{
			replicaSet: newResourceReplicaSet(resourceRequest{milliCPU: 1, memory: 2}),
			clusterInfo: schedulercache.NewClusterInfo(
				newResourceReplicaSet(resourceRequest{milliCPU: 5, memory: 19})),
			fits: false,
			test: "one resources fits",
			wErr: newInsufficientResourceError(memoryResoureceName, 2, 19, 20),
		},
		{
			replicaSet: newResourceReplicaSet(resourceRequest{milliCPU: 5, memory: 1}),
			clusterInfo: schedulercache.NewClusterInfo(
				newResourceReplicaSet(resourceRequest{milliCPU: 5, memory: 19})),
			fits: true,
			test: "equal edge case",
			wErr: nil,
		},
	}

	for _, test := range replicaSetsTests {
		cluster := federation.Cluster{Status: federation.ClusterStatus{Capacity: makeResources(10, 20).Capacity, Allocatable: makeAllocatableResources(10, 20)}}

		fit := ResourceFit{FakeClusterInfo(cluster)}
		fits, err := fit.ReplicaSetFitsResources(test.replicaSet, "cluster", test.clusterInfo)
		if !reflect.DeepEqual(err, test.wErr) {
			t.Errorf("%s: unexpected error: %v, want: %v", test.test, err, test.wErr)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestReplicaSetFitsSelector(t *testing.T) {
	tests := []struct {
		replicaSet    *extensions.ReplicaSet
		labels   []string
		fits   bool
		test   string
	}{
		{
			replicaSet:  &extensions.ReplicaSet{},
			fits: true,
			test: "no selector",
		},
		{
			replicaSet: &extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						federation.ClusterSelectorKey: "foo, bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "same labels",
		},
	}

	for _, test := range tests {
		cluster := federation.Cluster{ObjectMeta: api.ObjectMeta{Labels: test.labels}}

		fit := ClusterSelector{FakeClusterInfo(cluster)}
		fits, err := fit.RSAnnotationMatches(test.replicaSet, "cluster", schedulercache.NewClusterInfo())
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestPredicatesRegistered(t *testing.T) {
	var functionNames []string

	// Files and directories which predicates may be referenced
	targetFiles := []string{
		"./../../algorithmprovider/defaults/defaults.go", // Default algorithm
		"./../../factory/plugins.go",                     // Registered in init()
		"./../../../../../pkg/",                          // kubernetes/pkg, often used by kubelet or controller
	}

	// List all golang source files under ./predicates/, excluding test files and sub-directories.
	files, err := codeinspector.GetSourceCodeFiles(".")

	if err != nil {
		t.Errorf("unexpected error: %v when listing files in current directory", err)
	}

	// Get all public predicates in files.
	for _, filePath := range files {
		functions, err := codeinspector.GetPublicFunctions(filePath)
		if err == nil {
			functionNames = append(functionNames, functions...)
		} else {
			t.Errorf("unexpected error when parsing %s", filePath)
		}
	}

	// Check if all public predicates are referenced in target files.
	for _, functionName := range functionNames {
		args := []string{"-rl", functionName}
		args = append(args, targetFiles...)

		err := exec.Command("grep", args...).Run()
		if err != nil {
			switch err.Error() {
			case "exit status 2":
				t.Errorf("unexpected error when checking %s", functionName)
			case "exit status 1":
				t.Errorf("predicate %s is implemented as public but seems not registered or used in any other place",
					functionName)
			}
		}
	}
}
