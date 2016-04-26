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
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/util/codeinspector"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"

	"k8s.io/kubernetes/federation/apis/federation/unversioned"
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

func TestReplicaSetFitsSelector(t *testing.T) {
	tests := []struct {
		replicaSet    *extensions.ReplicaSet
		fits   bool
		test   string
	}{
		{
			replicaSet: &extensions.ReplicaSet{
				ObjectMeta: v1.ObjectMeta{
					Annotations: map[string]string{
						unversioned.ClusterSelectorKey: "foo, bar",
					},
				},
			},
			fits: true,
			test: "valid cluster",
		},
		{
			replicaSet: &extensions.ReplicaSet{
				ObjectMeta: v1.ObjectMeta{
					Annotations: map[string]string{
						unversioned.ClusterSelectorKey: "baz",
					},
				},
			},
			fits: false,
			test: "no specified cluster",
		},
	}

	for _, test := range tests {
		cluster := federation.Cluster{ObjectMeta: v1.ObjectMeta{Name: "foo"}}

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
		"./../../../../../../pkg/",                          // kubernetes/pkg, often used by kubelet or controller
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

