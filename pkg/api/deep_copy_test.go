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

package api_test

import (
	"io/ioutil"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func parseTimeOrDie(ts string) metav1.Time {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		panic(err)
	}
	return metav1.Time{Time: t}
}

var benchmarkPod api.Pod = api.Pod{
	TypeMeta: metav1.TypeMeta{
		Kind:       "Pod",
		APIVersion: "v1",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name:              "etcd-server-e2e-test-wojtekt-master",
		Namespace:         "default",
		SelfLink:          "/api/v1/namespaces/default/pods/etcd-server-e2e-test-wojtekt-master",
		UID:               types.UID("a671734a-e8e5-11e4-8fde-42010af09327"),
		ResourceVersion:   "22",
		CreationTimestamp: parseTimeOrDie("2015-04-22T11:49:36Z"),
	},
	Spec: api.PodSpec{
		Volumes: []api.Volume{
			{
				Name: "varetcd",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{
						Path: "/mnt/master-pd/var/etcd",
					},
				},
			},
		},
		Containers: []api.Container{
			{
				Name:  "etcd-container",
				Image: "gcr.io/google_containers/etcd:2.0.9",
				Command: []string{
					"/usr/local/bin/etcd",
					"--addr",
					"127.0.0.1:2379",
					"--bind-addr",
					"127.0.0.1:2379",
					"--data-dir",
					"/var/etcd/data",
				},
				Ports: []api.ContainerPort{
					{
						Name:          "serverport",
						HostPort:      2380,
						ContainerPort: 2380,
						Protocol:      "TCP",
					},
					{
						Name:          "clientport",
						HostPort:      2379,
						ContainerPort: 2379,
						Protocol:      "TCP",
					},
				},
				VolumeMounts: []api.VolumeMount{
					{
						Name:      "varetcd",
						MountPath: "/var/etcd",
					},
				},
				TerminationMessagePath: "/dev/termination-log",
				ImagePullPolicy:        api.PullIfNotPresent,
			},
		},
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		NodeName:      "e2e-test-wojtekt-master",
	},
	Status: api.PodStatus{
		Phase: api.PodRunning,
		Conditions: []api.PodCondition{
			{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
		ContainerStatuses: []api.ContainerStatus{
			{
				Name: "etcd-container",
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{
						StartedAt: parseTimeOrDie("2015-04-22T11:49:32Z"),
					},
				},
				Ready:        true,
				RestartCount: 0,
				Image:        "gcr.io/google_containers/etcd:2.0.9",
				ImageID:      "docker://b6b9a86dc06aa1361357ca1b105feba961f6a4145adca6c54e142c0be0fe87b0",
				ContainerID:  "docker://3cbbf818f1addfc252957b4504f56ef2907a313fe6afc47fc75373674255d46d",
			},
		},
	},
}

func BenchmarkPodCopy(b *testing.B) {
	var result *api.Pod
	for i := 0; i < b.N; i++ {
		obj, err := api.Scheme.DeepCopy(&benchmarkPod)
		if err != nil {
			b.Fatalf("Unexpected error copying pod: %v", err)
		}
		result = obj.(*api.Pod)
	}
	if !apiequality.Semantic.DeepEqual(benchmarkPod, *result) {
		b.Fatalf("Incorrect copy: expected %v, got %v", benchmarkPod, *result)
	}
}

func BenchmarkNodeCopy(b *testing.B) {
	data, err := ioutil.ReadFile("node_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var node api.Node
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &node); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	var result *api.Node
	for i := 0; i < b.N; i++ {
		obj, err := api.Scheme.DeepCopy(&node)
		if err != nil {
			b.Fatalf("Unexpected error copying node: %v", err)
		}
		result = obj.(*api.Node)
	}
	if !apiequality.Semantic.DeepEqual(node, *result) {
		b.Fatalf("Incorrect copy: expected %v, got %v", node, *result)
	}
}

func BenchmarkReplicationControllerCopy(b *testing.B) {
	data, err := ioutil.ReadFile("replication_controller_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var replicationController api.ReplicationController
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &replicationController); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	var result *api.ReplicationController
	for i := 0; i < b.N; i++ {
		obj, err := api.Scheme.DeepCopy(&replicationController)
		if err != nil {
			b.Fatalf("Unexpected error copying replication controller: %v", err)
		}
		result = obj.(*api.ReplicationController)
	}
	if !apiequality.Semantic.DeepEqual(replicationController, *result) {
		b.Fatalf("Incorrect copy: expected %v, got %v", replicationController, *result)
	}
}
