/*
Copyright 2019 The Kubernetes Authors.

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

package equivalence

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// makeBasicPod returns a Pod object with many of the fields populated.
func makeBasicPod(name string) *v1.Pod {
	isController := true
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "test-ns",
			Labels:    map[string]string{"app": "web", "env": "prod"},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "ReplicationController",
					Name:       "rc",
					UID:        "123",
					Controller: &isController,
				},
			},
		},
		Spec: v1.PodSpec{
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "failure-domain.beta.kubernetes.io/zone",
										Operator: "Exists",
									},
								},
							},
						},
					},
				},
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"app": "db"}},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"app": "web"}},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name:  "init-pause",
					Image: "gcr.io/google_containers/pause",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"cpu": resource.MustParse("1"),
							"mem": resource.MustParse("100Mi"),
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: "gcr.io/google_containers/pause",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"cpu": resource.MustParse("1"),
							"mem": resource.MustParse("100Mi"),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "nfs",
							MountPath: "/srv/data",
						},
					},
				},
			},
			NodeSelector: map[string]string{"node-type": "awesome"},
			Tolerations: []v1.Toleration{
				{
					Effect:   "NoSchedule",
					Key:      "experimental",
					Operator: "Exists",
				},
			},
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol2",
						},
					},
				},
				{
					Name: "nfs",
					VolumeSource: v1.VolumeSource{
						NFS: &v1.NFSVolumeSource{
							Server: "nfs.corp.example.com",
						},
					},
				},
			},
		},
	}
}

func TestGetEquivalenceHash(t *testing.T) {
	pod1 := makeBasicPod("pod1")
	pod2 := makeBasicPod("pod2")

	pod3 := makeBasicPod("pod3")
	pod3.Spec.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: "someEBSVol111",
				},
			},
		},
	}

	pod4 := makeBasicPod("pod4")
	pod4.Spec.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: "someEBSVol222",
				},
			},
		},
	}

	pod5 := makeBasicPod("pod5")
	pod5.Spec.Volumes = []v1.Volume{}

	pod6 := makeBasicPod("pod6")
	pod6.Spec.Volumes = nil

	pod7 := makeBasicPod("pod7")
	pod7.Spec.NodeSelector = nil

	pod8 := makeBasicPod("pod8")
	pod8.Spec.NodeSelector = make(map[string]string)

	tests := []struct {
		name         string
		podList      []*v1.Pod
		isEquivalent bool
	}{
		{
			name:         "pods with everything the same except name",
			podList:      []*v1.Pod{pod1, pod2},
			isEquivalent: true,
		},
		{
			name:         "pods that only differ in their PVC volume sources",
			podList:      []*v1.Pod{pod3, pod4},
			isEquivalent: false,
		},
		{
			name:         "pods that have no volumes, but one uses nil and one uses an empty slice",
			podList:      []*v1.Pod{pod5, pod6},
			isEquivalent: true,
		},
		{
			name:         "pods that have no NodeSelector, but one uses nil and one uses an empty map",
			podList:      []*v1.Pod{pod7, pod8},
			isEquivalent: true,
		},
	}

	var (
		targetEquivHash types.UID
		testPod         *v1.Pod
	)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for i, pod := range test.podList {
				equivHash := GetEquivHash(pod)
				if i == 0 {
					targetEquivHash = equivHash
					testPod = pod
				} else {
					if targetEquivHash != equivHash {
						if test.isEquivalent {
							t.Errorf("Failed: pod: %v is expected to be equivalent to: %v", testPod, pod)
						}
					}
				}
			}
		})
	}
}
