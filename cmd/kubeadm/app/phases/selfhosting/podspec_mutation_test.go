/*
Copyright 2017 The Kubernetes Authors.

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

package selfhosting

import (
	"reflect"
	"sort"
	"testing"

	"k8s.io/api/core/v1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestMutatePodSpec(t *testing.T) {
	var tests = []struct {
		component string
		podSpec   *v1.PodSpec
		expected  v1.PodSpec
	}{
		{
			component: kubeadmconstants.KubeAPIServer,
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "kube-apiserver",
						Command: []string{
							"--advertise-address=10.0.0.1",
						},
					},
				},
			},
			expected: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "kube-apiserver",
						Command: []string{
							"--advertise-address=$(HOST_IP)",
						},
						Env: []v1.EnvVar{
							{
								Name: "HOST_IP",
								ValueFrom: &v1.EnvVarSource{
									FieldRef: &v1.ObjectFieldSelector{
										FieldPath: "status.hostIP",
									},
								},
							},
						},
					},
				},

				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
		{
			component: kubeadmconstants.KubeControllerManager,
			podSpec:   &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
		{
			component: kubeadmconstants.KubeScheduler,
			podSpec:   &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
	}

	for _, rt := range tests {
		mutatePodSpec(GetDefaultMutators(), rt.component, rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed mutatePodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestAddNodeSelectorToPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
			},
		},
		{
			podSpec: &v1.PodSpec{
				NodeSelector: map[string]string{
					"foo": "bar",
				},
			},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					"foo":                                "bar",
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
			},
		},
	}

	for _, rt := range tests {
		addNodeSelectorToPodSpec(rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed addNodeSelectorToPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetMasterTolerationOnPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{},
			expected: v1.PodSpec{
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
			},
		},
		{
			podSpec: &v1.PodSpec{
				Tolerations: []v1.Toleration{
					{Key: "foo", Value: "bar"},
				},
			},
			expected: v1.PodSpec{
				Tolerations: []v1.Toleration{
					{Key: "foo", Value: "bar"},
					kubeadmconstants.MasterToleration,
				},
			},
		},
	}

	for _, rt := range tests {
		setMasterTolerationOnPodSpec(rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setMasterTolerationOnPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetRightDNSPolicyOnPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{},
			expected: v1.PodSpec{
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
		{
			podSpec: &v1.PodSpec{
				DNSPolicy: v1.DNSClusterFirst,
			},
			expected: v1.PodSpec{
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
	}

	for _, rt := range tests {
		setRightDNSPolicyOnPodSpec(rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setRightDNSPolicyOnPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetHostIPOnPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "kube-apiserver",
						Command: []string{
							"--advertise-address=10.0.0.1",
						},
						Env: []v1.EnvVar{},
					},
				},
			},
			expected: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "kube-apiserver",
						Command: []string{
							"--advertise-address=$(HOST_IP)",
						},
						Env: []v1.EnvVar{
							{
								Name: "HOST_IP",
								ValueFrom: &v1.EnvVarSource{
									FieldRef: &v1.ObjectFieldSelector{
										FieldPath: "status.hostIP",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, rt := range tests {
		setHostIPOnPodSpec(rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setHostIPOnPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetSelfHostedVolumesForAPIServer(t *testing.T) {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "ca-certs",
								MountPath: "/etc/ssl/certs",
							},
							{
								Name:      "k8s-certs",
								MountPath: "/etc/kubernetes/pki",
							},
						},
						Command: []string{
							"--foo=bar",
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "k8s-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/pki",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
				},
			},
			expected: v1.PodSpec{
				Containers: []v1.Container{
					{
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "ca-certs",
								MountPath: "/etc/ssl/certs",
							},
							{
								Name:      "k8s-certs",
								MountPath: "/etc/kubernetes/pki",
							},
						},
						Command: []string{
							"--foo=bar",
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name:         "k8s-certs",
						VolumeSource: apiServerCertificatesVolumeSource(),
					},
				},
			},
		},
	}

	for _, rt := range tests {
		setSelfHostedVolumesForAPIServer(rt.podSpec)
		sort.Strings(rt.podSpec.Containers[0].Command)
		sort.Strings(rt.expected.Containers[0].Command)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setSelfHostedVolumesForAPIServer:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetSelfHostedVolumesForControllerManager(t *testing.T) {
	hostPathFileOrCreate := v1.HostPathFileOrCreate
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "ca-certs",
								MountPath: "/etc/ssl/certs",
							},
							{
								Name:      "k8s-certs",
								MountPath: "/etc/kubernetes/pki",
							},
							{
								Name:      "kubeconfig",
								MountPath: "/etc/kubernetes/controller-manager.conf",
							},
						},
						Command: []string{
							"--kubeconfig=/etc/kubernetes/controller-manager.conf",
							"--foo=bar",
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "k8s-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/pki",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "kubeconfig",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/controller-manager.conf",
								Type: &hostPathFileOrCreate,
							},
						},
					},
				},
			},
			expected: v1.PodSpec{
				Containers: []v1.Container{
					{
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "ca-certs",
								MountPath: "/etc/ssl/certs",
							},
							{
								Name:      "k8s-certs",
								MountPath: "/etc/kubernetes/pki",
							},
							{
								Name:      "kubeconfig",
								MountPath: "/etc/kubernetes/kubeconfig",
							},
						},
						Command: []string{
							"--kubeconfig=/etc/kubernetes/kubeconfig/controller-manager.conf",
							"--foo=bar",
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name:         "k8s-certs",
						VolumeSource: controllerManagerCertificatesVolumeSource(),
					},
					{
						Name:         "kubeconfig",
						VolumeSource: kubeConfigVolumeSource(kubeadmconstants.ControllerManagerKubeConfigFileName),
					},
				},
			},
		},
	}

	for _, rt := range tests {
		setSelfHostedVolumesForControllerManager(rt.podSpec)
		sort.Strings(rt.podSpec.Containers[0].Command)
		sort.Strings(rt.expected.Containers[0].Command)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setSelfHostedVolumesForControllerManager:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetSelfHostedVolumesForScheduler(t *testing.T) {
	hostPathFileOrCreate := v1.HostPathFileOrCreate
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "kubeconfig",
								MountPath: "/etc/kubernetes/scheduler.conf",
							},
						},
						Command: []string{
							"--kubeconfig=/etc/kubernetes/scheduler.conf",
							"--foo=bar",
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "kubeconfig",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/scheduler.conf",
								Type: &hostPathFileOrCreate,
							},
						},
					},
				},
			},
			expected: v1.PodSpec{
				Containers: []v1.Container{
					{
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "kubeconfig",
								MountPath: "/etc/kubernetes/kubeconfig",
							},
						},
						Command: []string{
							"--kubeconfig=/etc/kubernetes/kubeconfig/scheduler.conf",
							"--foo=bar",
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name:         "kubeconfig",
						VolumeSource: kubeConfigVolumeSource(kubeadmconstants.SchedulerKubeConfigFileName),
					},
				},
			},
		},
	}

	for _, rt := range tests {
		setSelfHostedVolumesForScheduler(rt.podSpec)
		sort.Strings(rt.podSpec.Containers[0].Command)
		sort.Strings(rt.expected.Containers[0].Command)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setSelfHostedVolumesForScheduler:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}
