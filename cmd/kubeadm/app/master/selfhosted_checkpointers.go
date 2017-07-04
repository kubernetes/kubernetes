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

package master

import (
	"fmt"
	"io/ioutil"
	"time"

	"k8s.io/api/core/v1"
	ext "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
)

func launchCheckpointers(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {
	start := time.Now()

	// pod-checkpointer needs access to /etc/kubernetes/kubeconfig and since
	// /etc/kubernetes is already mounted, we can't spoof the location
	data, err := ioutil.ReadFile("/etc/kubernetes/admin.conf")
	if err != nil {
		return err
	}
	err = ioutil.WriteFile("/etc/kubernetes/kubeconfig", data, 0644)
	if err != nil {
		return err
	}

	podCheckpointerDS := getCheckpointerDS(cfg)
	if _, err := client.Extensions().DaemonSets(metav1.NamespaceSystem).Create(&podCheckpointerDS); err != nil {
		return fmt.Errorf("failed to create self-hosted %s deployment [%v]", podCheckpointer, err)
	}

	kencDS := getKencDS(cfg)
	if _, err := client.Extensions().DaemonSets(metav1.NamespaceSystem).Create(&kencDS); err != nil {
		return fmt.Errorf("failed to create self-hosted %s deployment [%v]", networkCheckpointer, err)
	}

	fmt.Printf("[self-hosted] self-hosted checkpointers ready after %f seconds\n", time.Since(start).Seconds())
	return nil
}

func getCheckpointerDS(cfg *kubeadmapi.MasterConfiguration) ext.DaemonSet {
	return ext.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "DaemonSet",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podCheckpointer,
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"k8s-app": podCheckpointer},
		},
		Spec: ext.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"k8s-app": podCheckpointer},
					Annotations: map[string]string{
						"checkpointer.alpha.coreos.com/checkpoint": "true",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  podCheckpointer,
							Image: images.GetCoreImage(images.PodCheckpointerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command: []string{
								"/checkpoint",
								"--v=4",
								"--lock-file=/var/run/lock/pod-checkpointer.lock",
							},
							Env: []v1.EnvVar{
								getFieldEnv("NODE_NAME", "spec.nodeName"),
								getFieldEnv("POD_NAME", "metadata.name"),
								getFieldEnv("POD_NAMESPACE", "metadata.namespace"),
							},
							ImagePullPolicy: "Always",
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/etc/kubernetes", Name: "etc-kubernetes"},
								{MountPath: "/var/run", Name: "var-run"},
							},
						},
					},
					HostNetwork:   true,
					NodeSelector:  map[string]string{"node-role.kubernetes.io/master": ""},
					RestartPolicy: "Always",
					Tolerations:   []v1.Toleration{kubeadmconstants.MasterToleration},
					Volumes: []v1.Volume{
						{
							Name: "etc-kubernetes",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: "/etc/kubernetes"},
							},
						},
						{
							Name: "var-run",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: "/var/run"},
							},
						},
					},
				},
			},
		},
	}
}

func getKencDS(cfg *kubeadmapi.MasterConfiguration) ext.DaemonSet {
	priv := true
	return ext.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "DaemonSet",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      networkCheckpointer,
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"k8s-app": networkCheckpointer},
		},
		Spec: ext.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app": networkCheckpointer,
					},
					Annotations: map[string]string{
						"checkpointer.alpha.coreos.com/checkpoint": "true",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            networkCheckpointer,
							Image:           images.GetCoreImage(images.NetworkCheckpointerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							SecurityContext: &v1.SecurityContext{Privileged: &priv},
							Command: []string{
								"/usr/bin/flock",
								"/var/lock/kenc.lock",
								"-c",
								"kenc -r -m iptables && kenc -m iptables",
							},
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/etc/kubernetes/selfhosted-etcd", Name: "checkpoint-dir", ReadOnly: false},
								{MountPath: "/var/etcd", Name: "etcd-dir", ReadOnly: false},
								{MountPath: "/var/lock", Name: "var-lock", ReadOnly: false},
							},
						},
					},
					HostNetwork:  true,
					NodeSelector: map[string]string{"node-role.kubernetes.io/master": ""},
					Tolerations:  []v1.Toleration{kubeadmconstants.MasterToleration},
					Volumes: []v1.Volume{
						{
							Name: "checkpoint-dir",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: "/etc/kubernetes/checkpoint-iptables"},
							},
						},
						{
							Name: "etcd-dir",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: "/var/etcd"},
							},
						},
						{
							Name: "var-lock",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: "/var/lock"},
							},
						},
					},
				},
			},
		},
	}
}
