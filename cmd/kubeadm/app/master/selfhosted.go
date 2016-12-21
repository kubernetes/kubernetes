/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/wait"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	ext "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

func CreateSelfHostedControlPlane(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {
	volumes := []v1.Volume{k8sVolume(cfg)}
	volumeMounts := []v1.VolumeMount{k8sVolumeMount()}
	if isCertsVolumeMountNeeded() {
		volumes = append(volumes, certsVolume(cfg))
		volumeMounts = append(volumeMounts, certsVolumeMount())
	}

	if isPkiVolumeMountNeeded() {
		volumes = append(volumes, pkiVolume(cfg))
		volumeMounts = append(volumeMounts, pkiVolumeMount())
	}

	apiServer := getAPIServerDS(cfg, volumes, volumeMounts)
	if _, err := client.Extensions().DaemonSets(api.NamespaceSystem).Create(&apiServer); err != nil {
		return fmt.Errorf("failed to create self-hosted %q daemon set [%v]", kubeAPIServer, err)
	}

	ctrlMgr := getControllerManagerDeployment(cfg, volumes, volumeMounts)
	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(&ctrlMgr); err != nil {
		return fmt.Errorf("failed to create self-hosted %q deployment [%v]", kubeControllerManager, err)
	}

	scheduler := getSchedulerDeployment(cfg)
	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(&scheduler); err != nil {
		return fmt.Errorf("failed to create self-hosted %q deployment [%v]", kubeScheduler, err)
	}

	return nil
}

func WaitForSelfHostedControlPlane(client *clientset.Clientset) error {
	start := time.Now()
	// TODO: Break this up into multiple wait's so we don't re-do every step:
	wait.PollInfinite(apiCallRetryInterval, func() (bool, error) {
		// TODO: This might be pointless, checking the pods is probably enough.
		apiDS, err := client.DaemonSets(api.NamespaceSystem).Get(kubeAPIServer,
			metav1.GetOptions{})
		if err != nil {
			fmt.Println("[debug] error getting apiserver DaemonSet:", err)
			return false, nil
		}
		fmt.Printf("[debug] %s DaemonSet current=%d, desired=%d\n",
			kubeAPIServer,
			apiDS.Status.CurrentNumberScheduled,
			apiDS.Status.DesiredNumberScheduled)
		if apiDS.Status.CurrentNumberScheduled != apiDS.Status.DesiredNumberScheduled {
			return false, nil
		}

		// Check that all API Server pods are running:
		// TODO: Do we need a stronger label link than this?
		listOpts := v1.ListOptions{LabelSelector: "k8s-app=kube-apiserver"}
		apiPods, err := client.Pods(api.NamespaceSystem).List(listOpts)
		if err != nil {
			fmt.Println("[debug] error getting apiserver pods:", err)
			return false, nil
		}
		fmt.Printf("[debug] Found %d apiserver pods\n", len(apiPods.Items))
		if int32(len(apiPods.Items)) != apiDS.Status.DesiredNumberScheduled {
			return false, nil
		}
		for _, pod := range apiPods.Items {
			fmt.Printf("[debug] Pod %s status: %s\n", pod.Name, pod.Status.Phase)
			if pod.Status.Phase != "Running" {
				return false, nil
			}
		}

		fmt.Printf("[debug] self-hosted control plane components ready after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})
	// TODO: Timeout eventually
	return nil
}

// Sources from bootkube templates.go
func getAPIServerDS(cfg *kubeadmapi.MasterConfiguration,
	volumes []v1.Volume, volumeMounts []v1.VolumeMount) ext.DaemonSet {

	ds := ext.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "DaemonSet",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      kubeAPIServer,
			Namespace: "kube-system",
			//Labels:    map[string]string{"k8s-app": "kube-apiserver"},
		},
		Spec: ext.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{
						// TODO: taken from bootkube, appears to be essential, without this
						// we don't get an apiserver pod...
						"k8s-app":   kubeAPIServer,
						"component": kubeAPIServer,
						"tier":      "control-plane",
					},
				},
				Spec: v1.PodSpec{
					// TODO: Make sure masters get this label
					NodeSelector: map[string]string{metav1.NodeLabelKubeadmAlphaRole: metav1.NodeLabelRoleMaster},
					HostNetwork:  true,
					Volumes:      volumes,
					Containers: []v1.Container{
						v1.Container{
							Name:          kubeAPIServer,
							Image:         images.GetCoreImage(images.KubeAPIServerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command:       getAPIServerCommand(cfg),
							Env:           getProxyEnvVars(),
							VolumeMounts:  volumeMounts,
							LivenessProbe: componentProbe(8080, "/healthz"),
							Resources:     componentResources("250m"),
						},
					},
				},
			},
		},
	}
	return ds
}

func getControllerManagerDeployment(cfg *kubeadmapi.MasterConfiguration,
	volumes []v1.Volume, volumeMounts []v1.VolumeMount) ext.Deployment {

	cmDep := ext.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "Deployment",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      kubeControllerManager,
			Namespace: "kube-system",
		},
		Spec: ext.DeploymentSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{
						// TODO: taken from bootkube, appears to be essential
						"k8s-app":   kubeControllerManager,
						"component": kubeControllerManager,
						"tier":      "control-plane",
					},
					Annotations: map[string]string{
						v1.TolerationsAnnotationKey: getMasterToleration(),
					},
				},
				Spec: v1.PodSpec{
					// TODO: Make sure masters get this label
					NodeSelector: map[string]string{metav1.NodeLabelKubeadmAlphaRole: metav1.NodeLabelRoleMaster},
					HostNetwork:  true,
					Volumes:      volumes,

					Containers: []v1.Container{
						v1.Container{
							Name:          kubeControllerManager,
							Image:         images.GetCoreImage(images.KubeControllerManagerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command:       getControllerManagerCommand(cfg),
							VolumeMounts:  volumeMounts,
							LivenessProbe: componentProbe(10252, "/healthz"),
							Resources:     componentResources("200m"),
							Env:           getProxyEnvVars(),
						},
					},
				},
			},
		},
	}
	return cmDep
}

func getMasterToleration() string {
	// Tolerate the master taint we add to our master nodes, as this can and should
	// run there.
	// TODO: Duplicated above
	masterToleration, _ := json.Marshal([]v1.Toleration{v1.Toleration{
		Key:      "dedicated",
		Value:    "master",
		Operator: v1.TolerationOpEqual,
		Effect:   v1.TaintEffectNoSchedule,
	}})
	return string(masterToleration)
}

func getSchedulerDeployment(cfg *kubeadmapi.MasterConfiguration) ext.Deployment {

	cmDep := ext.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "Deployment",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      kubeScheduler,
			Namespace: "kube-system",
		},
		Spec: ext.DeploymentSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app":   kubeScheduler,
						"component": kubeScheduler,
						"tier":      "control-plane",
					},
					Annotations: map[string]string{
						v1.TolerationsAnnotationKey: getMasterToleration(),
					},
				},
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{metav1.NodeLabelKubeadmAlphaRole: metav1.NodeLabelRoleMaster},
					HostNetwork:  true,

					Containers: []v1.Container{
						v1.Container{
							Name:          kubeScheduler,
							Image:         images.GetCoreImage(images.KubeSchedulerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command:       getSchedulerCommand(cfg, 10251),
							LivenessProbe: componentProbe(10251, "/healthz"),
							Resources:     componentResources("100m"),
							Env:           getProxyEnvVars(),
						},
					},
				},
			},
		},
	}
	return cmDep
}
