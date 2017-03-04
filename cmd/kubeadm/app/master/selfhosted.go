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
	"os"
	"path"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	ext "k8s.io/client-go/pkg/apis/extensions/v1beta1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
)

var (
	// maximum unavailable and surge instances per self-hosted component deployment
	maxUnavailable = intstr.FromInt(0)
	maxSurge       = intstr.FromInt(1)
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

	// Need lock for self-hosted
	volumes = append(volumes, flockVolume())
	volumeMounts = append(volumeMounts, flockVolumeMount())

	if err := launchSelfHostedAPIServer(cfg, client, volumes, volumeMounts); err != nil {
		return err
	}

	if err := launchSelfHostedScheduler(cfg, client, volumes, volumeMounts); err != nil {
		return err
	}

	if err := launchSelfHostedControllerManager(cfg, client, volumes, volumeMounts); err != nil {
		return err
	}

	return nil
}

func launchSelfHostedAPIServer(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset, volumes []v1.Volume, volumeMounts []v1.VolumeMount) error {
	start := time.Now()

	apiServer := getAPIServerDS(cfg, volumes, volumeMounts)
	if _, err := client.Extensions().DaemonSets(metav1.NamespaceSystem).Create(&apiServer); err != nil {
		return fmt.Errorf("failed to create self-hosted %q daemon set [%v]", kubeAPIServer, err)
	}

	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		// TODO: This might be pointless, checking the pods is probably enough.
		// It does however get us a count of how many there should be which may be useful
		// with HA.
		apiDS, err := client.DaemonSets(metav1.NamespaceSystem).Get("self-hosted-"+kubeAPIServer,
			metav1.GetOptions{})
		if err != nil {
			fmt.Println("[self-hosted] error getting apiserver DaemonSet:", err)
			return false, nil
		}
		fmt.Printf("[self-hosted] %s DaemonSet current=%d, desired=%d\n",
			kubeAPIServer,
			apiDS.Status.CurrentNumberScheduled,
			apiDS.Status.DesiredNumberScheduled)

		if apiDS.Status.CurrentNumberScheduled != apiDS.Status.DesiredNumberScheduled {
			return false, nil
		}

		return true, nil
	})

	// Wait for self-hosted API server to take ownership
	waitForPodsWithLabel(client, "self-hosted-"+kubeAPIServer, true)

	// Remove temporary API server
	apiServerStaticManifestPath := buildStaticManifestFilepath(kubeAPIServer)
	if err := os.RemoveAll(apiServerStaticManifestPath); err != nil {
		return fmt.Errorf("unable to delete temporary API server manifest [%v]", err)
	}

	WaitForAPI(client)

	fmt.Printf("[self-hosted] self-hosted kube-apiserver ready after %f seconds\n", time.Since(start).Seconds())
	return nil
}

func launchSelfHostedControllerManager(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset, volumes []v1.Volume, volumeMounts []v1.VolumeMount) error {
	start := time.Now()

	ctrlMgr := getControllerManagerDeployment(cfg, volumes, volumeMounts)
	if _, err := client.Extensions().Deployments(metav1.NamespaceSystem).Create(&ctrlMgr); err != nil {
		return fmt.Errorf("failed to create self-hosted %q deployment [%v]", kubeControllerManager, err)
	}

	waitForPodsWithLabel(client, "self-hosted-"+kubeControllerManager, true)

	ctrlMgrStaticManifestPath := buildStaticManifestFilepath(kubeControllerManager)
	if err := os.RemoveAll(ctrlMgrStaticManifestPath); err != nil {
		return fmt.Errorf("unable to delete temporary controller manager manifest [%v]", err)
	}

	fmt.Printf("[self-hosted] self-hosted kube-controller-manager ready after %f seconds\n", time.Since(start).Seconds())
	return nil

}

func launchSelfHostedScheduler(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset, volumes []v1.Volume, volumeMounts []v1.VolumeMount) error {
	start := time.Now()
	scheduler := getSchedulerDeployment(cfg, volumes, volumeMounts)
	if _, err := client.Extensions().Deployments(metav1.NamespaceSystem).Create(&scheduler); err != nil {
		return fmt.Errorf("failed to create self-hosted %q deployment [%v]", kubeScheduler, err)
	}

	waitForPodsWithLabel(client, "self-hosted-"+kubeScheduler, true)

	schedulerStaticManifestPath := buildStaticManifestFilepath(kubeScheduler)
	if err := os.RemoveAll(schedulerStaticManifestPath); err != nil {
		return fmt.Errorf("unable to delete temporary scheduler manifest [%v]", err)
	}

	fmt.Printf("[self-hosted] self-hosted kube-scheduler ready after %f seconds\n", time.Since(start).Seconds())
	return nil
}

// waitForPodsWithLabel will lookup pods with the given label and wait until they are all
// reporting status as running.
func waitForPodsWithLabel(client *clientset.Clientset, appLabel string, mustBeRunning bool) {
	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		// TODO: Do we need a stronger label link than this?
		listOpts := metav1.ListOptions{LabelSelector: fmt.Sprintf("k8s-app=%s", appLabel)}
		apiPods, err := client.Pods(metav1.NamespaceSystem).List(listOpts)
		if err != nil {
			fmt.Printf("[self-hosted] error getting %s pods [%v]\n", appLabel, err)
			return false, nil
		}
		fmt.Printf("[self-hosted] Found %d %s pods\n", len(apiPods.Items), appLabel)

		// TODO: HA
		if int32(len(apiPods.Items)) != 1 {
			return false, nil
		}
		for _, pod := range apiPods.Items {
			fmt.Printf("[self-hosted] Pod %s status: %s\n", pod.Name, pod.Status.Phase)
			if mustBeRunning && pod.Status.Phase != "Running" {
				return false, nil
			}
		}

		return true, nil
	})
}

// Sources from bootkube templates.go
func getAPIServerDS(cfg *kubeadmapi.MasterConfiguration, volumes []v1.Volume, volumeMounts []v1.VolumeMount) ext.DaemonSet {
	ds := ext.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "DaemonSet",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "self-hosted-" + kubeAPIServer,
			Namespace: "kube-system",
			Labels:    map[string]string{"k8s-app": "self-hosted-" + kubeAPIServer},
		},
		Spec: ext.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app":   "self-hosted-" + kubeAPIServer,
						"component": kubeAPIServer,
						"tier":      "control-plane",
					},
				},
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{kubeadmconstants.LabelNodeRoleMaster: ""},
					HostNetwork:  true,
					Volumes:      volumes,
					Containers: []v1.Container{
						{
							Name:          "self-hosted-" + kubeAPIServer,
							Image:         images.GetCoreImage(images.KubeAPIServerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command:       getAPIServerCommand(cfg, true),
							Env:           getSelfHostedAPIServerEnv(),
							VolumeMounts:  volumeMounts,
							LivenessProbe: componentProbe(6443, "/healthz", v1.URISchemeHTTPS),
							Resources:     componentResources("250m"),
						},
					},
					Tolerations: []v1.Toleration{kubeadmconstants.MasterToleration},
				},
			},
		},
	}
	return ds
}

func getControllerManagerDeployment(cfg *kubeadmapi.MasterConfiguration, volumes []v1.Volume, volumeMounts []v1.VolumeMount) ext.Deployment {
	d := ext.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "Deployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "self-hosted-" + kubeControllerManager,
			Namespace: "kube-system",
			Labels:    map[string]string{"k8s-app": "self-hosted-" + kubeControllerManager},
		},
		Spec: ext.DeploymentSpec{
			// TODO bootkube uses 2 replicas
			Strategy: ext.DeploymentStrategy{
				Type: ext.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &ext.RollingUpdateDeployment{
					MaxUnavailable: &maxUnavailable,
					MaxSurge:       &maxSurge,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app":   "self-hosted-" + kubeControllerManager,
						"component": kubeControllerManager,
						"tier":      "control-plane",
					},
				},
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{kubeadmconstants.LabelNodeRoleMaster: ""},
					HostNetwork:  true,
					Volumes:      volumes,
					Containers: []v1.Container{
						{
							Name:          "self-hosted-" + kubeControllerManager,
							Image:         images.GetCoreImage(images.KubeControllerManagerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command:       getControllerManagerCommand(cfg, true),
							VolumeMounts:  volumeMounts,
							LivenessProbe: componentProbe(10252, "/healthz", v1.URISchemeHTTP),
							Resources:     componentResources("200m"),
							Env:           getProxyEnvVars(),
						},
					},
					Tolerations: []v1.Toleration{kubeadmconstants.MasterToleration},
					DNSPolicy:   v1.DNSDefault,
				},
			},
		},
	}
	return d
}

func getSchedulerDeployment(cfg *kubeadmapi.MasterConfiguration, volumes []v1.Volume, volumeMounts []v1.VolumeMount) ext.Deployment {
	d := ext.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "Deployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "self-hosted-" + kubeScheduler,
			Namespace: "kube-system",
			Labels:    map[string]string{"k8s-app": "self-hosted-" + kubeScheduler},
		},
		Spec: ext.DeploymentSpec{
			// TODO bootkube uses 2 replicas
			Strategy: ext.DeploymentStrategy{
				Type: ext.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &ext.RollingUpdateDeployment{
					MaxUnavailable: &maxUnavailable,
					MaxSurge:       &maxSurge,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"k8s-app":   "self-hosted-" + kubeScheduler,
						"component": kubeScheduler,
						"tier":      "control-plane",
					},
				},
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{kubeadmconstants.LabelNodeRoleMaster: ""},
					HostNetwork:  true,
					Volumes:      volumes,
					Containers: []v1.Container{
						{
							Name:          "self-hosted-" + kubeScheduler,
							Image:         images.GetCoreImage(images.KubeSchedulerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							Command:       getSchedulerCommand(cfg, true),
							VolumeMounts:  volumeMounts,
							LivenessProbe: componentProbe(10251, "/healthz", v1.URISchemeHTTP),
							Resources:     componentResources("100m"),
							Env:           getProxyEnvVars(),
						},
					},
					Tolerations: []v1.Toleration{kubeadmconstants.MasterToleration},
				},
			},
		},
	}

	return d
}

func buildStaticManifestFilepath(name string) string {
	return path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "manifests", name+".yaml")
}
