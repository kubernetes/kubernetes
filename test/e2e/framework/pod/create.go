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

package pod

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	// BusyBoxImage is the image URI of BusyBox.
	BusyBoxImage = imageutils.GetE2EImage(imageutils.BusyBox)
)

// Config is a struct containing all arguments for creating a pod.
// SELinux testing requires to pass HostIPC and HostPID as boolean arguments.
type Config struct {
	NS                     string
	PVCs                   []*v1.PersistentVolumeClaim
	PVCsReadOnly           bool
	InlineVolumeSources    []*v1.VolumeSource
	IsPrivileged           bool
	Command                string
	HostIPC                bool
	HostPID                bool
	SeLinuxLabel           *v1.SELinuxOptions
	FsGroup                *int64
	NodeSelection          NodeSelection
	ImageID                int
	PodFSGroupChangePolicy *v1.PodFSGroupChangePolicy
}

// CreateUnschedulablePod with given claims based on node selector
func CreateUnschedulablePod(client clientset.Interface, namespace string, nodeSelector map[string]string, pvclaims []*v1.PersistentVolumeClaim, isPrivileged bool, command string) (*v1.Pod, error) {
	pod := MakePod(namespace, nodeSelector, pvclaims, isPrivileged, command)
	pod, err := client.CoreV1().Pods(namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("pod Create API error: %v", err)
	}
	// Waiting for pod to become Unschedulable
	err = WaitForPodNameUnschedulableInNamespace(client, pod.Name, namespace)
	if err != nil {
		return pod, fmt.Errorf("pod %q is not Unschedulable: %v", pod.Name, err)
	}
	// get fresh pod info
	pod, err = client.CoreV1().Pods(namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	if err != nil {
		return pod, fmt.Errorf("pod Get API error: %v", err)
	}
	return pod, nil
}

// CreateClientPod defines and creates a pod with a mounted PV. Pod runs infinite loop until killed.
func CreateClientPod(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim) (*v1.Pod, error) {
	return CreatePod(c, ns, nil, []*v1.PersistentVolumeClaim{pvc}, true, "")
}

// CreatePod with given claims based on node selector
func CreatePod(client clientset.Interface, namespace string, nodeSelector map[string]string, pvclaims []*v1.PersistentVolumeClaim, isPrivileged bool, command string) (*v1.Pod, error) {
	pod := MakePod(namespace, nodeSelector, pvclaims, isPrivileged, command)
	pod, err := client.CoreV1().Pods(namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("pod Create API error: %v", err)
	}
	// Waiting for pod to be running
	err = WaitForPodNameRunningInNamespace(client, pod.Name, namespace)
	if err != nil {
		return pod, fmt.Errorf("pod %q is not Running: %v", pod.Name, err)
	}
	// get fresh pod info
	pod, err = client.CoreV1().Pods(namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	if err != nil {
		return pod, fmt.Errorf("pod Get API error: %v", err)
	}
	return pod, nil
}

// CreateSecPod creates security pod with given claims
func CreateSecPod(client clientset.Interface, podConfig *Config, timeout time.Duration) (*v1.Pod, error) {
	return CreateSecPodWithNodeSelection(client, podConfig, timeout)
}

// CreateSecPodWithNodeSelection creates security pod with given claims
func CreateSecPodWithNodeSelection(client clientset.Interface, podConfig *Config, timeout time.Duration) (*v1.Pod, error) {
	pod, err := MakeSecPod(podConfig)
	if err != nil {
		return nil, fmt.Errorf("Unable to create pod: %v", err)
	}

	pod, err = client.CoreV1().Pods(podConfig.NS).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("pod Create API error: %v", err)
	}

	// Waiting for pod to be running
	err = WaitTimeoutForPodRunningInNamespace(client, pod.Name, podConfig.NS, timeout)
	if err != nil {
		return pod, fmt.Errorf("pod %q is not Running: %v", pod.Name, err)
	}
	// get fresh pod info
	pod, err = client.CoreV1().Pods(podConfig.NS).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	if err != nil {
		return pod, fmt.Errorf("pod Get API error: %v", err)
	}
	return pod, nil
}

// MakePod returns a pod definition based on the namespace. The pod references the PVC's
// name.  A slice of BASH commands can be supplied as args to be run by the pod
func MakePod(ns string, nodeSelector map[string]string, pvclaims []*v1.PersistentVolumeClaim, isPrivileged bool, command string) *v1.Pod {
	if len(command) == 0 {
		command = "trap exit TERM; while true; do sleep 1; done"
	}
	podSpec := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-tester-",
			Namespace:    ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "write-pod",
					Image:           GetDefaultTestImage(),
					Command:         GenerateScriptCmd(command),
					SecurityContext: GenerateContainerSecurityContext(isPrivileged),
				},
			},
			RestartPolicy: v1.RestartPolicyOnFailure,
		},
	}
	setVolumes(&podSpec.Spec, pvclaims, nil /*inline volume sources*/, false /*PVCs readonly*/)
	if nodeSelector != nil {
		podSpec.Spec.NodeSelector = nodeSelector
	}
	return podSpec
}

// MakeSecPod returns a pod definition based on the namespace. The pod references the PVC's
// name.  A slice of BASH commands can be supplied as args to be run by the pod.
func MakeSecPod(podConfig *Config) (*v1.Pod, error) {
	if podConfig.NS == "" {
		return nil, fmt.Errorf("Cannot create pod with empty namespace")
	}
	if len(podConfig.Command) == 0 {
		podConfig.Command = "trap exit TERM; while true; do sleep 1; done"
	}

	podName := "pod-" + string(uuid.NewUUID())
	if podConfig.FsGroup == nil && !NodeOSDistroIs("windows") {
		podConfig.FsGroup = func(i int64) *int64 {
			return &i
		}(1000)
	}
	podSpec := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: podConfig.NS,
		},
		Spec: *MakePodSpec(podConfig),
	}
	return podSpec, nil
}

// MakePodSpec returns a PodSpec definition
func MakePodSpec(podConfig *Config) *v1.PodSpec {
	image := imageutils.BusyBox
	if podConfig.ImageID != imageutils.None {
		image = podConfig.ImageID
	}
	podSpec := &v1.PodSpec{
		HostIPC:         podConfig.HostIPC,
		HostPID:         podConfig.HostPID,
		SecurityContext: GeneratePodSecurityContext(podConfig.FsGroup, podConfig.SeLinuxLabel),
		Containers: []v1.Container{
			{
				Name:            "write-pod",
				Image:           GetTestImage(image),
				Command:         GenerateScriptCmd(podConfig.Command),
				SecurityContext: GenerateContainerSecurityContext(podConfig.IsPrivileged),
			},
		},
		RestartPolicy: v1.RestartPolicyOnFailure,
	}

	if podConfig.PodFSGroupChangePolicy != nil {
		podSpec.SecurityContext.FSGroupChangePolicy = podConfig.PodFSGroupChangePolicy
	}

	setVolumes(podSpec, podConfig.PVCs, podConfig.InlineVolumeSources, podConfig.PVCsReadOnly)
	SetNodeSelection(podSpec, podConfig.NodeSelection)
	return podSpec
}

func setVolumes(podSpec *v1.PodSpec, pvcs []*v1.PersistentVolumeClaim, inlineVolumeSources []*v1.VolumeSource, pvcsReadOnly bool) {
	var volumeMounts = make([]v1.VolumeMount, 0)
	var volumeDevices = make([]v1.VolumeDevice, 0)
	var volumes = make([]v1.Volume, len(pvcs)+len(inlineVolumeSources))
	volumeIndex := 0
	for _, pvclaim := range pvcs {
		volumename := fmt.Sprintf("volume%v", volumeIndex+1)
		if pvclaim.Spec.VolumeMode != nil && *pvclaim.Spec.VolumeMode == v1.PersistentVolumeBlock {
			volumeDevices = append(volumeDevices, v1.VolumeDevice{Name: volumename, DevicePath: "/mnt/" + volumename})
		} else {
			volumeMounts = append(volumeMounts, v1.VolumeMount{Name: volumename, MountPath: "/mnt/" + volumename})
		}
		volumes[volumeIndex] = v1.Volume{
			Name: volumename,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvclaim.Name,
					ReadOnly:  pvcsReadOnly,
				},
			},
		}
		volumeIndex++
	}
	for _, src := range inlineVolumeSources {
		volumename := fmt.Sprintf("volume%v", volumeIndex+1)
		// In-line volumes can be only filesystem, not block.
		volumeMounts = append(volumeMounts, v1.VolumeMount{Name: volumename, MountPath: "/mnt/" + volumename})
		volumes[volumeIndex] = v1.Volume{Name: volumename, VolumeSource: *src}
		volumeIndex++
	}
	podSpec.Containers[0].VolumeMounts = volumeMounts
	podSpec.Containers[0].VolumeDevices = volumeDevices
	podSpec.Volumes = volumes
}
