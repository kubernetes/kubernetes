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

package deployment

import (
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// UpdateDeploymentWithRetries updates the specified deployment with retries.
func UpdateDeploymentWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateDeploymentFunc) (*appsv1.Deployment, error) {
	return testutils.UpdateDeploymentWithRetries(c, namespace, name, applyUpdate, framework.Logf, poll, pollShortTimeout)
}

// NewDeployment returns a deployment spec with the specified argument.
func NewDeployment(deploymentName string, replicas int32, podLabels map[string]string, imageName, image string, strategyType appsv1.DeploymentStrategyType) *appsv1.Deployment {
	zero := int64(0)
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   deploymentName,
			Labels: podLabels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: podLabels},
			Strategy: appsv1.DeploymentStrategy{
				Type: strategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:            imageName,
							Image:           image,
							SecurityContext: &v1.SecurityContext{},
						},
					},
				},
			},
		},
	}
}

// CreateDeployment creates a deployment.
func CreateDeployment(client clientset.Interface, replicas int32, podLabels map[string]string, nodeSelector map[string]string, namespace string, pvclaims []*v1.PersistentVolumeClaim, command string) (*appsv1.Deployment, error) {
	deploymentSpec := testDeployment(replicas, podLabels, nodeSelector, namespace, pvclaims, false, command)
	deployment, err := client.AppsV1().Deployments(namespace).Create(deploymentSpec)
	if err != nil {
		return nil, fmt.Errorf("deployment %q Create API error: %v", deploymentSpec.Name, err)
	}
	framework.Logf("Waiting deployment %q to complete", deploymentSpec.Name)
	err = WaitForDeploymentComplete(client, deployment)
	if err != nil {
		return nil, fmt.Errorf("deployment %q failed to complete: %v", deploymentSpec.Name, err)
	}
	return deployment, nil
}

// GetPodsForDeployment gets pods for the given deployment
func GetPodsForDeployment(client clientset.Interface, deployment *appsv1.Deployment) (*v1.PodList, error) {
	replicaSet, err := deploymentutil.GetNewReplicaSet(deployment, client.AppsV1())
	if err != nil {
		return nil, fmt.Errorf("Failed to get new replica set for deployment %q: %v", deployment.Name, err)
	}
	if replicaSet == nil {
		return nil, fmt.Errorf("expected a new replica set for deployment %q, found none", deployment.Name)
	}
	podListFunc := func(namespace string, options metav1.ListOptions) (*v1.PodList, error) {
		return client.CoreV1().Pods(namespace).List(options)
	}
	rsList := []*appsv1.ReplicaSet{replicaSet}
	podList, err := deploymentutil.ListPods(deployment, rsList, podListFunc)
	if err != nil {
		return nil, fmt.Errorf("Failed to list Pods of Deployment %q: %v", deployment.Name, err)
	}
	return podList, nil
}

// testDeployment creates a deployment definition based on the namespace. The deployment references the PVC's
// name.  A slice of BASH commands can be supplied as args to be run by the pod
func testDeployment(replicas int32, podLabels map[string]string, nodeSelector map[string]string, namespace string, pvclaims []*v1.PersistentVolumeClaim, isPrivileged bool, command string) *appsv1.Deployment {
	if len(command) == 0 {
		command = "trap exit TERM; while true; do sleep 1; done"
	}
	zero := int64(0)
	deploymentName := "deployment-" + string(uuid.NewUUID())
	deploymentSpec := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deploymentName,
			Namespace: namespace,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: podLabels,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:    "write-pod",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh"},
							Args:    []string{"-c", command},
							SecurityContext: &v1.SecurityContext{
								Privileged: &isPrivileged,
							},
						},
					},
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
	}
	var volumeMounts = make([]v1.VolumeMount, len(pvclaims))
	var volumes = make([]v1.Volume, len(pvclaims))
	for index, pvclaim := range pvclaims {
		volumename := fmt.Sprintf("volume%v", index+1)
		volumeMounts[index] = v1.VolumeMount{Name: volumename, MountPath: "/mnt/" + volumename}
		volumes[index] = v1.Volume{Name: volumename, VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: pvclaim.Name, ReadOnly: false}}}
	}
	deploymentSpec.Spec.Template.Spec.Containers[0].VolumeMounts = volumeMounts
	deploymentSpec.Spec.Template.Spec.Volumes = volumes
	if nodeSelector != nil {
		deploymentSpec.Spec.Template.Spec.NodeSelector = nodeSelector
	}
	return deploymentSpec
}
