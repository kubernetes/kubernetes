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
	"context"
	"fmt"
	"sort"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
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
	deployment, err := client.AppsV1().Deployments(namespace).Create(context.TODO(), deploymentSpec, metav1.CreateOptions{})
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
	replicaSetSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, err
	}

	replicaSetListOptions := metav1.ListOptions{LabelSelector: replicaSetSelector.String()}
	allReplicaSets, err := client.AppsV1().ReplicaSets(deployment.Namespace).List(context.TODO(), replicaSetListOptions)
	if err != nil {
		return nil, err
	}

	ownedReplicaSets := make([]*appsv1.ReplicaSet, 0, len(allReplicaSets.Items))
	for _, rs := range allReplicaSets.Items {
		if !metav1.IsControlledBy(&rs, deployment) {
			continue
		}

		ownedReplicaSets = append(ownedReplicaSets, &rs)
	}

	// We ignore pod-template-hash because:
	// 1. The hash result would be different upon podTemplateSpec API changes
	//    (e.g. the addition of a new field will cause the hash code to change)
	// 2. The deployment template won't have hash labels
	podTemplatesEqualsIgnoringHash := func(template1, template2 *v1.PodTemplateSpec) bool {
		t1Copy := template1.DeepCopy()
		t2Copy := template2.DeepCopy()
		// Remove hash labels from template.Labels before comparing
		delete(t1Copy.Labels, appsv1.DefaultDeploymentUniqueLabelKey)
		delete(t2Copy.Labels, appsv1.DefaultDeploymentUniqueLabelKey)
		return apiequality.Semantic.DeepEqual(t1Copy, t2Copy)
	}

	var replicaSet *appsv1.ReplicaSet
	// In rare cases, such as after cluster upgrades, Deployment may end up with
	// having more than one new ReplicaSets that have the same template as its template,
	// see https://github.com/kubernetes/kubernetes/issues/40415
	// We deterministically choose the oldest new ReplicaSet.
	sort.Sort(replicaSetsByCreationTimestamp(ownedReplicaSets))
	for _, rs := range ownedReplicaSets {
		if !podTemplatesEqualsIgnoringHash(&rs.Spec.Template, &deployment.Spec.Template) {
			continue
		}

		replicaSet = rs
		break
	}

	if replicaSet == nil {
		return nil, fmt.Errorf("expected a new replica set for deployment %q, found none", deployment.Name)
	}

	podSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, err
	}
	podListOptions := metav1.ListOptions{LabelSelector: podSelector.String()}
	allPods, err := client.CoreV1().Pods(deployment.Namespace).List(context.TODO(), podListOptions)
	if err != nil {
		return nil, err
	}

	replicaSetUID := replicaSet.UID
	ownedPods := &v1.PodList{Items: make([]v1.Pod, 0, len(allPods.Items))}
	for _, pod := range allPods.Items {
		controllerRef := metav1.GetControllerOf(&pod)
		if controllerRef != nil && controllerRef.UID == replicaSetUID {
			ownedPods.Items = append(ownedPods.Items, pod)
		}
	}

	return ownedPods, nil
}

// replicaSetsByCreationTimestamp sorts a list of ReplicaSet by creation timestamp, using their names as a tie breaker.
type replicaSetsByCreationTimestamp []*appsv1.ReplicaSet

func (o replicaSetsByCreationTimestamp) Len() int      { return len(o) }
func (o replicaSetsByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o replicaSetsByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(&o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(&o[j].CreationTimestamp)
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
