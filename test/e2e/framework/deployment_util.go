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

package framework

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	watchtools "k8s.io/client-go/tools/watch"
	appsinternal "k8s.io/kubernetes/pkg/apis/apps"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

func UpdateDeploymentWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateDeploymentFunc) (*apps.Deployment, error) {
	return testutils.UpdateDeploymentWithRetries(c, namespace, name, applyUpdate, Logf, Poll, pollShortTimeout)
}

// Waits for the deployment to clean up old rcs.
func WaitForDeploymentOldRSsNum(c clientset.Interface, ns, deploymentName string, desiredRSNum int) error {
	var oldRSs []*apps.ReplicaSet
	var d *apps.Deployment

	pollErr := wait.PollImmediate(Poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		d = deployment

		_, oldRSs, err = deploymentutil.GetOldReplicaSets(deployment, c.AppsV1())
		if err != nil {
			return false, err
		}
		return len(oldRSs) == desiredRSNum, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("%d old replica sets were not cleaned up for deployment %q", len(oldRSs)-desiredRSNum, deploymentName)
		logReplicaSetsOfDeployment(d, oldRSs, nil)
	}
	return pollErr
}

func logReplicaSetsOfDeployment(deployment *apps.Deployment, allOldRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet) {
	testutils.LogReplicaSetsOfDeployment(deployment, allOldRSs, newRS, Logf)
}

func WaitForObservedDeployment(c clientset.Interface, ns, deploymentName string, desiredGeneration int64) error {
	return testutils.WaitForObservedDeployment(c, ns, deploymentName, desiredGeneration)
}

func WaitForDeploymentWithCondition(c clientset.Interface, ns, deploymentName, reason string, condType apps.DeploymentConditionType) error {
	return testutils.WaitForDeploymentWithCondition(c, ns, deploymentName, reason, condType, Logf, Poll, pollLongTimeout)
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly most of the time, but an overwhelmed RS controller
// may result in taking longer to relabel a RS.
func WaitForDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName string, revision, image string) error {
	return testutils.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, revision, image, Logf, Poll, pollLongTimeout)
}

func NewDeployment(deploymentName string, replicas int32, podLabels map[string]string, imageName, image string, strategyType apps.DeploymentStrategyType) *apps.Deployment {
	zero := int64(0)
	return &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   deploymentName,
			Labels: podLabels,
		},
		Spec: apps.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: podLabels},
			Strategy: apps.DeploymentStrategy{
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
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

// Waits for the deployment to complete, and don't check if rolling update strategy is broken.
// Rolling update strategy is used only during a rolling update, and can be violated in other situations,
// such as shortly after a scaling event or the deployment is just created.
func WaitForDeploymentComplete(c clientset.Interface, d *apps.Deployment) error {
	return testutils.WaitForDeploymentComplete(c, d, Logf, Poll, pollLongTimeout)
}

// Waits for the deployment to complete, and check rolling update strategy isn't broken at any times.
// Rolling update strategy should not be broken during a rolling update.
func WaitForDeploymentCompleteAndCheckRolling(c clientset.Interface, d *apps.Deployment) error {
	return testutils.WaitForDeploymentCompleteAndCheckRolling(c, d, Logf, Poll, pollLongTimeout)
}

// WaitForDeploymentUpdatedReplicasGTE waits for given deployment to be observed by the controller and has at least a number of updatedReplicas
func WaitForDeploymentUpdatedReplicasGTE(c clientset.Interface, ns, deploymentName string, minUpdatedReplicas int32, desiredGeneration int64) error {
	return testutils.WaitForDeploymentUpdatedReplicasGTE(c, ns, deploymentName, minUpdatedReplicas, desiredGeneration, Poll, pollLongTimeout)
}

// WaitForDeploymentRollbackCleared waits for given deployment either started rolling back or doesn't need to rollback.
// Note that rollback should be cleared shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRollbackCleared(c clientset.Interface, ns, deploymentName string) error {
	return testutils.WaitForDeploymentRollbackCleared(c, ns, deploymentName, Poll, pollShortTimeout)
}

// WatchRecreateDeployment watches Recreate deployments and ensures no new pods will run at the same time with
// old pods.
func WatchRecreateDeployment(c clientset.Interface, d *apps.Deployment) error {
	if d.Spec.Strategy.Type != apps.RecreateDeploymentStrategyType {
		return fmt.Errorf("deployment %q does not use a Recreate strategy: %s", d.Name, d.Spec.Strategy.Type)
	}

	w, err := c.AppsV1().Deployments(d.Namespace).Watch(metav1.SingleObject(metav1.ObjectMeta{Name: d.Name, ResourceVersion: d.ResourceVersion}))
	if err != nil {
		return err
	}

	status := d.Status

	condition := func(event watch.Event) (bool, error) {
		d := event.Object.(*apps.Deployment)
		status = d.Status

		if d.Status.UpdatedReplicas > 0 && d.Status.Replicas != d.Status.UpdatedReplicas {
			_, allOldRSs, err := deploymentutil.GetOldReplicaSets(d, c.AppsV1())
			newRS, nerr := deploymentutil.GetNewReplicaSet(d, c.AppsV1())
			if err == nil && nerr == nil {
				Logf("%+v", d)
				logReplicaSetsOfDeployment(d, allOldRSs, newRS)
				logPodsOfDeployment(c, d, append(allOldRSs, newRS))
			}
			return false, fmt.Errorf("deployment %q is running new pods alongside old pods: %#v", d.Name, status)
		}

		return *(d.Spec.Replicas) == d.Status.Replicas &&
			*(d.Spec.Replicas) == d.Status.UpdatedReplicas &&
			d.Generation <= d.Status.ObservedGeneration, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, condition)
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("deployment %q never completed: %#v", d.Name, status)
	}
	return err
}

func ScaleDeployment(clientset clientset.Interface, scalesGetter scaleclient.ScalesGetter, ns, name string, size uint, wait bool) error {
	return ScaleResource(clientset, scalesGetter, ns, name, size, wait, appsinternal.Kind("Deployment"), appsinternal.Resource("deployments"))
}

func RunDeployment(config testutils.DeploymentConfig) error {
	By(fmt.Sprintf("creating deployment %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = DumpNodeDebugInfo
	config.ContainerDumpFunc = LogFailedContainers
	return testutils.RunDeployment(config)
}

func logPodsOfDeployment(c clientset.Interface, deployment *apps.Deployment, rsList []*apps.ReplicaSet) {
	testutils.LogPodsOfDeployment(c, deployment, rsList, Logf)
}

func WaitForDeploymentRevision(c clientset.Interface, d *apps.Deployment, targetRevision string) error {
	err := wait.PollImmediate(Poll, pollLongTimeout, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(d.Namespace).Get(d.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		revision := deployment.Annotations[deploymentutil.RevisionAnnotation]
		return revision == targetRevision, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for revision to become %q for deployment %q: %v", targetRevision, d.Name, err)
	}
	return nil
}

// CheckDeploymentRevisionAndImage checks if the input deployment's and its new replica set's revision and image are as expected.
func CheckDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName, revision, image string) error {
	return testutils.CheckDeploymentRevisionAndImage(c, ns, deploymentName, revision, image)
}

func CreateDeployment(client clientset.Interface, replicas int32, podLabels map[string]string, nodeSelector map[string]string, namespace string, pvclaims []*v1.PersistentVolumeClaim, command string) (*apps.Deployment, error) {
	deploymentSpec := MakeDeployment(replicas, podLabels, nodeSelector, namespace, pvclaims, false, command)
	deployment, err := client.AppsV1().Deployments(namespace).Create(deploymentSpec)
	if err != nil {
		return nil, fmt.Errorf("deployment %q Create API error: %v", deploymentSpec.Name, err)
	}
	Logf("Waiting deployment %q to complete", deploymentSpec.Name)
	err = WaitForDeploymentComplete(client, deployment)
	if err != nil {
		return nil, fmt.Errorf("deployment %q failed to complete: %v", deploymentSpec.Name, err)
	}
	return deployment, nil
}

// MakeDeployment creates a deployment definition based on the namespace. The deployment references the PVC's
// name.  A slice of BASH commands can be supplied as args to be run by the pod
func MakeDeployment(replicas int32, podLabels map[string]string, nodeSelector map[string]string, namespace string, pvclaims []*v1.PersistentVolumeClaim, isPrivileged bool, command string) *apps.Deployment {
	if len(command) == 0 {
		command = "trap exit TERM; while true; do sleep 1; done"
	}
	zero := int64(0)
	deploymentName := "deployment-" + string(uuid.NewUUID())
	deploymentSpec := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deploymentName,
			Namespace: namespace,
		},
		Spec: apps.DeploymentSpec{
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

// GetPodsForDeployment gets pods for the given deployment
func GetPodsForDeployment(client clientset.Interface, deployment *apps.Deployment) (*v1.PodList, error) {
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
	rsList := []*apps.ReplicaSet{replicaSet}
	podList, err := deploymentutil.ListPods(deployment, rsList, podListFunc)
	if err != nil {
		return nil, fmt.Errorf("Failed to list Pods of Deployment %q: %v", deployment.Name, err)
	}
	return podList, nil
}
