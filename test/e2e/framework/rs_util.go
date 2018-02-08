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
	"fmt"

	. "github.com/onsi/ginkgo"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	testutils "k8s.io/kubernetes/test/utils"
)

type updateRsFunc func(d *extensions.ReplicaSet)

func UpdateReplicaSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateReplicaSetFunc) (*extensions.ReplicaSet, error) {
	return testutils.UpdateReplicaSetWithRetries(c, namespace, name, applyUpdate, Logf, Poll, pollShortTimeout)
}

// CheckNewRSAnnotations check if the new RS's annotation is as expected
func CheckNewRSAnnotations(c clientset.Interface, ns, deploymentName string, expectedAnnotations map[string]string) error {
	deployment, err := c.ExtensionsV1beta1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	if err != nil {
		return err
	}
	for k, v := range expectedAnnotations {
		// Skip checking revision annotations
		if k != deploymentutil.RevisionAnnotation && v != newRS.Annotations[k] {
			return fmt.Errorf("Expected new RS annotations = %+v, got %+v", expectedAnnotations, newRS.Annotations)
		}
	}
	return nil
}

// WaitForReadyReplicaSet waits until the replicaset has all of its replicas ready.
func WaitForReadyReplicaSet(c clientset.Interface, ns, name string) error {
	err := wait.Poll(Poll, pollShortTimeout, func() (bool, error) {
		rs, err := c.ExtensionsV1beta1().ReplicaSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return *(rs.Spec.Replicas) == rs.Status.Replicas && *(rs.Spec.Replicas) == rs.Status.ReadyReplicas, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never became ready", name)
	}
	return err
}

func RunReplicaSet(config testutils.ReplicaSetConfig) error {
	By(fmt.Sprintf("creating replicaset %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = DumpNodeDebugInfo
	config.ContainerDumpFunc = LogFailedContainers
	return testutils.RunReplicaSet(config)
}

func NewReplicaSet(name, namespace string, replicas int32, podLabels map[string]string, imageName, image string) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicaSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: extensions.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: podLabels,
			},
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
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
