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

package replicaset

import (
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
)

// UpdateReplicaSetWithRetries updates replicaset template with retries.
func UpdateReplicaSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateReplicaSetFunc) (*appsv1.ReplicaSet, error) {
	return testutils.UpdateReplicaSetWithRetries(c, namespace, name, applyUpdate, e2elog.Logf, framework.Poll, framework.PollShortTimeout)
}

// CheckNewRSAnnotations check if the new RS's annotation is as expected
func CheckNewRSAnnotations(c clientset.Interface, ns, deploymentName string, expectedAnnotations map[string]string) error {
	deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
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
