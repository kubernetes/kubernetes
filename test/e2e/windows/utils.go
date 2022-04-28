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

package windows

import (
	"time"

	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
)

// waits for a deployment to be created and the desired replicas
// are updated and available, and no old pods are running.
func waitForDeployment(getDeploymentFunc func() (*appsv1.Deployment, error), interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		deployment, err := getDeploymentFunc()
		if err != nil {
			if apierrors.IsNotFound(err) {
				framework.Logf("deployment not found, continue waiting: %s", err)
				return false, nil
			}

			framework.Logf("error while deploying, error %s", err)
			return false, err
		}
		framework.Logf("deployment status %s", &deployment.Status)
		return util.DeploymentComplete(deployment, &deployment.Status), nil
	})
}
