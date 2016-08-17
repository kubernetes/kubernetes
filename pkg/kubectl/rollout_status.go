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

package kubectl

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// StatusViewer provides an interface for resources that provides rollout status.
type StatusViewer interface {
	Status(namespace, name string) (string, bool, error)
}

func StatusViewerFor(kind unversioned.GroupKind, c client.Interface) (StatusViewer, error) {
	switch kind {
	case extensions.Kind("Deployment"):
		return &DeploymentStatusViewer{c.Extensions()}, nil
	}
	return nil, fmt.Errorf("no status viewer has been implemented for %v", kind)
}

type DeploymentStatusViewer struct {
	c client.ExtensionsInterface
}

// Status returns a message describing deployment status, and a bool value indicating if the status is considered done
func (s *DeploymentStatusViewer) Status(namespace, name string) (string, bool, error) {
	deployment, err := s.c.Deployments(namespace).Get(name)
	if err != nil {
		return "", false, err
	}
	if deployment.Generation <= deployment.Status.ObservedGeneration {
		if deployment.Status.UpdatedReplicas == deployment.Spec.Replicas {
			return fmt.Sprintf("deployment %s successfully rolled out\n", name), true, nil
		}
		return fmt.Sprintf("Waiting for rollout to finish: %d out of %d new replicas have been updated...\n", deployment.Status.UpdatedReplicas, deployment.Spec.Replicas), false, nil
	}
	return fmt.Sprintf("Waiting for deployment spec update to be observed...\n"), false, nil
}
