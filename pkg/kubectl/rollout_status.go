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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

// StatusViewer provides an interface for resources that have rollout status.
type StatusViewer interface {
	Status(namespace, name string, revision int64) (string, bool, error)
}

func StatusViewerFor(kind schema.GroupKind, c internalclientset.Interface) (StatusViewer, error) {
	switch kind {
	case extensions.Kind("Deployment"), apps.Kind("Deployment"):
		return &DeploymentStatusViewer{c.Extensions()}, nil
	case extensions.Kind("DaemonSet"):
		return &DaemonSetStatusViewer{c.Extensions()}, nil
	}
	return nil, fmt.Errorf("no status viewer has been implemented for %v", kind)
}

type DeploymentStatusViewer struct {
	c extensionsclient.DeploymentsGetter
}

type DaemonSetStatusViewer struct {
	c extensionsclient.DaemonSetsGetter
}

// Status returns a message describing deployment status, and a bool value indicating if the status is considered done
func (s *DeploymentStatusViewer) Status(namespace, name string, revision int64) (string, bool, error) {
	deployment, err := s.c.Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", false, err
	}
	if revision > 0 {
		deploymentRev, err := util.Revision(deployment)
		if err != nil {
			return "", false, fmt.Errorf("cannot get the revision of deployment %q: %v", deployment.Name, err)
		}
		if revision != deploymentRev {
			return "", false, fmt.Errorf("desired revision (%d) is different from the running revision (%d)", revision, deploymentRev)
		}
	}
	if deployment.Generation <= deployment.Status.ObservedGeneration {
		cond := util.GetDeploymentConditionInternal(deployment.Status, extensions.DeploymentProgressing)
		if cond != nil && cond.Reason == util.TimedOutReason {
			return "", false, fmt.Errorf("deployment %q exceeded its progress deadline", name)
		}
		if deployment.Status.UpdatedReplicas < deployment.Spec.Replicas {
			return fmt.Sprintf("Waiting for rollout to finish: %d out of %d new replicas have been updated...\n", deployment.Status.UpdatedReplicas, deployment.Spec.Replicas), false, nil
		}
		if deployment.Status.Replicas > deployment.Status.UpdatedReplicas {
			return fmt.Sprintf("Waiting for rollout to finish: %d old replicas are pending termination...\n", deployment.Status.Replicas-deployment.Status.UpdatedReplicas), false, nil
		}
		if deployment.Status.AvailableReplicas < deployment.Status.UpdatedReplicas {
			return fmt.Sprintf("Waiting for rollout to finish: %d of %d updated replicas are available...\n", deployment.Status.AvailableReplicas, deployment.Status.UpdatedReplicas), false, nil
		}
		return fmt.Sprintf("deployment %q successfully rolled out\n", name), true, nil
	}
	return fmt.Sprintf("Waiting for deployment spec update to be observed...\n"), false, nil
}

// Status returns a message describing daemon set status, and a bool value indicating if the status is considered done
func (s *DaemonSetStatusViewer) Status(namespace, name string, revision int64) (string, bool, error) {
	//ignoring revision as DaemonSets does not have history yet

	daemon, err := s.c.DaemonSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", false, err
	}
	if daemon.Spec.UpdateStrategy.Type != extensions.RollingUpdateDaemonSetStrategyType {
		return "", true, fmt.Errorf("Status is available only for RollingUpdate strategy type")
	}
	if daemon.Generation <= daemon.Status.ObservedGeneration {
		if daemon.Status.UpdatedNumberScheduled < daemon.Status.DesiredNumberScheduled {
			return fmt.Sprintf("Waiting for rollout to finish: %d out of %d new pods have been updated...\n", daemon.Status.UpdatedNumberScheduled, daemon.Status.DesiredNumberScheduled), false, nil
		}
		if daemon.Status.NumberAvailable < daemon.Status.DesiredNumberScheduled {
			return fmt.Sprintf("Waiting for rollout to finish: %d of %d updated pods are available...\n", daemon.Status.NumberAvailable, daemon.Status.DesiredNumberScheduled), false, nil
		}
		return fmt.Sprintf("daemon set %q successfully rolled out\n", name), true, nil
	}
	return fmt.Sprintf("Waiting for daemon set spec update to be observed...\n"), false, nil
}
