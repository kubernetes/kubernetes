/*
Copyright The Kubernetes Authors.

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

package drain

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

// NodeConditionEvent identifies an observation made while draining or
// uncordoning a Node.
type NodeConditionEvent string

const (
	NodeConditionDrainStarted   NodeConditionEvent = "DrainStarted"
	NodeConditionDrainCompleted NodeConditionEvent = "DrainCompleted"
	NodeConditionDrainFailed    NodeConditionEvent = "DrainFailed"
	NodeConditionUncordoned     NodeConditionEvent = "Uncordoned"
)

const (
	reasonDrainStarted   = "KubectlDrainStarted"
	reasonDrainCompleted = "KubectlDrainCompleted"
	reasonDrainFailed    = "KubectlDrainFailed"
	reasonUncordoned     = "KubectlUncordoned"
)

// ConditionReporter publishes drain observations for a Node.
type ConditionReporter interface {
	Report(ctx context.Context, nodeName string, event NodeConditionEvent) error
}

type noopConditionReporter struct{}

// NewNoopConditionReporter creates a reporter that does not publish conditions.
func NewNoopConditionReporter() ConditionReporter {
	return noopConditionReporter{}
}

func (noopConditionReporter) Report(context.Context, string, NodeConditionEvent) error {
	return nil
}

// NodeConditionReporter publishes drain observations through the Node status
// subresource.
type NodeConditionReporter struct {
	lock           chan struct{}
	client         kubernetes.Interface
	dryRunStrategy cmdutil.DryRunStrategy
	now            func() metav1.Time
	backoff        wait.Backoff
}

// NewNodeConditionReporter creates a reporter backed by the Kubernetes API.
func NewNodeConditionReporter(client kubernetes.Interface, dryRunStrategy cmdutil.DryRunStrategy) *NodeConditionReporter {
	return &NodeConditionReporter{
		lock:           make(chan struct{}, 1),
		client:         client,
		dryRunStrategy: dryRunStrategy,
		now:            metav1.Now,
		backoff: wait.Backoff{
			Steps:    3,
			Duration: 100 * time.Millisecond,
			Factor:   2,
			Jitter:   0.1,
		},
	}
}

// Report publishes both drain conditions in one strategic merge patch.
func (r *NodeConditionReporter) Report(ctx context.Context, nodeName string, event NodeConditionEvent) error {
	select {
	case r.lock <- struct{}{}:
		defer func() { <-r.lock }()
	case <-ctx.Done():
		return ctx.Err()
	}

	if r.dryRunStrategy == cmdutil.DryRunClient {
		return nil
	}

	observationTime := r.now()

	return retry.RetryOnConflict(r.backoff, func() error {
		node, err := r.client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}

		patch := struct {
			Status struct {
				Conditions []corev1.NodeCondition `json:"conditions"`
			} `json:"status"`
		}{}
		patch.Status.Conditions, err = newDrainConditions(node.Status.Conditions, event, observationTime)
		if err != nil {
			return err
		}

		patchBytes, err := json.Marshal(patch)
		if err != nil {
			return err
		}
		patchOptions := metav1.PatchOptions{}
		if r.dryRunStrategy == cmdutil.DryRunServer {
			patchOptions.DryRun = []string{metav1.DryRunAll}
		}
		_, err = r.client.CoreV1().Nodes().Patch(ctx, nodeName, types.StrategicMergePatchType, patchBytes, patchOptions, "status")
		return err
	})
}

func conditionsForEvent(event NodeConditionEvent) ([]corev1.NodeCondition, error) {
	switch event {
	case NodeConditionDrainStarted:
		return []corev1.NodeCondition{
			{Type: corev1.NodeDrainInProgress, Status: corev1.ConditionTrue, Reason: reasonDrainStarted, Message: "Kubectl drain is in progress"},
			{Type: corev1.NodeDrained, Status: corev1.ConditionFalse, Reason: reasonDrainStarted, Message: "Kubectl drain is in progress"},
		}, nil
	case NodeConditionDrainCompleted:
		return []corev1.NodeCondition{
			{Type: corev1.NodeDrainInProgress, Status: corev1.ConditionFalse, Reason: reasonDrainCompleted, Message: "Kubectl drain exited 0"},
			{Type: corev1.NodeDrained, Status: corev1.ConditionTrue, Reason: reasonDrainCompleted, Message: "Kubectl drain exited 0"},
		}, nil
	case NodeConditionDrainFailed:
		return []corev1.NodeCondition{
			{Type: corev1.NodeDrainInProgress, Status: corev1.ConditionFalse, Reason: reasonDrainFailed, Message: "Kubectl drain exited 1"},
			{Type: corev1.NodeDrained, Status: corev1.ConditionFalse, Reason: reasonDrainFailed, Message: "Kubectl drain exited 1"},
		}, nil
	case NodeConditionUncordoned:
		return []corev1.NodeCondition{
			{Type: corev1.NodeDrainInProgress, Status: corev1.ConditionFalse, Reason: reasonUncordoned, Message: "Kubectl uncordon exited 0"},
			{Type: corev1.NodeDrained, Status: corev1.ConditionFalse, Reason: reasonUncordoned, Message: "Kubectl uncordon exited 0"},
		}, nil
	default:
		return nil, fmt.Errorf("unknown drain condition event %q", event)
	}
}

func newDrainConditions(existing []corev1.NodeCondition, event NodeConditionEvent, observationTime metav1.Time) ([]corev1.NodeCondition, error) {
	conditions, err := conditionsForEvent(event)
	if err != nil {
		return nil, err
	}
	for i := range conditions {
		conditions[i] = newDrainCondition(existing, conditions[i], observationTime)
	}
	return conditions, nil
}

func newDrainCondition(existing []corev1.NodeCondition, condition corev1.NodeCondition, observationTime metav1.Time) corev1.NodeCondition {
	transitionTime := observationTime
	for i := range existing {
		if existing[i].Type == condition.Type && existing[i].Status == condition.Status {
			transitionTime = existing[i].LastTransitionTime
			break
		}
	}
	condition.LastHeartbeatTime = observationTime
	condition.LastTransitionTime = transitionTime
	return condition
}
