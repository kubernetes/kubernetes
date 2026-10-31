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
	"errors"
	"fmt"
	"net"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	NodeConditionDrainStarted     NodeConditionEvent = "DrainStarted"
	NodeConditionDrainCompleted   NodeConditionEvent = "DrainCompleted"
	NodeConditionDrainFailed      NodeConditionEvent = "DrainFailed"
	NodeConditionDrainInterrupted NodeConditionEvent = "DrainInterrupted"
	NodeConditionUncordoned       NodeConditionEvent = "Uncordoned"
)

const (
	reasonDrainStarted     = "KubectlDrainStarted"
	reasonDrainCompleted   = "KubectlDrainCompleted"
	reasonDrainFailed      = "KubectlDrainFailed"
	reasonDrainInterrupted = "KubectlDrainInterrupted"
	reasonUncordoned       = "KubectlUncordoned"
)

// ConditionReporter publishes drain observations for a Node.
type ConditionReporter interface {
	Report(ctx context.Context, nodeName string, event NodeConditionEvent) error
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

	inProgressStatus, drainedStatus, reason, message, err := conditionValues(event)
	if err != nil {
		return err
	}
	observationTime := r.now()

	return retry.OnError(r.backoff, isRetriableConditionError, func() error {
		node, err := r.client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}

		patch := struct {
			Status struct {
				Conditions []corev1.NodeCondition `json:"conditions"`
			} `json:"status"`
		}{}
		patch.Status.Conditions = []corev1.NodeCondition{
			newDrainCondition(node.Status.Conditions, corev1.NodeDrainInProgress, inProgressStatus, reason, message, observationTime),
			newDrainCondition(node.Status.Conditions, corev1.NodeDrained, drainedStatus, reason, message, observationTime),
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

func conditionValues(event NodeConditionEvent) (corev1.ConditionStatus, corev1.ConditionStatus, string, string, error) {
	switch event {
	case NodeConditionDrainStarted:
		return corev1.ConditionTrue, corev1.ConditionFalse, reasonDrainStarted, "Kubectl drain is in progress", nil
	case NodeConditionDrainCompleted:
		return corev1.ConditionFalse, corev1.ConditionTrue, reasonDrainCompleted, "Kubectl drain exited 0", nil
	case NodeConditionDrainFailed:
		return corev1.ConditionFalse, corev1.ConditionFalse, reasonDrainFailed, "Kubectl drain exited 1", nil
	case NodeConditionDrainInterrupted:
		return corev1.ConditionFalse, corev1.ConditionFalse, reasonDrainInterrupted, "Kubectl drain was interrupted", nil
	case NodeConditionUncordoned:
		return corev1.ConditionFalse, corev1.ConditionFalse, reasonUncordoned, "Kubectl uncordon exited 0", nil
	default:
		return "", "", "", "", fmt.Errorf("unknown drain condition event %q", event)
	}
}

func newDrainCondition(existing []corev1.NodeCondition, conditionType corev1.NodeConditionType, status corev1.ConditionStatus, reason, message string, observationTime metav1.Time) corev1.NodeCondition {
	transitionTime := observationTime
	for i := range existing {
		if existing[i].Type == conditionType && existing[i].Status == status {
			transitionTime = existing[i].LastTransitionTime
			break
		}
	}
	return corev1.NodeCondition{
		Type:               conditionType,
		Status:             status,
		LastHeartbeatTime:  observationTime,
		LastTransitionTime: transitionTime,
		Reason:             reason,
		Message:            message,
	}
}

func isRetriableConditionError(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if apierrors.IsConflict(err) ||
		apierrors.IsInternalError(err) ||
		apierrors.IsServerTimeout(err) ||
		apierrors.IsServiceUnavailable(err) ||
		apierrors.IsTimeout(err) ||
		apierrors.IsTooManyRequests(err) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}
