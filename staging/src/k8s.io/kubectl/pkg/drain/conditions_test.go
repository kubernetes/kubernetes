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
	"io"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

func TestNodeConditionReporter(t *testing.T) {
	oldTransition := metav1.NewTime(time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC))
	now := metav1.NewTime(time.Date(2026, time.October, 21, 12, 0, 0, 0, time.UTC))
	tests := []struct {
		name               string
		event              NodeConditionEvent
		inProgressStatus   corev1.ConditionStatus
		drainedStatus      corev1.ConditionStatus
		reason             string
		message            string
		inProgressChanged  bool
		drainedChanged     bool
		dryRunStrategy     cmdutil.DryRunStrategy
		expectServerDryRun bool
	}{
		{
			name:              "started",
			event:             NodeConditionDrainStarted,
			inProgressStatus:  corev1.ConditionTrue,
			drainedStatus:     corev1.ConditionFalse,
			reason:            reasonDrainStarted,
			message:           "Kubectl drain is in progress",
			inProgressChanged: true,
		},
		{
			name:             "completed",
			event:            NodeConditionDrainCompleted,
			inProgressStatus: corev1.ConditionFalse,
			drainedStatus:    corev1.ConditionTrue,
			reason:           reasonDrainCompleted,
			message:          "Kubectl drain exited 0",
			drainedChanged:   true,
		},
		{
			name:             "failed",
			event:            NodeConditionDrainFailed,
			inProgressStatus: corev1.ConditionFalse,
			drainedStatus:    corev1.ConditionFalse,
			reason:           reasonDrainFailed,
			message:          "Kubectl drain exited 1",
		},
		{
			name:             "interrupted",
			event:            NodeConditionDrainInterrupted,
			inProgressStatus: corev1.ConditionFalse,
			drainedStatus:    corev1.ConditionFalse,
			reason:           reasonDrainInterrupted,
			message:          "Kubectl drain was interrupted",
		},
		{
			name:               "uncordoned server dry-run",
			event:              NodeConditionUncordoned,
			inProgressStatus:   corev1.ConditionFalse,
			drainedStatus:      corev1.ConditionFalse,
			reason:             reasonUncordoned,
			message:            "Kubectl uncordon exited 0",
			dryRunStrategy:     cmdutil.DryRunServer,
			expectServerDryRun: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node"},
				Status: corev1.NodeStatus{Conditions: []corev1.NodeCondition{
					{Type: corev1.NodeDrainInProgress, Status: corev1.ConditionFalse, LastTransitionTime: oldTransition},
					{Type: corev1.NodeDrained, Status: corev1.ConditionFalse, LastTransitionTime: oldTransition},
					{Type: corev1.NodeReady, Status: corev1.ConditionTrue, LastTransitionTime: oldTransition},
				}},
			}
			client := fake.NewClientset(node)
			reporter := NewNodeConditionReporter(client, test.dryRunStrategy)
			reporter.now = func() metav1.Time { return now }
			reporter.backoff = wait.Backoff{Steps: 1}

			if err := reporter.Report(context.Background(), node.Name, test.event); err != nil {
				t.Fatalf("Report() error = %v", err)
			}

			actions := client.Actions()
			if len(actions) != 2 {
				t.Fatalf("expected get and patch actions, got %#v", actions)
			}
			patchAction, ok := actions[1].(clienttesting.PatchActionImpl)
			if !ok {
				t.Fatalf("expected PatchActionImpl, got %T", actions[1])
			}
			if patchAction.GetSubresource() != "status" {
				t.Errorf("subresource = %q, want status", patchAction.GetSubresource())
			}
			if patchAction.GetPatchType() != types.StrategicMergePatchType {
				t.Errorf("patch type = %q, want %q", patchAction.GetPatchType(), types.StrategicMergePatchType)
			}
			gotServerDryRun := len(patchAction.GetPatchOptions().DryRun) == 1 && patchAction.GetPatchOptions().DryRun[0] == metav1.DryRunAll
			if gotServerDryRun != test.expectServerDryRun {
				t.Errorf("server dry-run = %t, want %t", gotServerDryRun, test.expectServerDryRun)
			}

			var patch map[string]json.RawMessage
			if err := json.Unmarshal(patchAction.GetPatch(), &patch); err != nil {
				t.Fatalf("unmarshal patch: %v", err)
			}
			if len(patch) != 1 || patch["status"] == nil {
				t.Fatalf("patch must contain only status, got %s", patchAction.GetPatch())
			}
			var status struct {
				Conditions []corev1.NodeCondition `json:"conditions"`
			}
			if err := json.Unmarshal(patch["status"], &status); err != nil {
				t.Fatalf("unmarshal status: %v", err)
			}
			if len(status.Conditions) != 2 {
				t.Fatalf("conditions = %#v, want two drain conditions", status.Conditions)
			}
			assertCondition := func(got corev1.NodeCondition, conditionType corev1.NodeConditionType, status corev1.ConditionStatus, changed bool) {
				t.Helper()
				if got.Type != conditionType || got.Status != status || got.Reason != test.reason || got.Message != test.message {
					t.Errorf("condition = %#v", got)
				}
				if !got.LastHeartbeatTime.Equal(&now) {
					t.Errorf("heartbeat = %v, want %v", got.LastHeartbeatTime, now)
				}
				wantTransition := oldTransition
				if changed {
					wantTransition = now
				}
				if !got.LastTransitionTime.Equal(&wantTransition) {
					t.Errorf("transition = %v, want %v", got.LastTransitionTime, wantTransition)
				}
			}
			assertCondition(status.Conditions[0], corev1.NodeDrainInProgress, test.inProgressStatus, test.inProgressChanged)
			assertCondition(status.Conditions[1], corev1.NodeDrained, test.drainedStatus, test.drainedChanged)

			oldNode, err := json.Marshal(node)
			if err != nil {
				t.Fatalf("marshal Node: %v", err)
			}
			mergedNode, err := strategicpatch.StrategicMergePatch(oldNode, patchAction.GetPatch(), &corev1.Node{})
			if err != nil {
				t.Fatalf("apply strategic merge patch: %v", err)
			}
			var patchedNode corev1.Node
			if err := json.Unmarshal(mergedNode, &patchedNode); err != nil {
				t.Fatalf("unmarshal patched Node: %v", err)
			}
			if condition := findCondition(patchedNode.Status.Conditions, corev1.NodeReady); condition == nil || condition.Status != corev1.ConditionTrue {
				t.Errorf("unrelated Ready condition was not preserved: %#v", patchedNode.Status.Conditions)
			}
		})
	}
}

func TestNodeConditionReporterClientDryRun(t *testing.T) {
	client := fake.NewClientset()
	reporter := NewNodeConditionReporter(client, cmdutil.DryRunClient)
	if err := reporter.Report(context.Background(), "node", NodeConditionDrainStarted); err != nil {
		t.Fatalf("Report() error = %v", err)
	}
	if actions := client.Actions(); len(actions) != 0 {
		t.Fatalf("expected no actions, got %#v", actions)
	}
}

func TestNodeConditionReporterSerializesUpdates(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}}
	client := fake.NewClientset(node)
	firstGetStarted := make(chan struct{})
	releaseFirstGet := make(chan struct{})
	defer func() {
		select {
		case <-releaseFirstGet:
		default:
			close(releaseFirstGet)
		}
	}()
	var getCalls atomic.Int32
	client.PrependReactor("get", "nodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if getCalls.Add(1) == 1 {
			close(firstGetStarted)
			<-releaseFirstGet
		}
		return false, nil, nil
	})
	reporter := NewNodeConditionReporter(client, cmdutil.DryRunNone)

	firstResult := make(chan error, 1)
	go func() {
		firstResult <- reporter.Report(context.Background(), node.Name, NodeConditionDrainStarted)
	}()
	select {
	case <-firstGetStarted:
	case <-time.After(time.Second):
		t.Fatal("first report did not reach the API client")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	secondErr := reporter.Report(ctx, node.Name, NodeConditionDrainInterrupted)
	getCallCount := getCalls.Load()

	close(releaseFirstGet)
	if err := <-firstResult; err != nil {
		t.Fatalf("first Report() error = %v", err)
	}
	if !errors.Is(secondErr, context.DeadlineExceeded) {
		t.Fatalf("second Report() error = %v, want context deadline exceeded", secondErr)
	}
	if getCallCount != 1 {
		t.Fatalf("concurrent API GET calls = %d, want 1", getCallCount)
	}
}

func TestNewDrainConditionSetsTransitionTimeForNewCondition(t *testing.T) {
	now := metav1.NewTime(time.Date(2026, time.October, 21, 12, 0, 0, 0, time.UTC))
	condition := newDrainCondition(nil, corev1.NodeDrainInProgress, corev1.ConditionTrue, reasonDrainStarted, "Kubectl drain is in progress", now)
	if !condition.LastTransitionTime.Equal(&now) {
		t.Fatalf("transition = %v, want %v", condition.LastTransitionTime, now)
	}
}

func TestNodeConditionReporterRetriesTransientErrors(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}}
	client := fake.NewClientset(node)
	attempts := 0
	client.PrependReactor("patch", "nodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempts++
		if attempts == 1 {
			return true, nil, apierrors.NewServiceUnavailable("try again")
		}
		return true, node, nil
	})
	reporter := NewNodeConditionReporter(client, cmdutil.DryRunNone)
	reporter.backoff = wait.Backoff{Steps: 2}

	if err := reporter.Report(context.Background(), node.Name, NodeConditionDrainStarted); err != nil {
		t.Fatalf("Report() error = %v", err)
	}
	if attempts != 2 {
		t.Fatalf("patch attempts = %d, want 2", attempts)
	}
}

func TestNodeConditionReporterDoesNotRetryForbidden(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}}
	client := fake.NewClientset(node)
	attempts := 0
	client.PrependReactor("patch", "nodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempts++
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "nodes/status"}, node.Name, errors.New("denied"))
	})
	reporter := NewNodeConditionReporter(client, cmdutil.DryRunNone)
	reporter.backoff = wait.Backoff{Steps: 3}

	if err := reporter.Report(context.Background(), node.Name, NodeConditionDrainStarted); !apierrors.IsForbidden(err) {
		t.Fatalf("Report() error = %v, want forbidden", err)
	}
	if attempts != 1 {
		t.Fatalf("patch attempts = %d, want 1", attempts)
	}
}

func TestRunNodeDrainWithoutConditionReporter(t *testing.T) {
	client := fake.NewClientset()
	drainer := &Helper{
		Ctx:    context.Background(),
		Client: client,
		Out:    io.Discard,
		ErrOut: io.Discard,
	}

	if err := RunNodeDrain(drainer, "node"); err != nil {
		t.Fatalf("RunNodeDrain() error = %v", err)
	}
	for _, action := range client.Actions() {
		if action.GetResource().Resource == "nodes" {
			t.Fatalf("unexpected Node action without a condition reporter: %#v", action)
		}
	}
}

func findCondition(conditions []corev1.NodeCondition, conditionType corev1.NodeConditionType) *corev1.NodeCondition {
	for i := range conditions {
		if conditions[i].Type == conditionType {
			return &conditions[i]
		}
	}
	return nil
}
