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

package podcheckpoint

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestPodCheckpointGeneration(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	clientSet, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(clientSet, "checkpoint-generation", t)
	checkpoints := clientSet.NodeV1alpha1().PodCheckpoints(ns.Name)
	createCheckpointSource(tCtx, t, clientSet, ns.Name, "source-pod")

	for _, tc := range []struct {
		name       string
		generation int64
	}{
		{name: "omitted"},
		{name: "client-supplied", generation: 42},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkpoint := newPodCheckpoint(ns.Name, tc.name, "source-pod")
			checkpoint.Generation = tc.generation
			created, err := checkpoints.Create(tCtx, checkpoint, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("create PodCheckpoint: %v", err)
			}
			if created.Generation != 1 {
				t.Errorf("created PodCheckpoint generation = %d, want 1", created.Generation)
			}
			stored, err := checkpoints.Get(tCtx, created.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get PodCheckpoint: %v", err)
			}
			if stored.Generation != 1 {
				t.Errorf("stored PodCheckpoint generation = %d, want 1", stored.Generation)
			}
		})
	}
}

// TestPodCheckpointControllerReconciles verifies routing to the source node
// before execution, and restore-lock removal when no Pod still needs the data.
func TestPodCheckpointControllerReconciles(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)

	tCtx := ktesting.Init(t)
	clientSet, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "podcheckpoint", t)
	checkpoints := clientSet.NodeV1alpha1().PodCheckpoints(ns.Name)
	source := createCheckpointSource(tCtx, t, clientSet, ns.Name, "source-pod")

	// Start the controller.
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	controller, err := podcheckpointcontroller.NewController(tCtx, "podcheckpoint-controller", informerFactory.Node().V1alpha1().PodCheckpoints(), informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Nodes(), clientSet)
	if err != nil {
		t.Fatal(err)
	}
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx)

	checkpoint := newPodCheckpoint(ns.Name, "cp-1", "source-pod")
	checkpoint.Finalizers = []string{podcheckpointcontroller.RestoreLockFinalizer}
	if _, err := checkpoints.Create(tCtx, checkpoint, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create PodCheckpoint: %v", err)
	}
	err = wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		stored, err := checkpoints.Get(ctx, checkpoint.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if stored.Status.NodeName == nil || stored.Status.SourcePodUID == nil {
			return false, nil
		}
		if *stored.Status.NodeName != source.Spec.NodeName || *stored.Status.SourcePodUID != source.UID {
			return false, fmt.Errorf("checkpoint was assigned to node %q with source UID %q", *stored.Status.NodeName, *stored.Status.SourcePodUID)
		}
		if len(stored.Status.Conditions) != 0 {
			return false, fmt.Errorf("routing changed execution conditions: %#v", stored.Status.Conditions)
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("controller did not assign the checkpoint: %v", err)
	}
	if err := checkpoints.Delete(tCtx, checkpoint.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	err = wait.PollUntilContextTimeout(tCtx, time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := checkpoints.Get(ctx, "cp-1", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		t.Fatalf("controller did not remove the inactive restore lock: %v", err)
	}
}

// TestPodCheckpointAPINotServedWhenDisabled verifies that with the
// PodLevelCheckpointRestore feature gate disabled, PodCheckpoint is not served
// even when enabled in runtime configuration.
func TestPodCheckpointAPINotServedWhenDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)

	tCtx := ktesting.Init(t)
	clientSet, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "podcheckpoint-disabled", t)
	checkpoints := clientSet.NodeV1alpha1().PodCheckpoints(ns.Name)

	if _, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "cp-1", "src-pod"), metav1.CreateOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected PodCheckpoint API to be unavailable when the feature gate is disabled, got %v", err)
	}
}

func TestPodCheckpointNodeReadIsolation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(tCtx, t, true, func(opts *options.ServerRunOptions) {
		opts.Authorization.Modes = []string{"Node", "RBAC"}
		opts.Admission.GenericAdmission.EnablePlugins = append(opts.Admission.GenericAdmission.EnablePlugins, "NodeRestriction")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-node-isolation", t)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := podcheckpointcontroller.NewController(tCtx, "podcheckpoint-controller", informerFactory.Node().V1alpha1().PodCheckpoints(), informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Nodes(), client)
	if err != nil {
		t.Fatal(err)
	}
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx)

	sources := make([]*v1.Pod, 2)
	nodeClients := make([]clientset.Interface, 2)
	for i, nodeName := range []string{"node-1", "node-2"} {
		source, err := client.CoreV1().Pods(ns.Name).Create(tCtx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("source-%d", i)},
			Spec: v1.PodSpec{NodeName: nodeName, Containers: []v1.Container{{
				Name: "app", Image: "pause", Env: []v1.EnvVar{{Name: "NODE_PRIVATE_VALUE", Value: nodeName}},
			}}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		source = markCheckpointSourceRunning(tCtx, t, client, source)
		sources[i] = source
		name := fmt.Sprintf("checkpoint-%d", i)
		if _, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, name, source.Name), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			checkpoint, err := checkpoints.Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return checkpoint.Status.NodeName != nil && *checkpoint.Status.NodeName == nodeName &&
				checkpoint.Status.SourcePodUID != nil && *checkpoint.Status.SourcePodUID == source.UID, nil
		}); err != nil {
			t.Fatalf("controller did not route %s: %v", name, err)
		}

		nodeConfig := restclient.CopyConfig(config)
		nodeConfig.Impersonate = restclient.ImpersonationConfig{UserName: "system:node:" + nodeName, Groups: []string{user.NodesGroup, user.AllAuthenticated}}
		nodeClient, err := clientset.NewForConfig(nodeConfig)
		if err != nil {
			t.Fatal(err)
		}
		nodeClients[i] = nodeClient
		if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			_, err := nodeClient.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, name, metav1.GetOptions{})
			if apierrors.IsForbidden(err) {
				return false, nil
			}
			return err == nil, err
		}); err != nil {
			t.Fatalf("node could not read its checkpoint after graph synchronization: %v", err)
		}
	}

	nodeCheckpoints := nodeClients[0].NodeV1alpha1().PodCheckpoints(metav1.NamespaceAll)
	for _, selector := range []string{"", fields.OneTermEqualSelector("status.nodeName", "node-2").String()} {
		if _, err := nodeCheckpoints.List(tCtx, metav1.ListOptions{FieldSelector: selector}); !apierrors.IsForbidden(err) {
			t.Errorf("node LIST with selector %q: got %v, want forbidden", selector, err)
		}
		watch, err := nodeCheckpoints.Watch(tCtx, metav1.ListOptions{FieldSelector: selector})
		if watch != nil {
			watch.Stop()
		}
		if !apierrors.IsForbidden(err) {
			t.Errorf("node WATCH with selector %q: got %v, want forbidden", selector, err)
		}
	}
	if _, err := nodeClients[0].NodeV1alpha1().PodCheckpoints(ns.Name).Get(tCtx, "checkpoint-1", metav1.GetOptions{}); !apierrors.IsForbidden(err) {
		t.Errorf("node GET on another node's checkpoint: got %v, want forbidden", err)
	}
	selector := fields.OneTermEqualSelector("status.nodeName", "node-1").String()
	list, err := nodeCheckpoints.List(tCtx, metav1.ListOptions{FieldSelector: selector})
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Items) != 1 || list.Items[0].Name != "checkpoint-0" {
		t.Fatalf("node-scoped LIST returned other checkpoints: %#v", list.Items)
	}
	watch, err := nodeCheckpoints.Watch(tCtx, metav1.ListOptions{FieldSelector: selector, ResourceVersion: list.ResourceVersion})
	if err != nil {
		t.Fatal(err)
	}
	defer watch.Stop()
	for _, name := range []string{"checkpoint-1", "checkpoint-0"} {
		checkpoint, err := checkpoints.Get(tCtx, name, metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}
		checkpoint.Annotations = map[string]string{"watch-check": "updated"}
		if _, err := checkpoints.Update(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	select {
	case event, open := <-watch.ResultChan():
		checkpoint, ok := event.Object.(*nodev1alpha1.PodCheckpoint)
		if !open || !ok || checkpoint.Name != "checkpoint-0" || checkpoint.Annotations["watch-check"] != "updated" {
			t.Fatalf("node-scoped WATCH returned unexpected event: %#v", event)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("node-scoped WATCH did not deliver its checkpoint update")
	}

	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := nodeClients[0].CoreV1().Pods(ns.Name).Get(ctx, sources[0].Name, metav1.GetOptions{})
		if apierrors.IsForbidden(err) {
			return false, nil
		}
		return err == nil, err
	}); err != nil {
		t.Fatalf("node could not read its source Pod before deletion: %v", err)
	}
	if err := client.CoreV1().Pods(ns.Name).Delete(tCtx, sources[0].Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}); err != nil {
		t.Fatal(err)
	}
	// Wait for the Pod's graph edge to disappear before checking that the
	// checkpoint remains readable for recovery after the source Pod is gone.
	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := nodeClients[0].CoreV1().Pods(ns.Name).Get(ctx, sources[0].Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if apierrors.IsForbidden(err) {
			return true, nil
		}
		return false, err
	}); err != nil {
		t.Fatalf("source Pod graph edge did not disappear: %v", err)
	}
	checkpoint, err := nodeClients[0].NodeV1alpha1().PodCheckpoints(ns.Name).Get(tCtx, "checkpoint-0", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("source deletion revoked access to its checkpoint: %v", err)
	}
	checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress, LastTransitionTime: metav1.Now()}}
	if _, err := nodeClients[0].NodeV1alpha1().PodCheckpoints(ns.Name).UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("assigned node could not update checkpoint status after source deletion: %v", err)
	}
	patch, err := json.Marshal(map[string]any{"status": map[string]any{"conditions": []metav1.Condition{{
		Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonFailed, LastTransitionTime: metav1.Now(),
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nodeClients[0].NodeV1alpha1().PodCheckpoints(ns.Name).Patch(tCtx, checkpoint.Name, types.MergePatchType, patch, metav1.PatchOptions{}, "status"); err != nil {
		t.Fatalf("assigned node could not patch checkpoint status after source deletion: %v", err)
	}
}
