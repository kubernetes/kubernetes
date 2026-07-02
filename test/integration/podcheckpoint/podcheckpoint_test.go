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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	cliflag "k8s.io/component-base/cli/flag"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// startAPIServer starts an in-process apiserver for the test. When serveCheckpointAPI
// is true it enables the alpha node.k8s.io/v1alpha1 group via --runtime-config;
// the PodLevelCheckpointRestore feature gate must be toggled separately by the caller
// (the gate also guards whether the group is served, see storage_checkpoint.go).
func startAPIServer(tCtx ktesting.TContext, t *testing.T, serveCheckpointAPI bool, configure ...func(*options.ServerRunOptions)) (clientset.Interface, *restclient.Config, framework.TearDownFunc) {
	return framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable admission plugins that interfere with a minimal pod/node setup.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition", "Priority", "StorageObjectInUseProtection"}
			if serveCheckpointAPI {
				opts.APIEnablement.RuntimeConfig = cliflag.ConfigurationMap{
					nodev1alpha1.SchemeGroupVersion.String(): "true",
				}
			}
			for _, f := range configure {
				f(opts)
			}
		},
	})
}

func createCheckpointSource(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, namespace, name string) *v1.Pod {
	t.Helper()
	pod, err := client.CoreV1().Pods(namespace).Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.PodSpec{NodeName: "node-1", Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return markCheckpointSourceRunning(tCtx, t, client, pod)
}

func bindCheckpointSource(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, pod *v1.Pod) *v1.Pod {
	t.Helper()
	pods := client.CoreV1().Pods(pod.Namespace)
	if err := pods.Bind(tCtx, &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: pod.Name, UID: pod.UID}, Target: v1.ObjectReference{Kind: "Node", Name: "node-1"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	bound, err := pods.Get(tCtx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return bound
}

func markCheckpointSourceRunning(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, pod *v1.Pod) *v1.Pod {
	t.Helper()
	var err error
	if pod.Spec.NodeName == "" {
		pod = bindCheckpointSource(tCtx, t, client, pod)
	}

	pod.Status.Phase = v1.PodRunning
	pod, err = client.CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return pod
}

func newPodCheckpoint(namespace, name, sourcePodName string) *nodev1alpha1.PodCheckpoint {
	return &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: nodev1alpha1.PodCheckpointSpec{
			SourcePod: &nodev1alpha1.PodReference{Name: sourcePodName},
		},
	}
}

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
// before execution.
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
	controller := podcheckpointcontroller.NewController(clientSet, informerFactory.Core().V1().Pods())
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx, 1)

	checkpoint := newPodCheckpoint(ns.Name, "cp-1", "source-pod")
	if _, err := checkpoints.Create(tCtx, checkpoint, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create PodCheckpoint: %v", err)
	}
	err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
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
	controller := podcheckpointcontroller.NewController(client, informerFactory.Core().V1().Pods())
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx, 1)

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
	if _, err := nodeClients[0].NodeV1alpha1().PodCheckpoints(ns.Name).Get(tCtx, "checkpoint-0", metav1.GetOptions{}); err != nil {
		t.Fatalf("source deletion revoked access to its checkpoint: %v", err)
	}
}
