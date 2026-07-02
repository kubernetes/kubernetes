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
	"slices"
	"testing"
	"time"

	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	cliflag "k8s.io/component-base/cli/flag"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// startAPIServer starts an in-process apiserver for the test. When serveCheckpointAPI
// is true it enables the alpha node.k8s.io/v1alpha1 group via --runtime-config;
// the PodLevelCheckpointRestore feature gate must be toggled separately by the caller
// (the gate also guards whether the group is served, see storage_checkpoint.go).
func startAPIServer(tCtx ktesting.TContext, t *testing.T, serveCheckpointAPI bool) (clientset.Interface, *restclient.Config, framework.TearDownFunc) {
	return framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable admission plugins that interfere with a minimal pod/node setup.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition", "Priority", "StorageObjectInUseProtection"}
			if serveCheckpointAPI {
				opts.APIEnablement.RuntimeConfig = cliflag.ConfigurationMap{
					nodev1alpha1.SchemeGroupVersion.String(): "true",
				}
			}
		},
	})
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

// TestPodCheckpointControllerReconciles verifies that the lifecycle controller
// removes the restore lock when no Pod is actively restoring from the checkpoint.
// Checkpoint execution and status updates belong to the kubelet.
func TestPodCheckpointControllerReconciles(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)

	tCtx := ktesting.Init(t)
	clientSet, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "podcheckpoint", t)
	checkpoints := clientSet.NodeV1alpha1().PodCheckpoints(ns.Name)

	// Start the controller.
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	controller := podcheckpointcontroller.NewController(clientSet, informerFactory.Core().V1().Pods())
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx, 1)

	checkpoint := newPodCheckpoint(ns.Name, "cp-1", "source-pod")
	checkpoint.Finalizers = []string{podcheckpointcontroller.RestoreLockFinalizer}
	if _, err := checkpoints.Create(tCtx, checkpoint, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create PodCheckpoint: %v", err)
	}

	err := wait.PollUntilContextTimeout(tCtx, time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		obj, err := checkpoints.Get(ctx, "cp-1", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return !slices.Contains(obj.Finalizers, podcheckpointcontroller.RestoreLockFinalizer), nil
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
