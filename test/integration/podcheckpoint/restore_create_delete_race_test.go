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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/plugin/pkg/admission/podrestoreauthorization"
	"k8s.io/kubernetes/test/integration/framework"
)

type restoreCreateBarrier struct {
	admission.Interface
	validated, resume chan struct{}
	once              sync.Once
}

func (b *restoreCreateBarrier) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	return b.Interface.(admission.MutationInterface).Admit(ctx, a, o)
}

func (b *restoreCreateBarrier) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if err := b.Interface.(admission.ValidationInterface).Validate(ctx, a, o); err != nil {
		return err
	}
	if a.GetOperation() != admission.Create || a.GetResource().Resource != "pods" || a.GetName() != "admission-race" {
		return nil
	}
	b.once.Do(func() { close(b.validated) })
	select {
	case <-b.resume:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestRestoreCreateCanFollowCheckpointDeletion(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	validated, resume := make(chan struct{}), make(chan struct{})
	release := sync.OnceFunc(func() { close(resume) })
	client, _, closeFn := startAPIServer(ctx, t, true, func(opts *options.ServerRunOptions) {
		opts.Admission.GenericAdmission.Decorators = append(opts.Admission.GenericAdmission.Decorators, admission.DecoratorFunc(func(handler admission.Interface, name string) admission.Interface {
			if name != podrestoreauthorization.PluginName {
				return handler
			}
			return &restoreCreateBarrier{Interface: handler, validated: validated, resume: resume}
		}))
	})
	defer closeFn()
	defer release()
	ns := framework.CreateNamespaceOrDie(client, "restore-admission-deletion", t)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := podcheckpointcontroller.NewController(client, informerFactory.Core().V1().Pods())
	require.NoError(t, err)
	controllerCtx, stopController := context.WithCancel(ctx)
	defer stopController()
	informerFactory.Start(controllerCtx.Done())
	go controller.Run(controllerCtx, 1)
	require.NoError(t, wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		current, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, checkpoint.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return len(current.Finalizers) != 0, nil
	}))
	spec := checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
	type result struct {
		pod *v1.Pod
		err error
	}
	created := make(chan result, 1)
	go func() {
		pod, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "admission-race"}, Spec: *spec}, metav1.CreateOptions{})
		created <- result{pod: pod, err: err}
	}()
	select {
	case <-validated:
	case <-time.After(10 * time.Second):
		t.Fatal("restore create did not reach the post-validation barrier")
	}
	_, err = client.CoreV1().Pods(ns.Name).Get(ctx, "admission-race", metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err), "validated Pod must not be visible before storage")
	require.NoError(t, client.NodeV1alpha1().PodCheckpoints(ns.Name).Delete(ctx, checkpoint.Name, metav1.DeleteOptions{}))
	require.NoError(t, wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, checkpoint.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	}))
	release()
	select {
	case result := <-created:
		require.NoError(t, result.err)
		require.Equal(t, checkpoint.Name, result.pod.Spec.RestoreFrom.Name)
		require.Contains(t, result.pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	case <-time.After(10 * time.Second):
		t.Fatal("validated restore create did not finish after checkpoint deletion")
	}
	_, err = client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, checkpoint.Name, metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err))
}
