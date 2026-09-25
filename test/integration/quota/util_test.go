/*
Copyright 2026 The Kubernetes Authors.

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

package quota

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// quotaPollInterval/quotaPollTimeout are the poll budget for quota assertions on this race-prone
// surface. The historical resourceQuotaTimeout (10s) was raised to a full minute by several
// deflake commits, so new tests use the larger budget.
const (
	quotaPollInterval = time.Second
	quotaPollTimeout  = time.Minute
)

// quotaServerOptions configures a quota integration test server. The zero value is valid: no
// admission config.
type quotaServerOptions struct {
	// admissionConfigFile is the path to an apiserver admission ConfigFile (e.g. for
	// limitedResources). "" disables it.
	admissionConfigFile string
}

// quotaTestServer is a running apiserver wired to the resourcequota controller. client is the
// concrete type so the existing package helpers (waitForQuota, scale) can be reused directly.
type quotaTestServer struct {
	client *clientset.Clientset
	ctx    context.Context
}

// startQuotaTestServer starts an apiserver (with the ServiceAccount admission plugin disabled,
// since no service-account controller runs here) and the resourcequota controller. It mirrors the
// inline setup in quota_test.go, with one addition: the
// controller is wired with quotainstall.DefaultUpdateFilter so that update-driven replenishment
// (e.g. a pod transitioning to a terminal phase) fires exactly as it does under the real
// controller-manager (cmd/kube-controller-manager/app/core.go). Returns the server handle and a
// teardown func; callers defer the teardown before creating namespaces so they tear down first.
func startQuotaTestServer(t *testing.T, opts quotaServerOptions) (*quotaTestServer, func()) {
	t.Helper()
	tCtx := ktesting.Init(t)

	_, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(o *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have a serviceaccount controller running.
			o.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
			if opts.admissionConfigFile != "" {
				o.Admission.GenericAdmission.ConfigFile = opts.admissionConfigFile
			}
		},
	})

	clientSet := clientset.NewForConfigOrDie(kubeConfig)

	informerFactory := informers.NewSharedInformerFactory(clientSet, controller.NoResyncPeriodFunc())

	discoveryFunc := clientSet.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informerFactory.ForResource)
	qc, err := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource, informerFactory)
	if err != nil {
		tearDownFn()
		t.Fatalf("unexpected err: %v", err)
	}
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               clientSet.CoreV1(),
		ResourceQuotaInformer:     informerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informerFactory,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
		// Match the real controller-manager so update events drive replenishment.
		UpdateFilter: quotainstall.DefaultUpdateFilter(),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(tCtx, resourceQuotaControllerOptions)
	if err != nil {
		tearDownFn()
		t.Fatalf("unexpected err: %v", err)
	}
	go resourceQuotaController.Run(tCtx, 2)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(tCtx, discoveryFunc, 30*time.Second)

	informerFactory.Start(tCtx.Done())
	close(informersStarted)

	return &quotaTestServer{client: clientSet, ctx: tCtx}, tearDownFn
}

// newResourceQuota builds a ResourceQuota in ns with the given hard limits and optional scopes.
// The namespace is set on the object so it can be passed to waitForQuota.
func newResourceQuota(ns, name string, hard v1.ResourceList, scopes ...v1.ResourceQuotaScope) *v1.ResourceQuota {
	return &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard:   hard,
			Scopes: scopes,
		},
	}
}

// newPod builds a single-container pod. requests/limits may be nil. The namespace is taken from
// the create call (createPodExpectAllowed, createPodExpectForbidden), not set here.
func newPod(name string, requests, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "container",
				Image: "busybox",
				Resources: v1.ResourceRequirements{
					Requests: requests,
					Limits:   limits,
				},
			}},
		},
	}
}

// createPodExpectAllowed polls attempting to create pod until the create succeeds, absorbing the
// admission-cache lag that follows a quota create or an object delete (the "now allowed"
// direction; cf. deflake commit b8676e4763f, which retries creates for the same reason). Returns
// the created pod.
func createPodExpectAllowed(ctx context.Context, c clientset.Interface, namespace string, pod *v1.Pod, t *testing.T) *v1.Pod {
	t.Helper()
	var created *v1.Pod
	err := wait.PollUntilContextTimeout(ctx, quotaPollInterval, quotaPollTimeout, true, func(ctx context.Context) (bool, error) {
		p, err := c.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
		switch {
		case err == nil:
			created = p
			return true, nil
		case apierrors.IsForbidden(err):
			return false, nil
		default:
			return false, err
		}
	})
	if err != nil {
		t.Fatalf("creating pod %s/%s should eventually be allowed but got: %v", namespace, pod.Name, err)
	}
	return created
}

// createPodExpectForbidden polls creating the pod until rejected with 403 Forbidden (a quota denial).
// Quota admission is eventually consistent: while its cached status.used still lags, a create that
// should be denied may briefly succeed; that pod (or a leftover) is deleted and polling continues until
// it stays forbidden. Never forbidden = test fails.
func createPodExpectForbidden(ctx context.Context, c clientset.Interface, namespace string, pod *v1.Pod, t *testing.T) {
	t.Helper()
	err := wait.PollUntilContextTimeout(ctx, quotaPollInterval, quotaPollTimeout, true, func(ctx context.Context) (bool, error) {
		p, err := c.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
		switch {
		case apierrors.IsForbidden(err):
			return true, nil
		case err != nil:
			return false, err
		default:
			// Unexpected success due to admission-cache lag (pods here are unscheduled, so the
			// delete is immediate): clean it up and keep polling.
			if delErr := c.CoreV1().Pods(namespace).Delete(ctx, p.Name, metav1.DeleteOptions{}); delErr != nil && !apierrors.IsNotFound(delErr) {
				return false, delErr
			}
			return false, nil
		}
	})
	if err != nil {
		t.Fatalf("creating pod %s/%s should eventually be forbidden by quota but got: %v", namespace, pod.Name, err)
	}
}
