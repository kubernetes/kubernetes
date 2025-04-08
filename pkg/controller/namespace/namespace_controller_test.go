package namespace

import (
	"context"
	"testing"
	"time"

	"go.uber.org/goleak"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestNamespaceController_Shutdown(t *testing.T) {
	cases := map[string]struct {
		runner func(ctx context.Context, nm *NamespaceController)
	}{
		"run": {
			runner: func(ctx context.Context, nm *NamespaceController) { nm.Run(ctx, 1) },
		},
		"shutdown": {
			runner: func(ctx context.Context, nm *NamespaceController) { nm.ShutDown() },
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, tCtx := ktesting.NewTestContext(t)

			// Mock discoverResourcesFn to return a list of resources, without
			// this the NamespacedResourcesDeleter will call os.Exit in
			// initOpCache as there are no applicable resources.
			discoverResourcesFn := func() ([]*metav1.APIResourceList, error) {
				return []*metav1.APIResourceList{
					{
						GroupVersion: "v1",
						APIResources: []metav1.APIResource{
							{
								Name:       "pods",
								Namespaced: true,
								Kind:       "Pod",
								Verbs:      []string{"get", "list", "delete", "deletecollection", "create", "update"},
							},
						},
					},
				}, nil
			}

			cl := fake.NewSimpleClientset()

			informerFactory := informers.NewSharedInformerFactory(cl, controller.NoResyncPeriodFunc())
			namespaceInformer := informerFactory.Core().V1().Namespaces()
			informerFactory.Start(tCtx.Done())

			informerFactory.WaitForCacheSync(tCtx.Done())

			defer goleak.VerifyNone(t, goleak.IgnoreCurrent())

			nm, err := NewNamespaceController(tCtx, cl, nil, discoverResourcesFn, namespaceInformer, 0, v1.FinalizerKubernetes)
			if err != nil {
				t.Errorf("failed to create namespace controller: %v", err)
			}

			ctx, _ := context.WithTimeout(tCtx, 100*time.Millisecond)
			tc.runner(ctx, nm)
		},
		)
	}
}
