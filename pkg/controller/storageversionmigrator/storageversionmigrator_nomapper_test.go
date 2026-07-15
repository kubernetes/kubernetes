/*
Regression test: an SVM targeting a resource absent from the RESTMapper must not panic.
Before the fix, sync() dereferenced a nil *schema.GroupVersionResource via gvr.String()
(resourceFor returns (nil, false, nil) when the resource is unknown), crashing the KCM.
See findings/storageversionmigrator.md.
*/

package storageversionmigrator

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	kubefake "k8s.io/client-go/kubernetes/fake"
)

func TestSyncDoesNotPanicWhenResourceNotInMapper(t *testing.T) {
	ctx := context.Background()

	// Non-terminal SVM with a resourceVersion set (passes the guards), targeting a
	// resource that is NOT in the test RESTMapper (which only knows apps/deployments).
	svm := newSVM("svm-nomap", "100")
	svm.Spec.Resource = metav1.GroupResource{
		Group:    "nonexistent.example.com",
		Resource: "widgets",
	}

	kubeClient := kubefake.NewClientset(svm)
	factory := informers.NewSharedInformerFactory(kubeClient, 0)
	svmInformer := factory.Storagemigration().V1().StorageVersionMigrations()
	require.NoError(t, svmInformer.Informer().GetStore().Add(svm))
	ctrl := newTestSVMController(kubeClient, svmInformer, &mockGraphBuilder{})

	// Before the fix this call panicked (nil pointer dereference on gvr.String()),
	// which crashes the whole kube-controller-manager.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("sync panicked for a resource absent from the RESTMapper: %v", r)
		}
	}()

	// CreationTimestamp is recent, so sync should requeue with an error (not panic).
	err := ctrl.sync(ctx, "svm-nomap")
	require.Error(t, err)
	require.Contains(t, err.Error(), "resource does not exist in rest mapper")
}
