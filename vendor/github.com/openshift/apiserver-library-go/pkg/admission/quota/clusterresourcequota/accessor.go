package clusterresourcequota

import (
	"context"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	kapierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilwait "k8s.io/apimachinery/pkg/util/wait"
	utilquota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/storage"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/utils/lru"

	quotav1 "github.com/openshift/api/quota/v1"
	quotatypedclient "github.com/openshift/client-go/quota/clientset/versioned/typed/quota/v1"
	quotalister "github.com/openshift/client-go/quota/listers/quota/v1"
	"github.com/openshift/library-go/pkg/quota/clusterquotamapping"
	quotautil "github.com/openshift/library-go/pkg/quota/quotautil"
)

type clusterQuotaAccessor struct {
	clusterQuotaLister quotalister.ClusterResourceQuotaLister
	namespaceLister    corev1listers.NamespaceLister
	clusterQuotaClient quotatypedclient.ClusterResourceQuotasGetter

	clusterQuotaMapper clusterquotamapping.ClusterQuotaMapper

	// updatedClusterQuotas holds a cache of quotas that we've updated.  This is used to pull the "really latest" during back to
	// back quota evaluations that touch the same quota doc.  This only works because we can compare etcd resourceVersions
	// for the same resource as integers.  Before this change: 22 updates with 12 conflicts.  after this change: 15 updates with 0 conflicts
	updatedClusterQuotas *lru.Cache
}

// newQuotaAccessor creates an object that conforms to the QuotaAccessor interface to be used to retrieve quota objects.
func newQuotaAccessor(
	clusterQuotaLister quotalister.ClusterResourceQuotaLister,
	namespaceLister corev1listers.NamespaceLister,
	clusterQuotaClient quotatypedclient.ClusterResourceQuotasGetter,
	clusterQuotaMapper clusterquotamapping.ClusterQuotaMapper,
) *clusterQuotaAccessor {
	updatedCache := lru.New(100)
	return &clusterQuotaAccessor{
		clusterQuotaLister:   clusterQuotaLister,
		namespaceLister:      namespaceLister,
		clusterQuotaClient:   clusterQuotaClient,
		clusterQuotaMapper:   clusterQuotaMapper,
		updatedClusterQuotas: updatedCache,
	}
}

// UpdateQuotaStatus the newQuota coming in will be incremented from the original.  The difference between the original
// and the new is the amount to add to the namespace total, but the total status is the used value itself
func (e *clusterQuotaAccessor) UpdateQuotaStatus(newQuota *corev1.ResourceQuota) error {
	clusterQuota, err := e.clusterQuotaLister.Get(newQuota.Name)
	if err != nil {
		return err
	}
	clusterQuota = e.checkCache(clusterQuota)

	// re-assign objectmeta
	// make a copy
	clusterQuota = clusterQuota.DeepCopy()
	clusterQuota.ObjectMeta = newQuota.ObjectMeta
	clusterQuota.Namespace = ""

	// determine change in usage
	usageDiff := utilquota.Subtract(newQuota.Status.Used, clusterQuota.Status.Total.Used)

	// update aggregate usage
	clusterQuota.Status.Total.Used = newQuota.Status.Used

	// update per namespace totals
	oldNamespaceTotals, _ := quotautil.GetResourceQuotasStatusByNamespace(clusterQuota.Status.Namespaces, newQuota.Namespace)
	namespaceTotalCopy := oldNamespaceTotals.DeepCopy()
	newNamespaceTotals := *namespaceTotalCopy
	newNamespaceTotals.Used = utilquota.Add(oldNamespaceTotals.Used, usageDiff)
	quotautil.InsertResourceQuotasStatus(&clusterQuota.Status.Namespaces, quotav1.ResourceQuotaStatusByNamespace{
		Namespace: newQuota.Namespace,
		Status:    newNamespaceTotals,
	})

	updatedQuota, err := e.clusterQuotaClient.ClusterResourceQuotas().UpdateStatus(context.TODO(), clusterQuota, metav1.UpdateOptions{})
	if err != nil {
		return err
	}

	e.updatedClusterQuotas.Add(clusterQuota.Name, updatedQuota)
	return nil
}

var etcdVersioner = storage.APIObjectVersioner{}

// checkCache compares the passed quota against the value in the look-aside cache and returns the newer
// if the cache is out of date, it deletes the stale entry.  This only works because of etcd resourceVersions
// being monotonically increasing integers
func (e *clusterQuotaAccessor) checkCache(clusterQuota *quotav1.ClusterResourceQuota) *quotav1.ClusterResourceQuota {
	uncastCachedQuota, ok := e.updatedClusterQuotas.Get(clusterQuota.Name)
	if !ok {
		return clusterQuota
	}
	cachedQuota := uncastCachedQuota.(*quotav1.ClusterResourceQuota)

	if etcdVersioner.CompareResourceVersion(clusterQuota, cachedQuota) >= 0 {
		e.updatedClusterQuotas.Remove(clusterQuota.Name)
		return clusterQuota
	}
	return cachedQuota
}

func (e *clusterQuotaAccessor) GetQuotas(namespaceName string) ([]corev1.ResourceQuota, error) {
	clusterQuotaNames, err := e.waitForReadyClusterQuotaNames(namespaceName)
	if err != nil {
		return nil, err
	}

	resourceQuotas := []corev1.ResourceQuota{}
	for _, clusterQuotaName := range clusterQuotaNames {
		clusterQuota, err := e.clusterQuotaLister.Get(clusterQuotaName)
		if kapierrors.IsNotFound(err) {
			continue
		}
		if err != nil {
			return nil, err
		}

		clusterQuota = e.checkCache(clusterQuota)

		// now convert to a ResourceQuota
		convertedQuota := corev1.ResourceQuota{}
		convertedQuota.ObjectMeta = clusterQuota.ObjectMeta
		convertedQuota.Namespace = namespaceName
		convertedQuota.Spec = clusterQuota.Spec.Quota
		convertedQuota.Status = clusterQuota.Status.Total
		resourceQuotas = append(resourceQuotas, convertedQuota)

	}

	return resourceQuotas, nil
}

func (e *clusterQuotaAccessor) waitForReadyClusterQuotaNames(namespaceName string) ([]string, error) {
	var clusterQuotaNames []string
	// wait for a valid mapping cache.  The overall response can be delayed for up to 10 seconds.
	err := utilwait.PollImmediate(100*time.Millisecond, 8*time.Second, func() (done bool, err error) {
		var namespaceSelectionFields clusterquotamapping.SelectionFields
		clusterQuotaNames, namespaceSelectionFields = e.clusterQuotaMapper.GetClusterQuotasFor(namespaceName)
		namespace, err := e.namespaceLister.Get(namespaceName)
		// if we can't find the namespace yet, just wait for the cache to update.  Requests to non-existent namespaces
		// may hang, but those people are doing something wrong and namespacelifecycle should reject them.
		if kapierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		if equality.Semantic.DeepEqual(namespaceSelectionFields, clusterquotamapping.GetSelectionFields(namespace)) {
			return true, nil
		}
		return false, nil
	})
	return clusterQuotaNames, err
}
