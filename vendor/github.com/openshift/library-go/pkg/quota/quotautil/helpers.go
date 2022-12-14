package quotautil

import (
	corev1 "k8s.io/api/core/v1"

	quotav1 "github.com/openshift/api/quota/v1"
)

func GetResourceQuotasStatusByNamespace(namespaceStatuses quotav1.ResourceQuotasStatusByNamespace, namespace string) (corev1.ResourceQuotaStatus, bool) {
	for i := range namespaceStatuses {
		curr := namespaceStatuses[i]
		if curr.Namespace == namespace {
			return curr.Status, true
		}
	}
	return corev1.ResourceQuotaStatus{}, false
}

func RemoveResourceQuotasStatusByNamespace(namespaceStatuses *quotav1.ResourceQuotasStatusByNamespace, namespace string) {
	newNamespaceStatuses := quotav1.ResourceQuotasStatusByNamespace{}
	for i := range *namespaceStatuses {
		curr := (*namespaceStatuses)[i]
		if curr.Namespace == namespace {
			continue
		}
		newNamespaceStatuses = append(newNamespaceStatuses, curr)
	}
	*namespaceStatuses = newNamespaceStatuses
}

func InsertResourceQuotasStatus(namespaceStatuses *quotav1.ResourceQuotasStatusByNamespace, newStatus quotav1.ResourceQuotaStatusByNamespace) {
	newNamespaceStatuses := quotav1.ResourceQuotasStatusByNamespace{}
	found := false
	for i := range *namespaceStatuses {
		curr := (*namespaceStatuses)[i]
		if curr.Namespace == newStatus.Namespace {
			// do this so that we don't change serialization order
			newNamespaceStatuses = append(newNamespaceStatuses, newStatus)
			found = true
			continue
		}
		newNamespaceStatuses = append(newNamespaceStatuses, curr)
	}
	if !found {
		newNamespaceStatuses = append(newNamespaceStatuses, newStatus)
	}
	*namespaceStatuses = newNamespaceStatuses
}
