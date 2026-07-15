package clusterquotamapping

import (
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	quotav1 "github.com/openshift/api/quota/v1"
)

type ClusterQuotaMapper interface {
	// GetClusterQuotasFor returns the list of clusterquota names that this namespace matches.  It also
	// returns the selectionFields associated with the namespace for the check so that callers can determine staleness
	GetClusterQuotasFor(namespaceName string) ([]string, SelectionFields)
	// GetNamespacesFor returns the list of namespace names that this cluster quota matches.  It also
	// returns the selector associated with the clusterquota for the check so that callers can determine staleness
	GetNamespacesFor(quotaName string) ([]string, quotav1.ClusterResourceQuotaSelector)

	AddListener(listener MappingChangeListener)
}

// MappingChangeListener is notified of changes to the mapping.  It must not block.
type MappingChangeListener interface {
	AddMapping(quotaName, namespaceName string)
	RemoveMapping(quotaName, namespaceName string)
}

type SelectionFields struct {
	Labels      map[string]string
	Annotations map[string]string
}

// clusterQuotaMapper gives thread safe access to the actual mappings that are being stored.
// Many method use a shareable read lock to check status followed by a non-shareable
// write lock which double checks the condition before proceeding.  Since locks aren't escalatable
// you have to perform the recheck because someone could have beaten you in.
type clusterQuotaMapper struct {
	lock sync.RWMutex

	// requiredQuotaToSelector indicates the latest label selector this controller has observed for a quota
	requiredQuotaToSelector map[string]quotav1.ClusterResourceQuotaSelector
	// requiredNamespaceToLabels indicates the latest selectionFields this controller has observed for a namespace
	requiredNamespaceToLabels map[string]SelectionFields
	// completedQuotaToSelector indicates the latest label selector this controller has scanned against namespaces
	completedQuotaToSelector map[string]quotav1.ClusterResourceQuotaSelector
	// completedNamespaceToLabels indicates the latest selectionFields this controller has scanned against cluster quotas
	completedNamespaceToLabels map[string]SelectionFields

	quotaToNamespaces map[string]sets.Set[string]
	namespaceToQuota  map[string]sets.Set[string]

	listeners []MappingChangeListener
}

func NewClusterQuotaMapper() *clusterQuotaMapper {
	return &clusterQuotaMapper{
		requiredQuotaToSelector:    map[string]quotav1.ClusterResourceQuotaSelector{},
		requiredNamespaceToLabels:  map[string]SelectionFields{},
		completedQuotaToSelector:   map[string]quotav1.ClusterResourceQuotaSelector{},
		completedNamespaceToLabels: map[string]SelectionFields{},

		quotaToNamespaces: map[string]sets.Set[string]{},
		namespaceToQuota:  map[string]sets.Set[string]{},
	}
}

func (m *clusterQuotaMapper) GetClusterQuotasFor(namespaceName string) ([]string, SelectionFields) {
	m.lock.RLock()
	defer m.lock.RUnlock()

	quotas, ok := m.namespaceToQuota[namespaceName]
	if !ok {
		return []string{}, m.completedNamespaceToLabels[namespaceName]
	}
	return sets.List(quotas), m.completedNamespaceToLabels[namespaceName]
}

func (m *clusterQuotaMapper) GetNamespacesFor(quotaName string) ([]string, quotav1.ClusterResourceQuotaSelector) {
	m.lock.RLock()
	defer m.lock.RUnlock()

	namespaces, ok := m.quotaToNamespaces[quotaName]
	if !ok {
		return []string{}, m.completedQuotaToSelector[quotaName]
	}
	return sets.List(namespaces), m.completedQuotaToSelector[quotaName]
}

func (m *clusterQuotaMapper) AddListener(listener MappingChangeListener) {
	m.lock.Lock()
	defer m.lock.Unlock()

	m.listeners = append(m.listeners, listener)
}

// requireQuota updates the selector requirements for the given quota.  This prevents stale updates to the mapping itself.
// returns true if a modification was made
func (m *clusterQuotaMapper) requireQuota(quota *quotav1.ClusterResourceQuota) bool {
	m.lock.RLock()
	selector, exists := m.requiredQuotaToSelector[quota.Name]
	m.lock.RUnlock()

	if selectorMatches(selector, exists, quota) {
		return false
	}

	m.lock.Lock()
	defer m.lock.Unlock()
	selector, exists = m.requiredQuotaToSelector[quota.Name]
	if selectorMatches(selector, exists, quota) {
		return false
	}

	m.requiredQuotaToSelector[quota.Name] = quota.Spec.Selector
	return true
}

// completeQuota updates the latest selector used to generate the mappings for this quota.  The value is returned
// by the Get methods for the mapping so that callers can determine staleness
func (m *clusterQuotaMapper) completeQuota(quota *quotav1.ClusterResourceQuota) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.completedQuotaToSelector[quota.Name] = quota.Spec.Selector
}

// removeQuota deletes a quota from all mappings
func (m *clusterQuotaMapper) removeQuota(quotaName string) {
	m.lock.Lock()
	defer m.lock.Unlock()

	delete(m.requiredQuotaToSelector, quotaName)
	delete(m.completedQuotaToSelector, quotaName)
	delete(m.quotaToNamespaces, quotaName)
	for namespaceName, quotas := range m.namespaceToQuota {
		if quotas.Has(quotaName) {
			quotas.Delete(quotaName)
			for _, listener := range m.listeners {
				listener.RemoveMapping(quotaName, namespaceName)
			}
		}
	}
}

// requireNamespace updates the label requirements for the given namespace.  This prevents stale updates to the mapping itself.
// returns true if a modification was made
func (m *clusterQuotaMapper) requireNamespace(namespace metav1.Object) bool {
	m.lock.RLock()
	selectionFields, exists := m.requiredNamespaceToLabels[namespace.GetName()]
	m.lock.RUnlock()

	if selectionFieldsMatch(selectionFields, exists, namespace) {
		return false
	}

	m.lock.Lock()
	defer m.lock.Unlock()
	selectionFields, exists = m.requiredNamespaceToLabels[namespace.GetName()]
	if selectionFieldsMatch(selectionFields, exists, namespace) {
		return false
	}

	m.requiredNamespaceToLabels[namespace.GetName()] = GetSelectionFields(namespace)
	return true
}

// completeNamespace updates the latest selectionFields used to generate the mappings for this namespace.  The value is returned
// by the Get methods for the mapping so that callers can determine staleness
func (m *clusterQuotaMapper) completeNamespace(namespace metav1.Object) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.completedNamespaceToLabels[namespace.GetName()] = GetSelectionFields(namespace)
}

// removeNamespace deletes a namespace from all mappings
func (m *clusterQuotaMapper) removeNamespace(namespaceName string) {
	m.lock.Lock()
	defer m.lock.Unlock()

	delete(m.requiredNamespaceToLabels, namespaceName)
	delete(m.completedNamespaceToLabels, namespaceName)
	delete(m.namespaceToQuota, namespaceName)
	for quotaName, namespaces := range m.quotaToNamespaces {
		if namespaces.Has(namespaceName) {
			namespaces.Delete(namespaceName)
			for _, listener := range m.listeners {
				listener.RemoveMapping(quotaName, namespaceName)
			}
		}
	}
}

func selectorMatches(selector quotav1.ClusterResourceQuotaSelector, exists bool, quota *quotav1.ClusterResourceQuota) bool {
	return exists && equality.Semantic.DeepEqual(selector, quota.Spec.Selector)
}
func selectionFieldsMatch(selectionFields SelectionFields, exists bool, namespace metav1.Object) bool {
	return exists && reflect.DeepEqual(selectionFields, GetSelectionFields(namespace))
}

// setMapping maps (or removes a mapping) between a clusterquota and a namespace
// It returns whether the action worked, whether the quota is out of date, whether the namespace is out of date
// This allows callers to decide whether to pull new information from the cache or simply skip execution
func (m *clusterQuotaMapper) setMapping(quota *quotav1.ClusterResourceQuota, namespace metav1.Object, remove bool) (bool /*added*/, bool /*quota matches*/, bool /*namespace matches*/) {
	m.lock.RLock()
	selector, selectorExists := m.requiredQuotaToSelector[quota.Name]
	selectionFields, selectionFieldsExist := m.requiredNamespaceToLabels[namespace.GetName()]
	m.lock.RUnlock()

	if !selectorMatches(selector, selectorExists, quota) {
		return false, false, selectionFieldsMatch(selectionFields, selectionFieldsExist, namespace)
	}
	if !selectionFieldsMatch(selectionFields, selectionFieldsExist, namespace) {
		return false, true, false
	}

	m.lock.Lock()
	defer m.lock.Unlock()
	selector, selectorExists = m.requiredQuotaToSelector[quota.Name]
	selectionFields, selectionFieldsExist = m.requiredNamespaceToLabels[namespace.GetName()]
	if !selectorMatches(selector, selectorExists, quota) {
		return false, false, selectionFieldsMatch(selectionFields, selectionFieldsExist, namespace)
	}
	if !selectionFieldsMatch(selectionFields, selectionFieldsExist, namespace) {
		return false, true, false
	}

	if remove {
		mutated := false

		namespaces, ok := m.quotaToNamespaces[quota.Name]
		if !ok {
			m.quotaToNamespaces[quota.Name] = sets.Set[string]{}
		} else {
			mutated = namespaces.Has(namespace.GetName())
			namespaces.Delete(namespace.GetName())
		}

		quotas, ok := m.namespaceToQuota[namespace.GetName()]
		if !ok {
			m.namespaceToQuota[namespace.GetName()] = sets.Set[string]{}
		} else {
			mutated = mutated || quotas.Has(quota.Name)
			quotas.Delete(quota.Name)
		}

		if mutated {
			for _, listener := range m.listeners {
				listener.RemoveMapping(quota.Name, namespace.GetName())
			}
		}

		return true, true, true
	}

	mutated := false

	namespaces, ok := m.quotaToNamespaces[quota.Name]
	if !ok {
		mutated = true
		m.quotaToNamespaces[quota.Name] = sets.New(namespace.GetName())
	} else {
		mutated = !namespaces.Has(namespace.GetName())
		namespaces.Insert(namespace.GetName())
	}

	quotas, ok := m.namespaceToQuota[namespace.GetName()]
	if !ok {
		mutated = true
		m.namespaceToQuota[namespace.GetName()] = sets.New(quota.Name)
	} else {
		mutated = mutated || !quotas.Has(quota.Name)
		quotas.Insert(quota.Name)
	}

	if mutated {
		for _, listener := range m.listeners {
			listener.AddMapping(quota.Name, namespace.GetName())
		}
	}

	return true, true, true

}

func GetSelectionFields(namespace metav1.Object) SelectionFields {
	return SelectionFields{Labels: namespace.GetLabels(), Annotations: namespace.GetAnnotations()}
}
