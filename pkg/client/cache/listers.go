/*
Copyright 2014 The Kubernetes Authors.

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

package cache

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
)

// AppendFunc is used to add a matching item to whatever list the caller is using
type AppendFunc func(interface{})

func ListAll(store Store, selector labels.Selector, appendFn AppendFunc) error {
	for _, m := range store.List() {
		metadata, err := meta.Accessor(m)
		if err != nil {
			return err
		}
		if selector.Matches(labels.Set(metadata.GetLabels())) {
			appendFn(m)
		}
	}
	return nil
}

func ListAllByNamespace(indexer Indexer, namespace string, selector labels.Selector, appendFn AppendFunc) error {
	if namespace == api.NamespaceAll {
		for _, m := range indexer.List() {
			metadata, err := meta.Accessor(m)
			if err != nil {
				return err
			}
			if selector.Matches(labels.Set(metadata.GetLabels())) {
				appendFn(m)
			}
		}
		return nil
	}

	items, err := indexer.Index(NamespaceIndex, &api.ObjectMeta{Namespace: namespace})
	if err != nil {
		// Ignore error; do slow search without index.
		glog.Warningf("can not retrieve list of objects using index : %v", err)
		for _, m := range indexer.List() {
			metadata, err := meta.Accessor(m)
			if err != nil {
				return err
			}
			if metadata.GetNamespace() == namespace && selector.Matches(labels.Set(metadata.GetLabels())) {
				appendFn(m)
			}

		}
		return nil
	}
	for _, m := range items {
		metadata, err := meta.Accessor(m)
		if err != nil {
			return err
		}
		if selector.Matches(labels.Set(metadata.GetLabels())) {
			appendFn(m)
		}
	}

	return nil
}

// GenericLister is a lister skin on a generic Indexer
type GenericLister interface {
	// List will return all objects across namespaces
	List(selector labels.Selector) (ret []runtime.Object, err error)
	// Get will attempt to retrieve assuming that name==key
	Get(name string) (runtime.Object, error)
	// ByNamespace will give you a GenericNamespaceLister for one namespace
	ByNamespace(namespace string) GenericNamespaceLister
}

// GenericNamespaceLister is a lister skin on a generic Indexer
type GenericNamespaceLister interface {
	// List will return all objects in this namespace
	List(selector labels.Selector) (ret []runtime.Object, err error)
	// Get will attempt to retrieve by namespace and name
	Get(name string) (runtime.Object, error)
}

func NewGenericLister(indexer Indexer, resource unversioned.GroupResource) GenericLister {
	return &genericLister{indexer: indexer, resource: resource}
}

type genericLister struct {
	indexer  Indexer
	resource unversioned.GroupResource
}

func (s *genericLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *genericLister) ByNamespace(namespace string) GenericNamespaceLister {
	return &genericNamespaceLister{indexer: s.indexer, namespace: namespace, resource: s.resource}
}

func (s *genericLister) Get(name string) (runtime.Object, error) {
	obj, exists, err := s.indexer.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(s.resource, name)
	}
	return obj.(runtime.Object), nil
}

type genericNamespaceLister struct {
	indexer   Indexer
	namespace string
	resource  unversioned.GroupResource
}

func (s *genericNamespaceLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	err = ListAllByNamespace(s.indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *genericNamespaceLister) Get(name string) (runtime.Object, error) {
	obj, exists, err := s.indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(s.resource, name)
	}
	return obj.(runtime.Object), nil
}

//  TODO: generate these classes and methods for all resources of interest using
// a script.  Can use "go generate" once 1.4 is supported by all users.

// NodeConditionPredicate is a function that indicates whether the given node's conditions meet
// some set of criteria defined by the function.
type NodeConditionPredicate func(node *api.Node) bool

// StoreToNodeLister makes a Store have the List method of the client.NodeInterface
// The Store must contain (only) Nodes.
type StoreToNodeLister struct {
	Store
}

func (s *StoreToNodeLister) List() (machines api.NodeList, err error) {
	for _, m := range s.Store.List() {
		machines.Items = append(machines.Items, *(m.(*api.Node)))
	}
	return machines, nil
}

// NodeCondition returns a storeToNodeConditionLister
func (s *StoreToNodeLister) NodeCondition(predicate NodeConditionPredicate) storeToNodeConditionLister {
	// TODO: Move this filtering server side. Currently our selectors don't facilitate searching through a list so we
	// have the reflector filter out the Unschedulable field and sift through node conditions in the lister.
	return storeToNodeConditionLister{s.Store, predicate}
}

// storeToNodeConditionLister filters and returns nodes matching the given type and status from the store.
type storeToNodeConditionLister struct {
	store     Store
	predicate NodeConditionPredicate
}

// List returns a list of nodes that match the conditions defined by the predicate functions in the storeToNodeConditionLister.
func (s storeToNodeConditionLister) List() (nodes []*api.Node, err error) {
	for _, m := range s.store.List() {
		node := m.(*api.Node)
		if s.predicate(node) {
			nodes = append(nodes, node)
		} else {
			glog.V(5).Infof("Node %s matches none of the conditions", node.Name)
		}
	}
	return
}

// StoreToDaemonSetLister gives a store List and Exists methods. The store must contain only DaemonSets.
type StoreToDaemonSetLister struct {
	Store
}

// Exists checks if the given daemon set exists in the store.
func (s *StoreToDaemonSetLister) Exists(ds *extensions.DaemonSet) (bool, error) {
	_, exists, err := s.Store.Get(ds)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// List lists all daemon sets in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToDaemonSetLister) List() (dss extensions.DaemonSetList, err error) {
	for _, c := range s.Store.List() {
		dss.Items = append(dss.Items, *(c.(*extensions.DaemonSet)))
	}
	return dss, nil
}

// GetPodDaemonSets returns a list of daemon sets managing a pod.
// Returns an error if and only if no matching daemon sets are found.
func (s *StoreToDaemonSetLister) GetPodDaemonSets(pod *api.Pod) (daemonSets []extensions.DaemonSet, err error) {
	var selector labels.Selector
	var daemonSet extensions.DaemonSet

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no daemon sets found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		daemonSet = *m.(*extensions.DaemonSet)
		if daemonSet.Namespace != pod.Namespace {
			continue
		}
		selector, err = unversioned.LabelSelectorAsSelector(daemonSet.Spec.Selector)
		if err != nil {
			// this should not happen if the DaemonSet passed validation
			return nil, err
		}

		// If a daemonSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		daemonSets = append(daemonSets, daemonSet)
	}
	if len(daemonSets) == 0 {
		err = fmt.Errorf("could not find daemon set for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StoreToEndpointsLister makes a Store that lists endpoints.
type StoreToEndpointsLister struct {
	Store
}

// List lists all endpoints in the store.
func (s *StoreToEndpointsLister) List() (services api.EndpointsList, err error) {
	for _, m := range s.Store.List() {
		services.Items = append(services.Items, *(m.(*api.Endpoints)))
	}
	return services, nil
}

// GetServiceEndpoints returns the endpoints of a service, matched on service name.
func (s *StoreToEndpointsLister) GetServiceEndpoints(svc *api.Service) (ep api.Endpoints, err error) {
	for _, m := range s.Store.List() {
		ep = *m.(*api.Endpoints)
		if svc.Name == ep.Name && svc.Namespace == ep.Namespace {
			return ep, nil
		}
	}
	err = fmt.Errorf("could not find endpoints for service: %v", svc.Name)
	return
}

// Typed wrapper around a store of PersistentVolumes
type StoreToPVFetcher struct {
	Store
}

// GetPersistentVolumeInfo returns cached data for the PersistentVolume 'id'.
func (s *StoreToPVFetcher) GetPersistentVolumeInfo(id string) (*api.PersistentVolume, error) {
	o, exists, err := s.Get(&api.PersistentVolume{ObjectMeta: api.ObjectMeta{Name: id}})

	if err != nil {
		return nil, fmt.Errorf("error retrieving PersistentVolume '%v' from cache: %v", id, err)
	}

	if !exists {
		return nil, fmt.Errorf("PersistentVolume '%v' not found", id)
	}

	return o.(*api.PersistentVolume), nil
}

// StoreToStatefulSetLister gives a store List and Exists methods. The store must contain only StatefulSets.
type StoreToStatefulSetLister struct {
	Store
}

// Exists checks if the given StatefulSet exists in the store.
func (s *StoreToStatefulSetLister) Exists(ps *apps.StatefulSet) (bool, error) {
	_, exists, err := s.Store.Get(ps)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// List lists all StatefulSets in the store.
func (s *StoreToStatefulSetLister) List() (psList []apps.StatefulSet, err error) {
	for _, ps := range s.Store.List() {
		psList = append(psList, *(ps.(*apps.StatefulSet)))
	}
	return psList, nil
}

type storeStatefulSetsNamespacer struct {
	store     Store
	namespace string
}

func (s *StoreToStatefulSetLister) StatefulSets(namespace string) storeStatefulSetsNamespacer {
	return storeStatefulSetsNamespacer{s.Store, namespace}
}

// GetPodStatefulSets returns a list of StatefulSets managing a pod. Returns an error only if no matching StatefulSets are found.
func (s *StoreToStatefulSetLister) GetPodStatefulSets(pod *api.Pod) (psList []apps.StatefulSet, err error) {
	var selector labels.Selector
	var ps apps.StatefulSet

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no StatefulSets found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		ps = *m.(*apps.StatefulSet)
		if ps.Namespace != pod.Namespace {
			continue
		}
		selector, err = unversioned.LabelSelectorAsSelector(ps.Spec.Selector)
		if err != nil {
			err = fmt.Errorf("invalid selector: %v", err)
			return
		}

		// If a StatefulSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		psList = append(psList, ps)
	}
	if len(psList) == 0 {
		err = fmt.Errorf("could not find StatefulSet for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StoreToCertificateRequestLister gives a store List and Exists methods. The store must contain only CertificateRequests.
type StoreToCertificateRequestLister struct {
	Store
}

// Exists checks if the given csr exists in the store.
func (s *StoreToCertificateRequestLister) Exists(csr *certificates.CertificateSigningRequest) (bool, error) {
	_, exists, err := s.Store.Get(csr)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToCertificateRequestLister lists all csrs in the store.
func (s *StoreToCertificateRequestLister) List() (csrs certificates.CertificateSigningRequestList, err error) {
	for _, c := range s.Store.List() {
		csrs.Items = append(csrs.Items, *(c.(*certificates.CertificateSigningRequest)))
	}
	return csrs, nil
}

type StoreToPodDisruptionBudgetLister struct {
	Store
}

// GetPodPodDisruptionBudgets returns a list of PodDisruptionBudgets matching a pod.  Returns an error only if no matching PodDisruptionBudgets are found.
func (s *StoreToPodDisruptionBudgetLister) GetPodPodDisruptionBudgets(pod *api.Pod) (pdbList []policy.PodDisruptionBudget, err error) {
	var selector labels.Selector

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no PodDisruptionBudgets found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		pdb, ok := m.(*policy.PodDisruptionBudget)
		if !ok {
			glog.Errorf("Unexpected: %v is not a PodDisruptionBudget", m)
			continue
		}
		if pdb.Namespace != pod.Namespace {
			continue
		}
		selector, err = unversioned.LabelSelectorAsSelector(pdb.Spec.Selector)
		if err != nil {
			glog.Warningf("invalid selector: %v", err)
			// TODO(mml): add an event to the PDB
			continue
		}

		// If a PDB with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		pdbList = append(pdbList, *pdb)
	}
	if len(pdbList) == 0 {
		err = fmt.Errorf("could not find PodDisruptionBudget for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StorageClassLister knows how to list storage classes
type StorageClassLister interface {
	List(selector labels.Selector) (ret []*storage.StorageClass, err error)
	Get(name string) (*storage.StorageClass, error)
}

// storageClassLister implements StorageClassLister
type storageClassLister struct {
	indexer Indexer
}

// NewStorageClassLister returns a new lister.
func NewStorageClassLister(indexer Indexer) StorageClassLister {
	return &storageClassLister{indexer: indexer}
}

// List returns a list of storage classes
func (s *storageClassLister) List(selector labels.Selector) (ret []*storage.StorageClass, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*storage.StorageClass))
	})
	return ret, err
}

// List returns a list of storage classes
func (s *storageClassLister) Get(name string) (*storage.StorageClass, error) {
	key := &storage.StorageClass{ObjectMeta: api.ObjectMeta{Name: name}}
	obj, exists, err := s.indexer.Get(key)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(storage.Resource("storageclass"), name)
	}
	return obj.(*storage.StorageClass), nil
}
