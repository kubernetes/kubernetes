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
	"k8s.io/client-go/1.5/pkg/api"
	"k8s.io/client-go/1.5/pkg/api/errors"
	"k8s.io/client-go/1.5/pkg/api/meta"
	"k8s.io/client-go/1.5/pkg/api/unversioned"
	"k8s.io/client-go/1.5/pkg/apis/apps"
	"k8s.io/client-go/1.5/pkg/apis/batch"
	"k8s.io/client-go/1.5/pkg/apis/certificates"
	"k8s.io/client-go/1.5/pkg/apis/extensions"
	"k8s.io/client-go/1.5/pkg/apis/policy"
	"k8s.io/client-go/1.5/pkg/labels"
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

	items, err := indexer.Index(NamespaceIndex, api.ObjectMeta{Namespace: namespace})
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

// StoreToDeploymentLister gives a store List and Exists methods. The store must contain only Deployments.
type StoreToDeploymentLister struct {
	Indexer
}

// Exists checks if the given deployment exists in the store.
func (s *StoreToDeploymentLister) Exists(deployment *extensions.Deployment) (bool, error) {
	_, exists, err := s.Indexer.Get(deployment)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToDeploymentLister lists all deployments in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToDeploymentLister) List() (deployments []extensions.Deployment, err error) {
	for _, c := range s.Indexer.List() {
		deployments = append(deployments, *(c.(*extensions.Deployment)))
	}
	return deployments, nil
}

// GetDeploymentsForReplicaSet returns a list of deployments managing a replica set. Returns an error only if no matching deployments are found.
func (s *StoreToDeploymentLister) GetDeploymentsForReplicaSet(rs *extensions.ReplicaSet) (deployments []extensions.Deployment, err error) {
	if len(rs.Labels) == 0 {
		err = fmt.Errorf("no deployments found for ReplicaSet %v because it has no labels", rs.Name)
		return
	}

	// TODO: MODIFY THIS METHOD so that it checks for the podTemplateSpecHash label
	dList, err := s.Deployments(rs.Namespace).List(labels.Everything())
	if err != nil {
		return
	}
	for _, d := range dList {
		selector, err := unversioned.LabelSelectorAsSelector(d.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		// If a deployment with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(rs.Labels)) {
			continue
		}
		deployments = append(deployments, d)
	}
	if len(deployments) == 0 {
		err = fmt.Errorf("could not find deployments set for ReplicaSet %s in namespace %s with labels: %v", rs.Name, rs.Namespace, rs.Labels)
	}
	return
}

type storeToDeploymentNamespacer struct {
	indexer   Indexer
	namespace string
}

// storeToDeploymentNamespacer lists deployments under its namespace in the store.
func (s storeToDeploymentNamespacer) List(selector labels.Selector) (deployments []extensions.Deployment, err error) {
	if s.namespace == api.NamespaceAll {
		for _, m := range s.indexer.List() {
			d := *(m.(*extensions.Deployment))
			if selector.Matches(labels.Set(d.Labels)) {
				deployments = append(deployments, d)
			}
		}
		return
	}

	key := &extensions.Deployment{ObjectMeta: api.ObjectMeta{Namespace: s.namespace}}
	items, err := s.indexer.Index(NamespaceIndex, key)
	if err != nil {
		// Ignore error; do slow search without index.
		glog.Warningf("can not retrieve list of objects using index : %v", err)
		for _, m := range s.indexer.List() {
			d := *(m.(*extensions.Deployment))
			if s.namespace == d.Namespace && selector.Matches(labels.Set(d.Labels)) {
				deployments = append(deployments, d)
			}
		}
		return deployments, nil
	}
	for _, m := range items {
		d := *(m.(*extensions.Deployment))
		if selector.Matches(labels.Set(d.Labels)) {
			deployments = append(deployments, d)
		}
	}
	return
}

func (s *StoreToDeploymentLister) Deployments(namespace string) storeToDeploymentNamespacer {
	return storeToDeploymentNamespacer{s.Indexer, namespace}
}

// GetDeploymentsForPods returns a list of deployments managing a pod. Returns an error only if no matching deployments are found.
func (s *StoreToDeploymentLister) GetDeploymentsForPod(pod *api.Pod) (deployments []extensions.Deployment, err error) {
	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no deployments found for Pod %v because it has no labels", pod.Name)
		return
	}

	if len(pod.Labels[extensions.DefaultDeploymentUniqueLabelKey]) == 0 {
		return
	}

	dList, err := s.Deployments(pod.Namespace).List(labels.Everything())
	if err != nil {
		return
	}
	for _, d := range dList {
		selector, err := unversioned.LabelSelectorAsSelector(d.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		// If a deployment with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		deployments = append(deployments, d)
	}
	if len(deployments) == 0 {
		err = fmt.Errorf("could not find deployments set for Pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StoreToReplicaSetLister gives a store List and Exists methods. The store must contain only ReplicaSets.
type StoreToReplicaSetLister struct {
	Store
}

// Exists checks if the given ReplicaSet exists in the store.
func (s *StoreToReplicaSetLister) Exists(rs *extensions.ReplicaSet) (bool, error) {
	_, exists, err := s.Store.Get(rs)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// List lists all ReplicaSets in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToReplicaSetLister) List() (rss []extensions.ReplicaSet, err error) {
	for _, rs := range s.Store.List() {
		rss = append(rss, *(rs.(*extensions.ReplicaSet)))
	}
	return rss, nil
}

type storeReplicaSetsNamespacer struct {
	store     Store
	namespace string
}

func (s storeReplicaSetsNamespacer) List(selector labels.Selector) (rss []extensions.ReplicaSet, err error) {
	for _, c := range s.store.List() {
		rs := *(c.(*extensions.ReplicaSet))
		if s.namespace == api.NamespaceAll || s.namespace == rs.Namespace {
			if selector.Matches(labels.Set(rs.Labels)) {
				rss = append(rss, rs)
			}
		}
	}
	return
}

func (s storeReplicaSetsNamespacer) Get(name string) (*extensions.ReplicaSet, error) {
	obj, exists, err := s.store.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(extensions.Resource("replicaset"), name)
	}
	return obj.(*extensions.ReplicaSet), nil
}

func (s *StoreToReplicaSetLister) ReplicaSets(namespace string) storeReplicaSetsNamespacer {
	return storeReplicaSetsNamespacer{s.Store, namespace}
}

// GetPodReplicaSets returns a list of ReplicaSets managing a pod. Returns an error only if no matching ReplicaSets are found.
func (s *StoreToReplicaSetLister) GetPodReplicaSets(pod *api.Pod) (rss []extensions.ReplicaSet, err error) {
	var selector labels.Selector
	var rs extensions.ReplicaSet

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no ReplicaSets found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		rs = *m.(*extensions.ReplicaSet)
		if rs.Namespace != pod.Namespace {
			continue
		}
		selector, err = unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			err = fmt.Errorf("invalid selector: %v", err)
			return
		}

		// If a ReplicaSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		rss = append(rss, rs)
	}
	if len(rss) == 0 {
		err = fmt.Errorf("could not find ReplicaSet for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
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

// StoreToJobLister gives a store List and Exists methods. The store must contain only Jobs.
type StoreToJobLister struct {
	Store
}

// Exists checks if the given job exists in the store.
func (s *StoreToJobLister) Exists(job *batch.Job) (bool, error) {
	_, exists, err := s.Store.Get(job)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToJobLister lists all jobs in the store.
func (s *StoreToJobLister) List() (jobs batch.JobList, err error) {
	for _, c := range s.Store.List() {
		jobs.Items = append(jobs.Items, *(c.(*batch.Job)))
	}
	return jobs, nil
}

// GetPodJobs returns a list of jobs managing a pod. Returns an error only if no matching jobs are found.
func (s *StoreToJobLister) GetPodJobs(pod *api.Pod) (jobs []batch.Job, err error) {
	var selector labels.Selector
	var job batch.Job

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no jobs found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		job = *m.(*batch.Job)
		if job.Namespace != pod.Namespace {
			continue
		}

		selector, _ = unversioned.LabelSelectorAsSelector(job.Spec.Selector)
		if !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		jobs = append(jobs, job)
	}
	if len(jobs) == 0 {
		err = fmt.Errorf("could not find jobs for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
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

// Typed wrapper around a store of PersistentVolumeClaims
type StoreToPVCFetcher struct {
	Store
}

// GetPersistentVolumeClaimInfo returns cached data for the PersistentVolumeClaim 'id'.
func (s *StoreToPVCFetcher) GetPersistentVolumeClaimInfo(namespace string, id string) (*api.PersistentVolumeClaim, error) {
	o, exists, err := s.Get(&api.PersistentVolumeClaim{ObjectMeta: api.ObjectMeta{Namespace: namespace, Name: id}})
	if err != nil {
		return nil, fmt.Errorf("error retrieving PersistentVolumeClaim '%s/%s' from cache: %v", namespace, id, err)
	}

	if !exists {
		return nil, fmt.Errorf("PersistentVolumeClaim '%s/%s' not found", namespace, id)
	}

	return o.(*api.PersistentVolumeClaim), nil
}

// StoreToPetSetLister gives a store List and Exists methods. The store must contain only PetSets.
type StoreToPetSetLister struct {
	Store
}

// Exists checks if the given PetSet exists in the store.
func (s *StoreToPetSetLister) Exists(ps *apps.PetSet) (bool, error) {
	_, exists, err := s.Store.Get(ps)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// List lists all PetSets in the store.
func (s *StoreToPetSetLister) List() (psList []apps.PetSet, err error) {
	for _, ps := range s.Store.List() {
		psList = append(psList, *(ps.(*apps.PetSet)))
	}
	return psList, nil
}

type storePetSetsNamespacer struct {
	store     Store
	namespace string
}

func (s *StoreToPetSetLister) PetSets(namespace string) storePetSetsNamespacer {
	return storePetSetsNamespacer{s.Store, namespace}
}

// GetPodPetSets returns a list of PetSets managing a pod. Returns an error only if no matching PetSets are found.
func (s *StoreToPetSetLister) GetPodPetSets(pod *api.Pod) (psList []apps.PetSet, err error) {
	var selector labels.Selector
	var ps apps.PetSet

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no PetSets found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		ps = *m.(*apps.PetSet)
		if ps.Namespace != pod.Namespace {
			continue
		}
		selector, err = unversioned.LabelSelectorAsSelector(ps.Spec.Selector)
		if err != nil {
			err = fmt.Errorf("invalid selector: %v", err)
			return
		}

		// If a PetSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		psList = append(psList, ps)
	}
	if len(psList) == 0 {
		err = fmt.Errorf("could not find PetSet for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
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

// IndexerToNamespaceLister gives an Indexer List method
type IndexerToNamespaceLister struct {
	Indexer
}

// List returns a list of namespaces
func (i *IndexerToNamespaceLister) List(selector labels.Selector) (namespaces []*api.Namespace, err error) {
	for _, m := range i.Indexer.List() {
		namespace := m.(*api.Namespace)
		if selector.Matches(labels.Set(namespace.Labels)) {
			namespaces = append(namespaces, namespace)
		}
	}

	return namespaces, nil
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
