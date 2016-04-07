/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/labels"
)

//  TODO: generate these classes and methods for all resources of interest using
// a script.  Can use "go generate" once 1.4 is supported by all users.

// StoreToPodLister makes a Store have the List method of the client.PodInterface
// The Store must contain (only) Pods.
//
// Example:
// s := cache.NewStore()
// lw := cache.ListWatch{Client: c, FieldSelector: sel, Resource: "pods"}
// r := cache.NewReflector(lw, &api.Pod{}, s).Run()
// l := StoreToPodLister{s}
// l.List()
type StoreToPodLister struct {
	Indexer
}

// Please note that selector is filtering among the pods that have gotten into
// the store; there may have been some filtering that already happened before
// that.
//
// TODO: converge on the interface in pkg/client.
func (s *StoreToPodLister) List(selector labels.Selector) (pods []*api.Pod, err error) {
	// TODO: it'd be great to just call
	// s.Pods(api.NamespaceAll).List(selector), however then we'd have to
	// remake the list.Items as a []*api.Pod. So leave this separate for
	// now.
	for _, m := range s.Indexer.List() {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			pods = append(pods, pod)
		}
	}
	return pods, nil
}

// Pods is taking baby steps to be more like the api in pkg/client
func (s *StoreToPodLister) Pods(namespace string) storePodsNamespacer {
	return storePodsNamespacer{s.Indexer, namespace}
}

type storePodsNamespacer struct {
	indexer   Indexer
	namespace string
}

// Please note that selector is filtering among the pods that have gotten into
// the store; there may have been some filtering that already happened before
// that.
func (s storePodsNamespacer) List(selector labels.Selector) (pods api.PodList, err error) {
	list := api.PodList{}
	if s.namespace == api.NamespaceAll {
		for _, m := range s.indexer.List() {
			pod := m.(*api.Pod)
			if selector.Matches(labels.Set(pod.Labels)) {
				list.Items = append(list.Items, *pod)
			}
		}
		return list, nil
	}

	key := &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: s.namespace}}
	items, err := s.indexer.Index(NamespaceIndex, key)
	if err != nil {
		return api.PodList{}, err
	}
	for _, m := range items {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			list.Items = append(list.Items, *pod)
		}
	}
	return list, nil
}

// Exists returns true if a pod matching the namespace/name of the given pod exists in the store.
func (s *StoreToPodLister) Exists(pod *api.Pod) (bool, error) {
	_, exists, err := s.Indexer.Get(pod)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// NodeConditionPredicate is a function that indicates whether the given node's conditions meet
// some set of criteria defined by the function.
type NodeConditionPredicate func(node api.Node) bool

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
func (s storeToNodeConditionLister) List() (nodes api.NodeList, err error) {
	for _, m := range s.store.List() {
		node := *m.(*api.Node)
		if s.predicate(node) {
			nodes.Items = append(nodes.Items, node)
		} else {
			glog.V(5).Infof("Node %s matches none of the conditions", node.Name)
		}
	}
	return
}

// StoreToReplicationControllerLister gives a store List and Exists methods. The store must contain only ReplicationControllers.
type StoreToReplicationControllerLister struct {
	Indexer
}

// Exists checks if the given rc exists in the store.
func (s *StoreToReplicationControllerLister) Exists(controller *api.ReplicationController) (bool, error) {
	_, exists, err := s.Indexer.Get(controller)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToReplicationControllerLister lists all controllers in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToReplicationControllerLister) List() (controllers []api.ReplicationController, err error) {
	for _, c := range s.Indexer.List() {
		controllers = append(controllers, *(c.(*api.ReplicationController)))
	}
	return controllers, nil
}

func (s *StoreToReplicationControllerLister) ReplicationControllers(namespace string) storeReplicationControllersNamespacer {
	return storeReplicationControllersNamespacer{s.Indexer, namespace}
}

type storeReplicationControllersNamespacer struct {
	indexer   Indexer
	namespace string
}

func (s storeReplicationControllersNamespacer) List(selector labels.Selector) (controllers []api.ReplicationController, err error) {
	if s.namespace == api.NamespaceAll {
		for _, m := range s.indexer.List() {
			rc := *(m.(*api.ReplicationController))
			if selector.Matches(labels.Set(rc.Labels)) {
				controllers = append(controllers, rc)
			}
		}
		return
	}

	key := &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: s.namespace}}
	items, err := s.indexer.Index(NamespaceIndex, key)
	if err != nil {
		return
	}
	for _, m := range items {
		rc := *(m.(*api.ReplicationController))
		if selector.Matches(labels.Set(rc.Labels)) {
			controllers = append(controllers, rc)
		}
	}
	return
}

// GetPodControllers returns a list of replication controllers managing a pod. Returns an error only if no matching controllers are found.
func (s *StoreToReplicationControllerLister) GetPodControllers(pod *api.Pod) (controllers []api.ReplicationController, err error) {
	var selector labels.Selector
	var rc api.ReplicationController

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no controllers found for pod %v because it has no labels", pod.Name)
		return
	}

	key := &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace}}
	items, err := s.Indexer.Index(NamespaceIndex, key)
	if err != nil {
		return
	}

	for _, m := range items {
		rc = *m.(*api.ReplicationController)
		labelSet := labels.Set(rc.Spec.Selector)
		selector = labels.Set(rc.Spec.Selector).AsSelector()

		// If an rc with a nil or empty selector creeps in, it should match nothing, not everything.
		if labelSet.AsSelector().Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		controllers = append(controllers, rc)
	}
	if len(controllers) == 0 {
		err = fmt.Errorf("could not find controller for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StoreToDeploymentLister gives a store List and Exists methods. The store must contain only Deployments.
type StoreToDeploymentLister struct {
	Store
}

// Exists checks if the given deployment exists in the store.
func (s *StoreToDeploymentLister) Exists(deployment *extensions.Deployment) (bool, error) {
	_, exists, err := s.Store.Get(deployment)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToDeploymentLister lists all deployments in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToDeploymentLister) List() (deployments []extensions.Deployment, err error) {
	for _, c := range s.Store.List() {
		deployments = append(deployments, *(c.(*extensions.Deployment)))
	}
	return deployments, nil
}

// GetDeploymentsForReplicaSet returns a list of deployments managing a replica set. Returns an error only if no matching deployments are found.
func (s *StoreToDeploymentLister) GetDeploymentsForReplicaSet(rs *extensions.ReplicaSet) (deployments []extensions.Deployment, err error) {
	var d extensions.Deployment

	if len(rs.Labels) == 0 {
		err = fmt.Errorf("no deployments found for ReplicaSet %v because it has no labels", rs.Name)
		return
	}

	// TODO: MODIFY THIS METHOD so that it checks for the podTemplateSpecHash label
	for _, m := range s.Store.List() {
		d = *m.(*extensions.Deployment)
		if d.Namespace != rs.Namespace {
			continue
		}

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

// StoreToServiceLister makes a Store that has the List method of the client.ServiceInterface
// The Store must contain (only) Services.
type StoreToServiceLister struct {
	Store
}

func (s *StoreToServiceLister) List() (services api.ServiceList, err error) {
	for _, m := range s.Store.List() {
		services.Items = append(services.Items, *(m.(*api.Service)))
	}
	return services, nil
}

// TODO: Move this back to scheduler as a helper function that takes a Store,
// rather than a method of StoreToServiceLister.
func (s *StoreToServiceLister) GetPodServices(pod *api.Pod) (services []api.Service, err error) {
	var selector labels.Selector
	var service api.Service

	for _, m := range s.Store.List() {
		service = *m.(*api.Service)
		// consider only services that are in the same namespace as the pod
		if service.Namespace != pod.Namespace {
			continue
		}
		if service.Spec.Selector == nil {
			// services with nil selectors match nothing, not everything.
			continue
		}
		selector = labels.Set(service.Spec.Selector).AsSelector()
		if selector.Matches(labels.Set(pod.Labels)) {
			services = append(services, service)
		}
	}
	if len(services) == 0 {
		err = fmt.Errorf("could not find service for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
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
		return nil, fmt.Errorf("PersistentVolume '%v' is not in cache", id)
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
		return nil, fmt.Errorf("PersistentVolumeClaim '%s/%s' is not in cache", namespace, id)
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
