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
	"k8s.io/kubernetes/pkg/apis/experimental"
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
	Store
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
	for _, m := range s.Store.List() {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			pods = append(pods, pod)
		}
	}
	return pods, nil
}

// Pods is taking baby steps to be more like the api in pkg/client
func (s *StoreToPodLister) Pods(namespace string) storePodsNamespacer {
	return storePodsNamespacer{s.Store, namespace}
}

type storePodsNamespacer struct {
	store     Store
	namespace string
}

// Please note that selector is filtering among the pods that have gotten into
// the store; there may have been some filtering that already happened before
// that.
func (s storePodsNamespacer) List(selector labels.Selector) (pods api.PodList, err error) {
	list := api.PodList{}
	for _, m := range s.store.List() {
		pod := m.(*api.Pod)
		if s.namespace == api.NamespaceAll || s.namespace == pod.Namespace {
			if selector.Matches(labels.Set(pod.Labels)) {
				list.Items = append(list.Items, *pod)
			}
		}
	}
	return list, nil
}

// Exists returns true if a pod matching the namespace/name of the given pod exists in the store.
func (s *StoreToPodLister) Exists(pod *api.Pod) (bool, error) {
	_, exists, err := s.Store.Get(pod)
	if err != nil {
		return false, err
	}
	return exists, nil
}

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
func (s *StoreToNodeLister) NodeCondition(conditionType api.NodeConditionType, conditionStatus api.ConditionStatus) storeToNodeConditionLister {
	// TODO: Move this filtering server side. Currently our selectors don't facilitate searching through a list so we
	// have the reflector filter out the Unschedulable field and sift through node conditions in the lister.
	return storeToNodeConditionLister{s.Store, conditionType, conditionStatus}
}

// storeToNodeConditionLister filters and returns nodes matching the given type and status from the store.
type storeToNodeConditionLister struct {
	store           Store
	conditionType   api.NodeConditionType
	conditionStatus api.ConditionStatus
}

// List returns a list of nodes that match the condition type/status in the storeToNodeConditionLister.
func (s storeToNodeConditionLister) List() (nodes api.NodeList, err error) {
	for _, m := range s.store.List() {
		node := *m.(*api.Node)

		// We currently only use a conditionType of "Ready". If the kubelet doesn't
		// periodically report the status of a node, the nodecontroller sets its
		// ConditionStatus to "Unknown". If the kubelet thinks a node is unhealthy
		// it can (in theory) set its ConditionStatus to "False".
		var nodeCondition *api.NodeCondition

		// Get the last condition of the required type
		for _, cond := range node.Status.Conditions {
			if cond.Type == s.conditionType {
				nodeCondition = &cond
			} else {
				glog.V(4).Infof("Ignoring condition type %v for node %v", cond.Type, node.Name)
			}
		}

		// Check that the condition has the required status
		if nodeCondition != nil {
			if nodeCondition.Status == s.conditionStatus {
				nodes.Items = append(nodes.Items, node)
			} else {
				glog.V(4).Infof("Ignoring node %v with condition status %v", node.Name, nodeCondition.Status)
			}
		} else {
			glog.V(2).Infof("Node %s doesn't have conditions of type %v", node.Name, s.conditionType)
		}
	}
	return
}

// TODO Move this back to scheduler as a helper function that takes a Store,
// rather than a method of StoreToNodeLister.
// GetNodeInfo returns cached data for the node 'id'.
func (s *StoreToNodeLister) GetNodeInfo(id string) (*api.Node, error) {
	node, exists, err := s.Get(&api.Node{ObjectMeta: api.ObjectMeta{Name: id}})

	if err != nil {
		return nil, fmt.Errorf("error retrieving node '%v' from cache: %v", id, err)
	}

	if !exists {
		return nil, fmt.Errorf("node '%v' is not in cache", id)
	}

	return node.(*api.Node), nil
}

// StoreToReplicationControllerLister gives a store List and Exists methods. The store must contain only ReplicationControllers.
type StoreToReplicationControllerLister struct {
	Store
}

// Exists checks if the given rc exists in the store.
func (s *StoreToReplicationControllerLister) Exists(controller *api.ReplicationController) (bool, error) {
	_, exists, err := s.Store.Get(controller)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToReplicationControllerLister lists all controllers in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToReplicationControllerLister) List() (controllers []api.ReplicationController, err error) {
	for _, c := range s.Store.List() {
		controllers = append(controllers, *(c.(*api.ReplicationController)))
	}
	return controllers, nil
}

// GetPodControllers returns a list of replication controllers managing a pod. Returns an error only if no matching controllers are found.
func (s *StoreToReplicationControllerLister) GetPodControllers(pod *api.Pod) (controllers []api.ReplicationController, err error) {
	var selector labels.Selector
	var rc api.ReplicationController

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("No controllers found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		rc = *m.(*api.ReplicationController)
		if rc.Namespace != pod.Namespace {
			continue
		}
		labelSet := labels.Set(rc.Spec.Selector)
		selector = labels.Set(rc.Spec.Selector).AsSelector()

		// If an rc with a nil or empty selector creeps in, it should match nothing, not everything.
		if labelSet.AsSelector().Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		controllers = append(controllers, rc)
	}
	if len(controllers) == 0 {
		err = fmt.Errorf("Could not find daemon set for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StoreToDaemonSetLister gives a store List and Exists methods. The store must contain only DaemonSets.
type StoreToDaemonSetLister struct {
	Store
}

// Exists checks if the given daemon set exists in the store.
func (s *StoreToDaemonSetLister) Exists(ds *experimental.DaemonSet) (bool, error) {
	_, exists, err := s.Store.Get(ds)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// List lists all daemon sets in the store.
// TODO: converge on the interface in pkg/client
func (s *StoreToDaemonSetLister) List() (dss []experimental.DaemonSet, err error) {
	for _, c := range s.Store.List() {
		dss = append(dss, *(c.(*experimental.DaemonSet)))
	}
	return dss, nil
}

// GetPodDaemonSets returns a list of daemon sets managing a pod.
// Returns an error if and only if no matching daemon sets are found.
func (s *StoreToDaemonSetLister) GetPodDaemonSets(pod *api.Pod) (daemonSets []experimental.DaemonSet, err error) {
	var selector labels.Selector
	var daemonSet experimental.DaemonSet

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("No daemon sets found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		daemonSet = *m.(*experimental.DaemonSet)
		if daemonSet.Namespace != pod.Namespace {
			continue
		}
		selector = labels.Set(daemonSet.Spec.Selector).AsSelector()

		// If a daemonSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		daemonSets = append(daemonSets, daemonSet)
	}
	if len(daemonSets) == 0 {
		err = fmt.Errorf("Could not find daemon set for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
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
		err = fmt.Errorf("Could not find service for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
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
	err = fmt.Errorf("Could not find endpoints for service: %v", svc.Name)
	return
}

// StoreToJobLister gives a store List and Exists methods. The store must contain only Jobs.
type StoreToJobLister struct {
	Store
}

// Exists checks if the given job exists in the store.
func (s *StoreToJobLister) Exists(job *experimental.Job) (bool, error) {
	_, exists, err := s.Store.Get(job)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// StoreToJobLister lists all jobs in the store.
func (s *StoreToJobLister) List() (jobs []experimental.Job, err error) {
	for _, c := range s.Store.List() {
		jobs = append(jobs, *(c.(*experimental.Job)))
	}
	return jobs, nil
}

// GetPodControllers returns a list of jobs managing a pod. Returns an error only if no matching jobs are found.
func (s *StoreToJobLister) GetPodJobs(pod *api.Pod) (jobs []experimental.Job, err error) {
	var selector labels.Selector
	var job experimental.Job

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("No jobs found for pod %v because it has no labels", pod.Name)
		return
	}

	for _, m := range s.Store.List() {
		job = *m.(*experimental.Job)
		if job.Namespace != pod.Namespace {
			continue
		}
		labelSet := labels.Set(job.Spec.Selector)
		selector = labels.Set(job.Spec.Selector).AsSelector()

		// Job with a nil or empty selector match nothing
		if labelSet.AsSelector().Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		jobs = append(jobs, job)
	}
	if len(jobs) == 0 {
		err = fmt.Errorf("Could not find jobs for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// StoreToNetworksLister makes a Store that lists networks
type StoreToNetworksLister struct {
	Store
}

// List lists all networks in the store
func (s *StoreToNetworksLister) List() (networks api.NetworkList, err error) {
	for _, m := range s.Store.List() {
		networks.Items = append(networks.Items, *(m.(*api.Network)))
	}
	return networks, nil
}

// GetNamespaceNetwork returns the network of a namespace
func (s *StoreToNetworksLister) GetNamespaceNetwork(ns *api.Namespace) *api.Network {
	var net *api.Network

	if ns.Spec.Network != "" {
		for _, m := range s.Store.List() {
			n := m.(*api.Network)
			if ns.Spec.Network == n.Name {
				net = n
				break
			}
		}
	}

	return net
}
