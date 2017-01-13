/*
Copyright 2016 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/pkg/api/errors"
	"k8s.io/client-go/pkg/api/v1"
	extensionsinternal "k8s.io/client-go/pkg/apis/extensions"
	extensions "k8s.io/client-go/pkg/apis/extensions/v1beta1"
)

//  TODO: generate these classes and methods for all resources of interest using
// a script.  Can use "go generate" once 1.4 is supported by all users.

// Lister makes an Index have the List method.  The Stores must contain only the expected type
// Example:
// s := cache.NewStore()
// lw := cache.ListWatch{Client: c, FieldSelector: sel, Resource: "pods"}
// r := cache.NewReflector(lw, &extensions.Deployment{}, s).Run()
// l := StoreToDeploymentLister{s}
// l.List()

// StoreToDeploymentLister helps list deployments
type StoreToDeploymentLister struct {
	Indexer Indexer
}

func (s *StoreToDeploymentLister) List(selector labels.Selector) (ret []*extensions.Deployment, err error) {
	err = ListAll(s.Indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*extensions.Deployment))
	})
	return ret, err
}

func (s *StoreToDeploymentLister) Deployments(namespace string) storeDeploymentsNamespacer {
	return storeDeploymentsNamespacer{Indexer: s.Indexer, namespace: namespace}
}

type storeDeploymentsNamespacer struct {
	Indexer   Indexer
	namespace string
}

func (s storeDeploymentsNamespacer) List(selector labels.Selector) (ret []*extensions.Deployment, err error) {
	err = ListAllByNamespace(s.Indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(*extensions.Deployment))
	})
	return ret, err
}

func (s storeDeploymentsNamespacer) Get(name string) (*extensions.Deployment, error) {
	obj, exists, err := s.Indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(extensionsinternal.Resource("deployment"), name)
	}
	return obj.(*extensions.Deployment), nil
}

// GetDeploymentsForReplicaSet returns a list of deployments managing a replica set. Returns an error only if no matching deployments are found.
func (s *StoreToDeploymentLister) GetDeploymentsForReplicaSet(rs *extensions.ReplicaSet) (deployments []*extensions.Deployment, err error) {
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
		selector, err := metav1.LabelSelectorAsSelector(d.Spec.Selector)
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

// GetDeploymentsForDeployments returns a list of deployments managing a pod. Returns an error only if no matching deployments are found.
// TODO eliminate shallow copies
func (s *StoreToDeploymentLister) GetDeploymentsForPod(pod *v1.Pod) (deployments []*extensions.Deployment, err error) {
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
		selector, err := metav1.LabelSelectorAsSelector(d.Spec.Selector)
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

// StoreToReplicaSetLister helps list replicasets
type StoreToReplicaSetLister struct {
	Indexer Indexer
}

func (s *StoreToReplicaSetLister) List(selector labels.Selector) (ret []*extensions.ReplicaSet, err error) {
	err = ListAll(s.Indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*extensions.ReplicaSet))
	})
	return ret, err
}

func (s *StoreToReplicaSetLister) ReplicaSets(namespace string) storeReplicaSetsNamespacer {
	return storeReplicaSetsNamespacer{Indexer: s.Indexer, namespace: namespace}
}

type storeReplicaSetsNamespacer struct {
	Indexer   Indexer
	namespace string
}

func (s storeReplicaSetsNamespacer) List(selector labels.Selector) (ret []*extensions.ReplicaSet, err error) {
	err = ListAllByNamespace(s.Indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(*extensions.ReplicaSet))
	})
	return ret, err
}

func (s storeReplicaSetsNamespacer) Get(name string) (*extensions.ReplicaSet, error) {
	obj, exists, err := s.Indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(extensionsinternal.Resource("replicaset"), name)
	}
	return obj.(*extensions.ReplicaSet), nil
}

// GetPodReplicaSets returns a list of ReplicaSets managing a pod. Returns an error only if no matching ReplicaSets are found.
func (s *StoreToReplicaSetLister) GetPodReplicaSets(pod *v1.Pod) (rss []*extensions.ReplicaSet, err error) {
	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no ReplicaSets found for pod %v because it has no labels", pod.Name)
		return
	}

	list, err := s.ReplicaSets(pod.Namespace).List(labels.Everything())
	if err != nil {
		return
	}
	for _, rs := range list {
		if rs.Namespace != pod.Namespace {
			continue
		}
		selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid selector: %v", err)
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
