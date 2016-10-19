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
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/labels"
)

//  TODO: generate these classes and methods for all resources of interest using
// a script.  Can use "go generate" once 1.4 is supported by all users.

// Lister makes an Index have the List method.  The Stores must contain only the expected type
// Example:
// s := cache.NewStore()
// lw := cache.ListWatch{Client: c, FieldSelector: sel, Resource: "pods"}
// r := cache.NewReflector(lw, &rbac.ClusterRole{}, s).Run()
// l := clusterRoleLister{s}
// l.List()

func NewClusterRoleLister(indexer Indexer) ClusterRoleLister {
	return &clusterRoleLister{indexer: indexer}
}
func NewClusterRoleBindingLister(indexer Indexer) ClusterRoleBindingLister {
	return &clusterRoleBindingLister{indexer: indexer}
}
func NewRoleLister(indexer Indexer) RoleLister {
	return &roleLister{indexer: indexer}
}
func NewRoleBindingLister(indexer Indexer) RoleBindingLister {
	return &roleBindingLister{indexer: indexer}
}

// these interfaces are used by the rbac authorizer
type authorizerClusterRoleGetter interface {
	GetClusterRole(name string) (*rbac.ClusterRole, error)
}

type authorizerClusterRoleBindingLister interface {
	ListClusterRoleBindings() ([]*rbac.ClusterRoleBinding, error)
}

type authorizerRoleGetter interface {
	GetRole(namespace, name string) (*rbac.Role, error)
}

type authorizerRoleBindingLister interface {
	ListRoleBindings(namespace string) ([]*rbac.RoleBinding, error)
}

type ClusterRoleLister interface {
	authorizerClusterRoleGetter
	List(selector labels.Selector) (ret []*rbac.ClusterRole, err error)
	Get(name string) (*rbac.ClusterRole, error)
}

type clusterRoleLister struct {
	indexer Indexer
}

func (s *clusterRoleLister) List(selector labels.Selector) (ret []*rbac.ClusterRole, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*rbac.ClusterRole))
	})
	return ret, err
}

func (s clusterRoleLister) Get(name string) (*rbac.ClusterRole, error) {
	obj, exists, err := s.indexer.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(rbac.Resource("clusterrole"), name)
	}
	return obj.(*rbac.ClusterRole), nil
}

func (s clusterRoleLister) GetClusterRole(name string) (*rbac.ClusterRole, error) {
	return s.Get(name)
}

type ClusterRoleBindingLister interface {
	authorizerClusterRoleBindingLister
	List(selector labels.Selector) (ret []*rbac.ClusterRoleBinding, err error)
	Get(name string) (*rbac.ClusterRoleBinding, error)
}

type clusterRoleBindingLister struct {
	indexer Indexer
}

func (s *clusterRoleBindingLister) List(selector labels.Selector) (ret []*rbac.ClusterRoleBinding, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*rbac.ClusterRoleBinding))
	})
	return ret, err
}

func (s clusterRoleBindingLister) Get(name string) (*rbac.ClusterRoleBinding, error) {
	obj, exists, err := s.indexer.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(rbac.Resource("clusterrolebinding"), name)
	}
	return obj.(*rbac.ClusterRoleBinding), nil
}

func (s clusterRoleBindingLister) ListClusterRoleBindings() ([]*rbac.ClusterRoleBinding, error) {
	return s.List(labels.Everything())
}

type RoleLister interface {
	authorizerRoleGetter
	List(selector labels.Selector) (ret []*rbac.Role, err error)
	Roles(namespace string) RoleNamespaceLister
}

type RoleNamespaceLister interface {
	List(selector labels.Selector) (ret []*rbac.Role, err error)
	Get(name string) (*rbac.Role, error)
}

type roleLister struct {
	indexer Indexer
}

func (s *roleLister) List(selector labels.Selector) (ret []*rbac.Role, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*rbac.Role))
	})
	return ret, err
}

func (s *roleLister) Roles(namespace string) RoleNamespaceLister {
	return roleNamespaceLister{indexer: s.indexer, namespace: namespace}
}

func (s roleLister) GetRole(namespace, name string) (*rbac.Role, error) {
	return s.Roles(namespace).Get(name)
}

type roleNamespaceLister struct {
	indexer   Indexer
	namespace string
}

func (s roleNamespaceLister) List(selector labels.Selector) (ret []*rbac.Role, err error) {
	err = ListAllByNamespace(s.indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(*rbac.Role))
	})
	return ret, err
}

func (s roleNamespaceLister) Get(name string) (*rbac.Role, error) {
	obj, exists, err := s.indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(rbac.Resource("role"), name)
	}
	return obj.(*rbac.Role), nil
}

type RoleBindingLister interface {
	authorizerRoleBindingLister
	List(selector labels.Selector) (ret []*rbac.RoleBinding, err error)
	RoleBindings(namespace string) RoleBindingNamespaceLister
}

type RoleBindingNamespaceLister interface {
	List(selector labels.Selector) (ret []*rbac.RoleBinding, err error)
	Get(name string) (*rbac.RoleBinding, error)
}

type roleBindingLister struct {
	indexer Indexer
}

func (s *roleBindingLister) List(selector labels.Selector) (ret []*rbac.RoleBinding, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*rbac.RoleBinding))
	})
	return ret, err
}

func (s *roleBindingLister) RoleBindings(namespace string) RoleBindingNamespaceLister {
	return roleBindingNamespaceLister{indexer: s.indexer, namespace: namespace}
}

func (s roleBindingLister) ListRoleBindings(namespace string) ([]*rbac.RoleBinding, error) {
	return s.RoleBindings(namespace).List(labels.Everything())
}

type roleBindingNamespaceLister struct {
	indexer   Indexer
	namespace string
}

func (s roleBindingNamespaceLister) List(selector labels.Selector) (ret []*rbac.RoleBinding, err error) {
	err = ListAllByNamespace(s.indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(*rbac.RoleBinding))
	})
	return ret, err
}

func (s roleBindingNamespaceLister) Get(name string) (*rbac.RoleBinding, error) {
	obj, exists, err := s.indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(rbac.Resource("rolebinding"), name)
	}
	return obj.(*rbac.RoleBinding), nil
}
