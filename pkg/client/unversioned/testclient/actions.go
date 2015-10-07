/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package testclient

import (
	"strings"

	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
)

func NewRootGetAction(resource, name string) GetActionImpl {
	action := GetActionImpl{}
	action.Verb = "get"
	action.Resource = resource
	action.Name = name

	return action
}

func NewGetAction(resource, namespace, name string) GetActionImpl {
	action := GetActionImpl{}
	action.Verb = "get"
	action.Resource = resource
	action.Namespace = namespace
	action.Name = name

	return action
}

func NewRootListAction(resource string, label labels.Selector, field fields.Selector) ListActionImpl {
	action := ListActionImpl{}
	action.Verb = "list"
	action.Resource = resource
	action.ListRestrictions = ListRestrictions{label, field}

	return action
}

func NewListAction(resource, namespace string, label labels.Selector, field fields.Selector) ListActionImpl {
	action := ListActionImpl{}
	action.Verb = "list"
	action.Resource = resource
	action.Namespace = namespace
	action.ListRestrictions = ListRestrictions{label, field}

	return action
}

func NewRootCreateAction(resource string, object runtime.Object) CreateActionImpl {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = resource
	action.Object = object

	return action
}

func NewCreateAction(resource, namespace string, object runtime.Object) CreateActionImpl {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = resource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewRootUpdateAction(resource string, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Object = object

	return action
}

func NewUpdateAction(resource, namespace string, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewRootPatchAction(resource string, object runtime.Object) PatchActionImpl {
	action := PatchActionImpl{}
	action.Verb = "patch"
	action.Resource = resource
	action.Object = object

	return action
}

func NewPatchAction(resource, namespace string, object runtime.Object) PatchActionImpl {
	action := PatchActionImpl{}
	action.Verb = "patch"
	action.Resource = resource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewUpdateSubresourceAction(resource, subresource, namespace string, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Subresource = subresource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewRootDeleteAction(resource, name string) DeleteActionImpl {
	action := DeleteActionImpl{}
	action.Verb = "delete"
	action.Resource = resource
	action.Name = name

	return action
}

func NewDeleteAction(resource, namespace, name string) DeleteActionImpl {
	action := DeleteActionImpl{}
	action.Verb = "delete"
	action.Resource = resource
	action.Namespace = namespace
	action.Name = name

	return action
}

func NewRootWatchAction(resource string, label labels.Selector, field fields.Selector, resourceVersion string) WatchActionImpl {
	action := WatchActionImpl{}
	action.Verb = "watch"
	action.Resource = resource
	action.WatchRestrictions = WatchRestrictions{label, field, resourceVersion}

	return action
}

func NewWatchAction(resource, namespace string, label labels.Selector, field fields.Selector, resourceVersion string) WatchActionImpl {
	action := WatchActionImpl{}
	action.Verb = "watch"
	action.Resource = resource
	action.Namespace = namespace
	action.WatchRestrictions = WatchRestrictions{label, field, resourceVersion}

	return action
}

func NewProxyGetAction(resource, namespace, name, path string, params map[string]string) ProxyGetActionImpl {
	action := ProxyGetActionImpl{}
	action.Verb = "get"
	action.Resource = resource
	action.Namespace = namespace
	action.Name = name
	action.Path = path
	action.Params = params
	return action
}

type ListRestrictions struct {
	Labels labels.Selector
	Fields fields.Selector
}
type WatchRestrictions struct {
	Labels          labels.Selector
	Fields          fields.Selector
	ResourceVersion string
}

type Action interface {
	GetNamespace() string
	GetVerb() string
	GetResource() string
	GetSubresource() string
	Matches(verb, resource string) bool
}

type GenericAction interface {
	Action
	GetValue() interface{}
}

type GetAction interface {
	Action
	GetName() string
}

type ListAction interface {
	Action
	GetListRestrictions() ListRestrictions
}

type CreateAction interface {
	Action
	GetObject() runtime.Object
}

type UpdateAction interface {
	Action
	GetObject() runtime.Object
}

type DeleteAction interface {
	Action
	GetName() string
}

type WatchAction interface {
	Action
	GetWatchRestrictions() WatchRestrictions
}

type ProxyGetAction interface {
	Action
	GetName() string
	GetPath() string
	GetParams() map[string]string
}

type ActionImpl struct {
	Namespace   string
	Verb        string
	Resource    string
	Subresource string
}

func (a ActionImpl) GetNamespace() string {
	return a.Namespace
}
func (a ActionImpl) GetVerb() string {
	return a.Verb
}
func (a ActionImpl) GetResource() string {
	return a.Resource
}
func (a ActionImpl) GetSubresource() string {
	return a.Subresource
}
func (a ActionImpl) Matches(verb, resource string) bool {
	return strings.ToLower(verb) == strings.ToLower(a.Verb) &&
		strings.ToLower(resource) == strings.ToLower(a.Resource)
}

type GenericActionImpl struct {
	ActionImpl
	Value interface{}
}

func (a GenericActionImpl) GetValue() interface{} {
	return a.Value
}

type GetActionImpl struct {
	ActionImpl
	Name string
}

func (a GetActionImpl) GetName() string {
	return a.Name
}

type ListActionImpl struct {
	ActionImpl
	ListRestrictions ListRestrictions
}

func (a ListActionImpl) GetListRestrictions() ListRestrictions {
	return a.ListRestrictions
}

type CreateActionImpl struct {
	ActionImpl
	Object runtime.Object
}

func (a CreateActionImpl) GetObject() runtime.Object {
	return a.Object
}

type UpdateActionImpl struct {
	ActionImpl
	Object runtime.Object
}

func (a UpdateActionImpl) GetObject() runtime.Object {
	return a.Object
}

type PatchActionImpl struct {
	ActionImpl
	Object runtime.Object
}

func (a PatchActionImpl) GetObject() runtime.Object {
	return a.Object
}

type DeleteActionImpl struct {
	ActionImpl
	Name string
}

func (a DeleteActionImpl) GetName() string {
	return a.Name
}

type WatchActionImpl struct {
	ActionImpl
	WatchRestrictions WatchRestrictions
}

func (a WatchActionImpl) GetWatchRestrictions() WatchRestrictions {
	return a.WatchRestrictions
}

type ProxyGetActionImpl struct {
	ActionImpl
	Name   string
	Path   string
	Params map[string]string
}

func (a ProxyGetActionImpl) GetName() string {
	return a.Name
}

func (a ProxyGetActionImpl) GetPath() string {
	return a.Path
}

func (a ProxyGetActionImpl) GetParams() map[string]string {
	return a.Params
}
