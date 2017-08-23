/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"path"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func NewRootGetAction(resource schema.GroupVersionResource, name string) GetActionImpl {
	action := GetActionImpl{}
	action.Verb = "get"
	action.Resource = resource
	action.Name = name

	return action
}

func NewGetAction(resource schema.GroupVersionResource, namespace, name string) GetActionImpl {
	action := GetActionImpl{}
	action.Verb = "get"
	action.Resource = resource
	action.Namespace = namespace
	action.Name = name

	return action
}

func NewRootListAction(resource schema.GroupVersionResource, kind schema.GroupVersionKind, opts interface{}) ListActionImpl {
	action := ListActionImpl{}
	action.Verb = "list"
	action.Resource = resource
	action.Kind = kind
	labelSelector, fieldSelector, _ := ExtractFromListOptions(opts)
	action.ListRestrictions = ListRestrictions{labelSelector, fieldSelector}

	return action
}

func NewListAction(resource schema.GroupVersionResource, kind schema.GroupVersionKind, namespace string, opts interface{}) ListActionImpl {
	action := ListActionImpl{}
	action.Verb = "list"
	action.Resource = resource
	action.Kind = kind
	action.Namespace = namespace
	labelSelector, fieldSelector, _ := ExtractFromListOptions(opts)
	action.ListRestrictions = ListRestrictions{labelSelector, fieldSelector}

	return action
}

func NewRootCreateAction(resource schema.GroupVersionResource, object runtime.Object) CreateActionImpl {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = resource
	action.Object = object

	return action
}

func NewCreateAction(resource schema.GroupVersionResource, namespace string, object runtime.Object) CreateActionImpl {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = resource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewRootUpdateAction(resource schema.GroupVersionResource, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Object = object

	return action
}

func NewUpdateAction(resource schema.GroupVersionResource, namespace string, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewRootPatchAction(resource schema.GroupVersionResource, name string, patch []byte) PatchActionImpl {
	action := PatchActionImpl{}
	action.Verb = "patch"
	action.Resource = resource
	action.Name = name
	action.Patch = patch

	return action
}

func NewPatchAction(resource schema.GroupVersionResource, namespace string, name string, patch []byte) PatchActionImpl {
	action := PatchActionImpl{}
	action.Verb = "patch"
	action.Resource = resource
	action.Namespace = namespace
	action.Name = name
	action.Patch = patch

	return action
}

func NewRootPatchSubresourceAction(resource schema.GroupVersionResource, name string, patch []byte, subresources ...string) PatchActionImpl {
	action := PatchActionImpl{}
	action.Verb = "patch"
	action.Resource = resource
	action.Subresource = path.Join(subresources...)
	action.Name = name
	action.Patch = patch

	return action
}

func NewPatchSubresourceAction(resource schema.GroupVersionResource, namespace, name string, patch []byte, subresources ...string) PatchActionImpl {
	action := PatchActionImpl{}
	action.Verb = "patch"
	action.Resource = resource
	action.Subresource = path.Join(subresources...)
	action.Namespace = namespace
	action.Name = name
	action.Patch = patch

	return action
}

func NewRootUpdateSubresourceAction(resource schema.GroupVersionResource, subresource string, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Subresource = subresource
	action.Object = object

	return action
}
func NewUpdateSubresourceAction(resource schema.GroupVersionResource, subresource string, namespace string, object runtime.Object) UpdateActionImpl {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = resource
	action.Subresource = subresource
	action.Namespace = namespace
	action.Object = object

	return action
}

func NewRootDeleteAction(resource schema.GroupVersionResource, name string) DeleteActionImpl {
	action := DeleteActionImpl{}
	action.Verb = "delete"
	action.Resource = resource
	action.Name = name

	return action
}

func NewDeleteAction(resource schema.GroupVersionResource, namespace, name string) DeleteActionImpl {
	action := DeleteActionImpl{}
	action.Verb = "delete"
	action.Resource = resource
	action.Namespace = namespace
	action.Name = name

	return action
}

func NewRootDeleteCollectionAction(resource schema.GroupVersionResource, opts interface{}) DeleteCollectionActionImpl {
	action := DeleteCollectionActionImpl{}
	action.Verb = "delete-collection"
	action.Resource = resource
	labelSelector, fieldSelector, _ := ExtractFromListOptions(opts)
	action.ListRestrictions = ListRestrictions{labelSelector, fieldSelector}

	return action
}

func NewDeleteCollectionAction(resource schema.GroupVersionResource, namespace string, opts interface{}) DeleteCollectionActionImpl {
	action := DeleteCollectionActionImpl{}
	action.Verb = "delete-collection"
	action.Resource = resource
	action.Namespace = namespace
	labelSelector, fieldSelector, _ := ExtractFromListOptions(opts)
	action.ListRestrictions = ListRestrictions{labelSelector, fieldSelector}

	return action
}

func NewRootWatchAction(resource schema.GroupVersionResource, opts interface{}) WatchActionImpl {
	action := WatchActionImpl{}
	action.Verb = "watch"
	action.Resource = resource
	labelSelector, fieldSelector, resourceVersion := ExtractFromListOptions(opts)
	action.WatchRestrictions = WatchRestrictions{labelSelector, fieldSelector, resourceVersion}

	return action
}

func ExtractFromListOptions(opts interface{}) (labelSelector labels.Selector, fieldSelector fields.Selector, resourceVersion string) {
	var err error
	switch t := opts.(type) {
	case metav1.ListOptions:
		labelSelector, err = labels.Parse(t.LabelSelector)
		if err != nil {
			panic(fmt.Errorf("invalid selector %q: %v", t.LabelSelector, err))
		}
		fieldSelector, err = fields.ParseSelector(t.FieldSelector)
		if err != nil {
			panic(fmt.Errorf("invalid selector %q: %v", t.FieldSelector, err))
		}
		resourceVersion = t.ResourceVersion
	default:
		panic(fmt.Errorf("expect a ListOptions %T", opts))
	}
	if labelSelector == nil {
		labelSelector = labels.Everything()
	}
	if fieldSelector == nil {
		fieldSelector = fields.Everything()
	}
	return labelSelector, fieldSelector, resourceVersion
}

func NewWatchAction(resource schema.GroupVersionResource, namespace string, opts interface{}) WatchActionImpl {
	action := WatchActionImpl{}
	action.Verb = "watch"
	action.Resource = resource
	action.Namespace = namespace
	labelSelector, fieldSelector, resourceVersion := ExtractFromListOptions(opts)
	action.WatchRestrictions = WatchRestrictions{labelSelector, fieldSelector, resourceVersion}

	return action
}

func NewProxyGetAction(resource schema.GroupVersionResource, namespace, scheme, name, port, path string, params map[string]string) ProxyGetActionImpl {
	action := ProxyGetActionImpl{}
	action.Verb = "get"
	action.Resource = resource
	action.Namespace = namespace
	action.Scheme = scheme
	action.Name = name
	action.Port = port
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
	GetResource() schema.GroupVersionResource
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

type DeleteCollectionAction interface {
	Action
	GetListRestrictions() ListRestrictions
}

type PatchAction interface {
	Action
	GetName() string
	GetPatch() []byte
}

type WatchAction interface {
	Action
	GetWatchRestrictions() WatchRestrictions
}

type ProxyGetAction interface {
	Action
	GetScheme() string
	GetName() string
	GetPort() string
	GetPath() string
	GetParams() map[string]string
}

type ActionImpl struct {
	Namespace   string
	Verb        string
	Resource    schema.GroupVersionResource
	Subresource string
}

func (a ActionImpl) GetNamespace() string {
	return a.Namespace
}
func (a ActionImpl) GetVerb() string {
	return a.Verb
}
func (a ActionImpl) GetResource() schema.GroupVersionResource {
	return a.Resource
}
func (a ActionImpl) GetSubresource() string {
	return a.Subresource
}
func (a ActionImpl) Matches(verb, resource string) bool {
	return strings.ToLower(verb) == strings.ToLower(a.Verb) &&
		strings.ToLower(resource) == strings.ToLower(a.Resource.Resource)
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
	Kind             schema.GroupVersionKind
	ListRestrictions ListRestrictions
}

func (a ListActionImpl) GetKind() schema.GroupVersionKind {
	return a.Kind
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
	Name  string
	Patch []byte
}

func (a PatchActionImpl) GetName() string {
	return a.Name
}

func (a PatchActionImpl) GetPatch() []byte {
	return a.Patch
}

type DeleteActionImpl struct {
	ActionImpl
	Name string
}

func (a DeleteActionImpl) GetName() string {
	return a.Name
}

type DeleteCollectionActionImpl struct {
	ActionImpl
	ListRestrictions ListRestrictions
}

func (a DeleteCollectionActionImpl) GetListRestrictions() ListRestrictions {
	return a.ListRestrictions
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
	Scheme string
	Name   string
	Port   string
	Path   string
	Params map[string]string
}

func (a ProxyGetActionImpl) GetScheme() string {
	return a.Scheme
}

func (a ProxyGetActionImpl) GetName() string {
	return a.Name
}

func (a ProxyGetActionImpl) GetPort() string {
	return a.Port
}

func (a ProxyGetActionImpl) GetPath() string {
	return a.Path
}

func (a ProxyGetActionImpl) GetParams() map[string]string {
	return a.Params
}
