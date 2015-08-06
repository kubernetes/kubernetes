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

package minion

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/fielderrors"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

// nodeStrategy implements behavior for nodes
type nodeStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Nodes is the default logic that applies when creating and updating Node
// objects.
var Strategy = nodeStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for nodes.
func (nodeStrategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate is false for nodes.
func (nodeStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (nodeStrategy) PrepareForCreate(obj runtime.Object) {
	_ = obj.(*api.Node)
	// Nodes allow *all* fields, including status, to be set on create.
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (nodeStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newNode := obj.(*api.Node)
	oldNode := old.(*api.Node)
	newNode.Status = oldNode.Status
}

// Validate validates a new node.
func (nodeStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	node := obj.(*api.Node)
	return validation.ValidateNode(node)
}

// ValidateUpdate is the default update validation for an end user.
func (nodeStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	errorList := validation.ValidateNode(obj.(*api.Node))
	return append(errorList, validation.ValidateNodeUpdate(old.(*api.Node), obj.(*api.Node))...)
}

func (nodeStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type nodeStatusStrategy struct {
	nodeStrategy
}

var StatusStrategy = nodeStatusStrategy{Strategy}

func (nodeStatusStrategy) PrepareForCreate(obj runtime.Object) {
	_ = obj.(*api.Node)
	// Nodes allow *all* fields, including status, to be set on create.
}

func (nodeStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newNode := obj.(*api.Node)
	oldNode := old.(*api.Node)
	newNode.Spec = oldNode.Spec
}

func (nodeStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateNodeUpdate(old.(*api.Node), obj.(*api.Node))
}

// ResourceGetter is an interface for retrieving resources by ResourceLocation.
type ResourceGetter interface {
	Get(api.Context, string) (runtime.Object, error)
}

// NodeToSelectableFields returns a label set that represents the object.
func NodeToSelectableFields(node *api.Node) fields.Set {
	return fields.Set{
		"metadata.name":      node.Name,
		"spec.unschedulable": fmt.Sprint(node.Spec.Unschedulable),
	}
}

// MatchNode returns a generic matcher for a given label and field selector.
func MatchNode(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			nodeObj, ok := obj.(*api.Node)
			if !ok {
				return nil, nil, fmt.Errorf("not a node")
			}
			return labels.Set(nodeObj.ObjectMeta.Labels), NodeToSelectableFields(nodeObj), nil
		},
	}
}

// ResourceLocation returns an URL and transport which one can use to send traffic for the specified node.
func ResourceLocation(getter ResourceGetter, connection client.ConnectionInfoGetter, ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	name, portReq, valid := util.SplitPort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid node request %q", id))
	}

	nodeObj, err := getter.Get(ctx, name)
	if err != nil {
		return nil, nil, err
	}
	node := nodeObj.(*api.Node)
	hostIP, err := nodeutil.GetNodeHostIP(node)
	if err != nil {
		return nil, nil, err
	}
	host := hostIP.String()

	if portReq == "" || strconv.Itoa(ports.KubeletPort) == portReq {
		scheme, port, transport, err := connection.GetConnectionInfo(host)
		if err != nil {
			return nil, nil, err
		}
		return &url.URL{
				Scheme: scheme,
				Host: net.JoinHostPort(
					host,
					strconv.FormatUint(uint64(port), 10),
				),
			},
			transport,
			nil
	}
	return &url.URL{Host: net.JoinHostPort(host, portReq)}, nil, nil
}
