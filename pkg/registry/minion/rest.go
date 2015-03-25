/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
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
	// Nodes allow *all* fields, including status, to be set.
}

// Validate validates a new node.
func (nodeStrategy) Validate(obj runtime.Object) fielderrors.ValidationErrorList {
	node := obj.(*api.Node)
	return validation.ValidateMinion(node)
}

// ValidateUpdate is the default update validation for an end user.
func (nodeStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateMinionUpdate(old.(*api.Node), obj.(*api.Node))
}

// ResourceGetter is an interface for retrieving resources by ResourceLocation.
type ResourceGetter interface {
	Get(api.Context, string) (runtime.Object, error)
}

// MatchNode returns a generic matcher for a given label and field selector.
func MatchNode(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		nodeObj, ok := obj.(*api.Node)
		if !ok {
			return false, fmt.Errorf("not a node")
		}
		// TODO: Add support for filtering based on field, once NodeStatus is defined.
		return label.Matches(labels.Set(nodeObj.Labels)), nil
	})
}

// ResourceLocation returns a URL to which one can send traffic for the specified node.
func ResourceLocation(getter ResourceGetter, connection client.ConnectionInfoGetter, ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	nodeObj, err := getter.Get(ctx, id)
	if err != nil {
		return nil, nil, err
	}
	node := nodeObj.(*api.Node)
	host := node.Name

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
