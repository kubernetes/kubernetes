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

package node

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	pkgstorage "k8s.io/kubernetes/pkg/storage"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/validation/field"
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
func (nodeStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	_ = obj.(*api.Node)
	// Nodes allow *all* fields, including status, to be set on create.
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (nodeStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newNode := obj.(*api.Node)
	oldNode := old.(*api.Node)
	newNode.Status = oldNode.Status
}

// Validate validates a new node.
func (nodeStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	node := obj.(*api.Node)
	return validation.ValidateNode(node)
}

// Canonicalize normalizes the object after validation.
func (nodeStrategy) Canonicalize(obj runtime.Object) {
}

// ValidateUpdate is the default update validation for an end user.
func (nodeStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	errorList := validation.ValidateNode(obj.(*api.Node))
	return append(errorList, validation.ValidateNodeUpdate(obj.(*api.Node), old.(*api.Node))...)
}

func (nodeStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func (ns nodeStrategy) Export(ctx api.Context, obj runtime.Object, exact bool) error {
	n, ok := obj.(*api.Node)
	if !ok {
		// unexpected programmer error
		return fmt.Errorf("unexpected object: %v", obj)
	}
	ns.PrepareForCreate(ctx, obj)
	if exact {
		return nil
	}
	// Nodes are the only resources that allow direct status edits, therefore
	// we clear that without exact so that the node value can be reused.
	n.Status = api.NodeStatus{}
	return nil
}

type nodeStatusStrategy struct {
	nodeStrategy
}

var StatusStrategy = nodeStatusStrategy{Strategy}

func (nodeStatusStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	_ = obj.(*api.Node)
	// Nodes allow *all* fields, including status, to be set on create.
}

func (nodeStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newNode := obj.(*api.Node)
	oldNode := old.(*api.Node)
	newNode.Spec = oldNode.Spec
}

func (nodeStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateNodeUpdate(obj.(*api.Node), old.(*api.Node))
}

// Canonicalize normalizes the object after validation.
func (nodeStatusStrategy) Canonicalize(obj runtime.Object) {
}

// ResourceGetter is an interface for retrieving resources by ResourceLocation.
type ResourceGetter interface {
	Get(api.Context, string) (runtime.Object, error)
}

// NodeToSelectableFields returns a field set that represents the object.
func NodeToSelectableFields(node *api.Node) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(node.ObjectMeta, false)
	specificFieldsSet := fields.Set{
		"spec.unschedulable": fmt.Sprint(node.Spec.Unschedulable),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
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
		IndexFields: []string{"metadata.name"},
	}
}

func NodeNameTriggerFunc(obj runtime.Object) []pkgstorage.MatchValue {
	node := obj.(*api.Node)
	result := pkgstorage.MatchValue{IndexName: "metadata.name", Value: node.ObjectMeta.Name}
	return []pkgstorage.MatchValue{result}
}

// ResourceLocation returns an URL and transport which one can use to send traffic for the specified node.
func ResourceLocation(getter ResourceGetter, connection client.ConnectionInfoGetter, proxyTransport http.RoundTripper, ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	schemeReq, name, portReq, valid := utilnet.SplitSchemeNamePort(id)
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

	// We check if we want to get a default Kubelet's transport. It happens if either:
	// - no port is specified in request (Kubelet's port is default),
	// - we're using Port stored as a DaemonEndpoint and requested port is a Kubelet's port stored in the DaemonEndpoint,
	// - there's no information in the API about DaemonEnpoint (legacy cluster) and requested port is equal to ports.KubeletPort (cluster-wide config)
	kubeletPort := node.Status.DaemonEndpoints.KubeletEndpoint.Port
	if kubeletPort == 0 {
		kubeletPort = ports.KubeletPort
	}
	if portReq == "" || strconv.Itoa(int(kubeletPort)) == portReq {
		scheme, port, kubeletTransport, err := connection.GetConnectionInfo(ctx, node.Name)
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
			kubeletTransport,
			nil
	}
	return &url.URL{Scheme: schemeReq, Host: net.JoinHostPort(host, portReq)}, proxyTransport, nil
}
