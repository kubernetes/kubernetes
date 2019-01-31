/*
Copyright 2019 The Kubernetes Authors.

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

package nodetaint

import (
	"fmt"
	"io"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// PluginName is the name of the plugin.
	PluginName = "TaintNodesByCondition"
	// TaintNodeNotReady is the not-ready label as specified in the API.
	TaintNodeNotReady = "node.kubernetes.io/not-ready"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// NewPlugin creates a new NodeTaint admission plugin.
// This plugin identifies requests from nodes
func NewPlugin() *Plugin {
	return &Plugin{
		Handler:  admission.NewHandler(admission.Create),
		features: utilfeature.DefaultFeatureGate,
	}
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
	// allows overriding for testing
	features utilfeature.FeatureGate
}

var (
	_ = admission.Interface(&Plugin{})
)

var (
	nodeResource = api.Resource("nodes")
)

// Admit is the main function that checks node identity and adds taints as needed.
func (p *Plugin) Admit(a admission.Attributes) error {
	// If TaintNodesByCondition is not enabled, we don't need to do anything.
	if !p.features.Enabled(features.TaintNodesByCondition) {
		return nil
	}

	// Our job is just to taint nodes.
	if a.GetResource().GroupResource() != nodeResource || a.GetSubresource() != "" {
		return nil
	}

	node, ok := a.GetObject().(*api.Node)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
	}

	// Taint node with NotReady taint at creation if TaintNodesByCondition is
	// enabled. This is needed to make sure that nodes are added to the cluster
	// with the NotReady taint. Otherwise, a new node may receive the taint with
	// some delay causing pods to be scheduled on a not-ready node.
	// Node controller will remove the taint when the node becomes ready.
	addNotReadyTaint(node)
	return nil
}

func addNotReadyTaint(node *api.Node) {
	notReadyTaint := api.Taint{
		Key:    TaintNodeNotReady,
		Effect: api.TaintEffectNoSchedule,
	}
	for _, taint := range node.Spec.Taints {
		if taint.MatchTaint(notReadyTaint) {
			// the taint already exists.
			return
		}
	}
	node.Spec.Taints = append(node.Spec.Taints, notReadyTaint)
}
