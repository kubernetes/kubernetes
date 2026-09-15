/*
Copyright The Kubernetes Authors.

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

package nodetainttimeadded

import (
	"context"
	"fmt"
	"io"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	// PluginName is the name of the plugin.
	PluginName = "NodeTaintTimeAddedDefaulting"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// NewPlugin creates a new NodeTaintTimeAddedDefaulting admission plugin.
// This plugin sets the timeAdded of the node taints which don't have it set.
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
}

var (
	_ = admission.Interface(&Plugin{})
)

var (
	nodeResource = api.Resource("nodes")
)

// Admit sets the timeAdded of the node taints which don't have it set. Taints
// are unique by key and effect, so a taint counts as newly added when there is
// no taint with the same key and effect on the node which is being updated: it
// gets the current time. The taints which already existed keep the timeAdded
// they had, which stops clients that read a node, change its taints and write
// it back from resetting the timestamps of the taints they did not touch. A
// timeAdded provided by the client is never overwritten.
func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	// Our job is just to timestamp node taints.
	if a.GetResource().GroupResource() != nodeResource || a.GetSubresource() != "" {
		return nil
	}

	node, ok := a.GetObject().(*api.Node)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
	}

	var oldNode *api.Node
	if a.GetOperation() == admission.Update {
		oldNode, ok = a.GetOldObject().(*api.Node)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetOldObject()))
		}
	}

	var now *metav1.Time
	for i := range node.Spec.Taints {
		taint := &node.Spec.Taints[i]
		if taint.TimeAdded != nil {
			continue
		}
		if oldTaint := findTaint(oldNode, *taint); oldTaint != nil && oldTaint.TimeAdded != nil {
			taint.TimeAdded = oldTaint.TimeAdded.DeepCopy()
			continue
		}
		if now == nil {
			// Truncate to seconds because sub-second resolution does not
			// survive round-tripping through the API.
			now = &metav1.Time{Time: time.Now().Truncate(time.Second)}
		}
		taint.TimeAdded = now.DeepCopy()
	}
	return nil
}

// findTaint returns the taint of the node which matches the given taint by key
// and effect, if there is one.
func findTaint(node *api.Node, taint api.Taint) *api.Taint {
	if node == nil {
		return nil
	}
	for i := range node.Spec.Taints {
		if node.Spec.Taints[i].MatchTaint(taint) {
			return &node.Spec.Taints[i]
		}
	}
	return nil
}
