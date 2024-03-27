/*
Copyright 2024 The Kubernetes Authors.

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

package disableservicelinks

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

// PluginName indicates name of admission plugin.
const PluginName = "DisableServiceLinks"

// Register is called by the apiserver to register the plugin factory.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newDisableServiceLinks(), nil
	})
}

// newDisableServiceLinks creates a new instance of the DisableServiceLinks admission controller.
func newDisableServiceLinks() *plugin {
	return &plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Make sure we are implementing the interface.
var _ admission.MutationInterface = &plugin{}

type plugin struct {
	*admission.Handler
}

// Admit updates the EnableServiceLinks of a pod and set it to false.
func (p *plugin) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) error {
	op := attributes.GetOperation()

	// noop admission.Update for future support
	if op == admission.Update {
		return nil
	}

	// Ignore all calls to subresources or resources other than pods.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != core.Resource("pods") {
		return nil
	}

	pod, ok := attributes.GetObject().(*core.Pod)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected *core.Pod but got %T", attributes.GetObject()))
	}

	pod.Spec.EnableServiceLinks = ptr.To(false)

	return nil
}
