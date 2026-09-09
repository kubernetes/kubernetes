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

package admit

import (
	"context"
	"io"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/klog/v2"
)

// PluginName indicates name of admission plugin.
const PluginName = "AlwaysAdmit"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewAlwaysAdmit(), nil
	})
}

// alwaysAdmit is an implementation of admission.Interface which always says yes to an admit request.
type alwaysAdmit struct{}

var _ admission.MutationInterface = alwaysAdmit{}
var _ admission.ValidationInterface = alwaysAdmit{}

// Admit makes an admission decision based on the request attributes
func (alwaysAdmit) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	return nil
}

// Validate makes an admission decision based on the request attributes.  It is NOT allowed to mutate.
func (alwaysAdmit) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	return nil
}

// Handles returns true if this admission controller can handle the given operation
// where operation can be one of CREATE, UPDATE, DELETE, or CONNECT
func (alwaysAdmit) Handles(operation admission.Operation) bool {
	return true
}

// NewAlwaysAdmit creates a new always admit admission handler
func NewAlwaysAdmit() admission.Interface {
	// DEPRECATED: AlwaysAdmit admit all admission request, it is no use.
	klog.Warningf("%s admission controller is deprecated. "+
		"Please remove this controller from your configuration files and scripts.", PluginName)
	return new(alwaysAdmit)
}
