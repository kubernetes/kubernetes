/*
Copyright 2022 The Kubernetes Authors.

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

package execprobetimeout

import (
	"context"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// PluginName indicates name of admission plugin.
const PluginName = "ExtendDefaultExecProbeTimeout"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
}

var (
	_ admission.MutationInterface = &Plugin{}
)

// NewPlugin constructs a new instance of the ExtendDefaultExecProbeTimeout admission interface.
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if len(a.GetSubresource()) != 0 || a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	for _, container := range pod.Spec.Containers {
		if container.LivenessProbe != nil {
			if container.LivenessProbe.ProbeHandler.Exec != nil {
				if container.LivenessProbe.TimeoutSeconds == 1 {
					container.LivenessProbe.TimeoutSeconds = 5
				}
			}
		}

		if container.ReadinessProbe != nil {
			if container.ReadinessProbe.ProbeHandler.Exec != nil {
				if container.ReadinessProbe.TimeoutSeconds == 1 {
					container.ReadinessProbe.TimeoutSeconds = 5
				}
			}
		}

		if container.StartupProbe != nil {
			if container.StartupProbe.ProbeHandler.Exec != nil {
				if container.StartupProbe.TimeoutSeconds == 1 {
					container.StartupProbe.TimeoutSeconds = 5
				}
			}
		}
	}

	return nil
}
