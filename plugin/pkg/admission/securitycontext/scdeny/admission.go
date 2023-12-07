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

package scdeny

import (
	"context"
	"fmt"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// PluginName indicates name of admission plugin.
const PluginName = "SecurityContextDeny"

const docLink = "https://k8s.io/docs/reference/access-authn-authz/admission-controllers/#securitycontextdeny"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		if utilfeature.DefaultFeatureGate.Enabled(features.SecurityContextDeny) {
			return NewSecurityContextDeny(), nil
		} else {
			return nil, fmt.Errorf("%s admission controller is an alpha feature, planned to be removed, and requires the SecurityContextDeny feature gate to be enabled, see %s for more information", PluginName, docLink)
		}
	})
}

// Plugin implements admission.Interface.
type Plugin struct {
	*admission.Handler
}

var _ admission.ValidationInterface = &Plugin{}

// NewSecurityContextDeny creates a new instance of the SecurityContextDeny admission controller
func NewSecurityContextDeny() *Plugin {
	// DEPRECATED: SecurityContextDeny will be removed in favor of PodSecurity admission.
	klog.Warningf("%s admission controller is deprecated. "+
		"Please remove this controller from your configuration files and scripts. "+
		"See %s for more information.",
		PluginName, docLink)
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Validate will deny any pod that defines SupplementalGroups, SELinuxOptions, RunAsUser or FSGroup
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if a.GetSubresource() != "" || a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	if pod.Spec.SecurityContext != nil {
		if pod.Spec.SecurityContext.SupplementalGroups != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("pod.Spec.SecurityContext.SupplementalGroups is forbidden"))
		}
		if pod.Spec.SecurityContext.SELinuxOptions != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("pod.Spec.SecurityContext.SELinuxOptions is forbidden"))
		}
		if pod.Spec.SecurityContext.RunAsUser != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("pod.Spec.SecurityContext.RunAsUser is forbidden"))
		}
		if pod.Spec.SecurityContext.FSGroup != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("pod.Spec.SecurityContext.FSGroup is forbidden"))
		}
	}

	for _, v := range pod.Spec.InitContainers {
		if v.SecurityContext != nil {
			if v.SecurityContext.SELinuxOptions != nil {
				return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("SecurityContext.SELinuxOptions is forbidden"))
			}
			if v.SecurityContext.RunAsUser != nil {
				return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("SecurityContext.RunAsUser is forbidden"))
			}
		}
	}

	for _, v := range pod.Spec.Containers {
		if v.SecurityContext != nil {
			if v.SecurityContext.SELinuxOptions != nil {
				return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("SecurityContext.SELinuxOptions is forbidden"))
			}
			if v.SecurityContext.RunAsUser != nil {
				return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("SecurityContext.RunAsUser is forbidden"))
			}
		}
	}
	return nil
}
