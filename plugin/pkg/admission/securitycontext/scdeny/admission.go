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
	"fmt"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

func init() {
	admission.RegisterPlugin("SecurityContextDeny", func(config io.Reader) (admission.Interface, error) {
		return NewSecurityContextDeny(), nil
	})
}

type plugin struct {
	*admission.Handler
}

// NewSecurityContextDeny creates a new instance of the SecurityContextDeny admission controller
func NewSecurityContextDeny() admission.Interface {
	return &plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Admit will deny any pod that defines SELinuxOptions or RunAsUser.
func (p *plugin) Admit(a admission.Attributes) (err error) {
	if a.GetSubresource() != "" || a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SupplementalGroups != nil {
		return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("SecurityContext.SupplementalGroups is forbidden"))
	}
	if pod.Spec.SecurityContext != nil {
		if pod.Spec.SecurityContext.SELinuxOptions != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("pod.Spec.SecurityContext.SELinuxOptions is forbidden"))
		}
		if pod.Spec.SecurityContext.RunAsUser != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("pod.Spec.SecurityContext.RunAsUser is forbidden"))
		}
	}

	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.FSGroup != nil {
		return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, fmt.Errorf("SecurityContext.FSGroup is forbidden"))
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
