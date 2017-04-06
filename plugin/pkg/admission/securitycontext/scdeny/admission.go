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
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

const (
	SUPPLEMENTAL_GROUPS = "SupplementalGroups"
	SE_LINUX_OPTION     = "SELinuxOption"
	RUN_AS_USER         = "RunAsUser"
	FS_GROUP            = "FSGroup"
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

// Admit will deny any pod that defines SELinuxOptions, RunAsUser, SupplementalGroups, FSGroup.
func (p *plugin) Admit(a admission.Attributes) (err error) {
	if a.GetSubresource() != "" || a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	if err = checkPodSecurityContext(pod.Spec.SecurityContext); err != nil {
		return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, err)
	}

	for _, container := range pod.Spec.InitContainers {
		if err = checkSecurityContext(container.SecurityContext); err != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, err)
		}
	}

	for _, container := range pod.Spec.Containers {
		if err = checkSecurityContext(container.SecurityContext); err != nil {
			return apierrors.NewForbidden(a.GetResource().GroupResource(), pod.Name, err)
		}
	}

	return
}

func checkPodSecurityContext(podContext *api.PodSecurityContext) error {
	var (
		kind string
		err error
	)
	if podContext != nil {
		if podContext.SupplementalGroups != nil {
			kind = SUPPLEMENTAL_GROUPS
		} else if podContext.SELinuxOptions != nil {
			kind = SE_LINUX_OPTION
		} else if podContext.RunAsUser != nil {
			kind = RUN_AS_USER
		} else if podContext.FSGroup != nil {
			kind = FS_GROUP
		}

		if kind != "" {
			err = fmt.Errorf("pod.Spec.SecurityContext.%s is forbidden", kind)
		}
	}

	return err
}

func checkSecurityContext(context *api.SecurityContext) error {
	var (
		kind string
		err error
	)
	if context != nil {
		if context.SELinuxOptions != nil {
			kind = SE_LINUX_OPTION
		} else if context.RunAsUser != nil {
			kind = RUN_AS_USER
		}

		if kind != "" {
			err = fmt.Errorf("SecurityContext.%s is forbidden", kind)
		}
	}

	return err
}
