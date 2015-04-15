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

package securitycontext

import (
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	"fmt"
	scapi "github.com/GoogleCloudPlatform/kubernetes/pkg/securitycontext"
	"github.com/golang/glog"
)

func init() {
	admission.RegisterPlugin("SecurityContext", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewSecurityContext(client), nil
	})
}

type plugin struct {
	client client.Interface
}

func NewSecurityContext(client client.Interface) admission.Interface {
	return &plugin{client}
}

func (p *plugin) Admit(a admission.Attributes) (err error) {
	if a.GetOperation() == "DELETE" {
		return nil
	}
	if a.GetResource() != string(api.ResourcePods) {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	scp := p.getSecurityContextProvider(pod)
	if scp == nil {
		return nil
	}

	errs := scp.ValidateSecurityContext(pod)

	//disable enforcement doesn't return errors so this won't fail even if policy is broken
	if len(errs) > 0 {
		glog.Warningf("Initial validation of pod %s/%s failed, applying defaults. Broken policy: %v", pod.Namespace, pod.Name, errs)
	}

	//always apply defaults
	scp.ApplySecurityContext(pod)

	//revalidate, if policy is still broken then do not admit
	if errs := scp.ValidateSecurityContext(pod); len(errs) > 0 {
		msg := fmt.Sprintf("Validation failed even after default security context was applied for pod %s/%s.  Broken policy: %v", pod.Namespace, pod.Name, errs)
		return apierrors.NewForbidden(a.GetResource(), pod.Name, fmt.Errorf(msg))
	}

	return nil
}

func (q *plugin) getSecurityContextProvider(pod *api.Pod) scapi.SecurityContextProvider {
	//TODO this should be retrieved from the service account
	//	return scapi.NewPermitSecurityContextProvider()
	return scapi.NewRestrictSecurityContextProvider()
}
