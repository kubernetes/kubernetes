/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package denyprivileged

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func init() {
	admission.RegisterPlugin("DenyExecOnPrivileged", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewDenyExecOnPrivileged(client), nil
	})
}

// denyExecOnPrivileged is an implementation of admission.Interface which says no to a pod/exec on
// a privileged pod
type denyExecOnPrivileged struct {
	*admission.Handler
	client client.Interface
}

func (d *denyExecOnPrivileged) Admit(a admission.Attributes) (err error) {
	connectRequest, ok := a.GetObject().(*rest.ConnectRequest)
	if !ok {
		return errors.NewBadRequest("a connect request was received, but could not convert the request object.")
	}
	// Only handle exec requests on pods
	if connectRequest.ResourcePath != "pods/exec" {
		return nil
	}
	pod, err := d.client.Pods(a.GetNamespace()).Get(connectRequest.Name)
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if isPrivileged(pod) {
		return admission.NewForbidden(a, fmt.Errorf("Cannot exec into a privileged container"))
	}
	return nil
}

// isPrivileged will return true a pod has any privileged containers
func isPrivileged(pod *api.Pod) bool {
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext == nil {
			continue
		}
		if *c.SecurityContext.Privileged {
			return true
		}
	}
	return false
}

// NewDenyExecOnPrivileged creates a new admission controller that denies an exec operation on a privileged pod
func NewDenyExecOnPrivileged(client client.Interface) admission.Interface {
	return &denyExecOnPrivileged{
		Handler: admission.NewHandler(admission.Connect),
		client:  client,
	}
}
