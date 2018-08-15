/*
Copyright 2015 The Kubernetes Authors.

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

package exec

import (
	"fmt"
	"io"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

const (
	// DenyEscalatingExec indicates name of admission plugin.
	DenyEscalatingExec = "DenyEscalatingExec"
	// DenyExecOnPrivileged indicates name of admission plugin.
	// Deprecated, should use DenyEscalatingExec instead.
	DenyExecOnPrivileged = "DenyExecOnPrivileged"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(DenyEscalatingExec, func(config io.Reader) (admission.Interface, error) {
		return NewDenyEscalatingExec(), nil
	})

	// This is for legacy support of the DenyExecOnPrivileged admission controller.  Most
	// of the time DenyEscalatingExec should be preferred.
	plugins.Register(DenyExecOnPrivileged, func(config io.Reader) (admission.Interface, error) {
		return NewDenyExecOnPrivileged(), nil
	})
}

// DenyExec is an implementation of admission.Interface which says no to a pod/exec on
// a pod using host based configurations.
type DenyExec struct {
	*admission.Handler
	client internalclientset.Interface

	// these flags control which items will be checked to deny exec/attach
	hostNetwork bool
	hostIPC     bool
	hostPID     bool
	privileged  bool
}

var _ admission.ValidationInterface = &DenyExec{}

var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&DenyExec{})

// NewDenyEscalatingExec creates a new admission controller that denies an exec operation on a pod
// using host based configurations.
func NewDenyEscalatingExec() *DenyExec {
	return &DenyExec{
		Handler:     admission.NewHandler(admission.Connect),
		hostNetwork: true,
		hostIPC:     true,
		hostPID:     true,
		privileged:  true,
	}
}

// NewDenyExecOnPrivileged creates a new admission controller that is only checking the privileged
// option. This is for legacy support of the DenyExecOnPrivileged admission controller.
// Most of the time NewDenyEscalatingExec should be preferred.
func NewDenyExecOnPrivileged() *DenyExec {
	return &DenyExec{
		Handler:     admission.NewHandler(admission.Connect),
		hostNetwork: false,
		hostIPC:     false,
		hostPID:     false,
		privileged:  true,
	}
}

// Validate makes an admission decision based on the request attributes
func (d *DenyExec) Validate(a admission.Attributes) (err error) {
	path := a.GetResource().Resource
	if subresource := a.GetSubresource(); subresource != "" {
		path = path + "/" + subresource
	}
	// Only handle exec or attach requests on pods
	if path != "pods/exec" && path != "pods/attach" {
		return nil
	}
	pod, err := d.client.Core().Pods(a.GetNamespace()).Get(a.GetName(), metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	if pod.Spec.SecurityContext != nil {
		securityContext := pod.Spec.SecurityContext
		if d.hostNetwork && securityContext.HostNetwork {
			return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a container using host network"))
		}

		if d.hostPID && securityContext.HostPID {
			return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a container using host pid"))
		}

		if d.hostIPC && securityContext.HostIPC {
			return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a container using host ipc"))
		}
	}

	if d.privileged && isPrivileged(pod) {
		return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a privileged container"))
	}

	return nil
}

// isPrivileged will return true a pod has any privileged containers
func isPrivileged(pod *api.Pod) bool {
	for _, c := range pod.Spec.InitContainers {
		if c.SecurityContext == nil || c.SecurityContext.Privileged == nil {
			continue
		}
		if *c.SecurityContext.Privileged {
			return true
		}
	}
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext == nil || c.SecurityContext.Privileged == nil {
			continue
		}
		if *c.SecurityContext.Privileged {
			return true
		}
	}
	return false
}

// SetInternalKubeClientSet implements the WantsInternalKubeClientSet interface.
func (d *DenyExec) SetInternalKubeClientSet(client internalclientset.Interface) {
	d.client = client
}

// ValidateInitialization implements the InitializationValidator interface.
func (d *DenyExec) ValidateInitialization() error {
	if d.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
