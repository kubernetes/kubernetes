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
	"context"
	"fmt"
	"io"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

const (
	// DenyEscalatingExec indicates name of admission plugin.
	// Deprecated, will be removed in v1.18.
	// Use of PodSecurityPolicy or a custom admission plugin to limit creation of pods is recommended instead.
	DenyEscalatingExec = "DenyEscalatingExec"
	// DenyExecOnPrivileged indicates name of admission plugin.
	// Deprecated, will be removed in v1.18.
	// Use of PodSecurityPolicy or a custom admission plugin to limit creation of pods is recommended instead.
	DenyExecOnPrivileged = "DenyExecOnPrivileged"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(DenyEscalatingExec, func(config io.Reader) (admission.Interface, error) {
		klog.Warningf("the %s admission plugin is deprecated and will be removed in v1.18", DenyEscalatingExec)
		klog.Warningf("use of PodSecurityPolicy or a custom admission plugin to limit creation of pods is recommended instead")
		return NewDenyEscalatingExec(), nil
	})

	// This is for legacy support of the DenyExecOnPrivileged admission controller.  Most
	// of the time DenyEscalatingExec should be preferred.
	plugins.Register(DenyExecOnPrivileged, func(config io.Reader) (admission.Interface, error) {
		klog.Warningf("the %s admission plugin is deprecated and will be removed in v1.18", DenyExecOnPrivileged)
		klog.Warningf("use of PodSecurityPolicy or a custom admission plugin to limit creation of pods is recommended instead")
		return NewDenyExecOnPrivileged(), nil
	})
}

// DenyExec is an implementation of admission.Interface which says no to a pod/exec on
// a pod using host based configurations.
type DenyExec struct {
	*admission.Handler
	client kubernetes.Interface

	// these flags control which items will be checked to deny exec/attach
	hostNetwork bool
	hostIPC     bool
	hostPID     bool
	privileged  bool
}

var _ admission.ValidationInterface = &DenyExec{}
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&DenyExec{})

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

// SetExternalKubeClientSet implements the WantsInternalKubeClientSet interface.
func (d *DenyExec) SetExternalKubeClientSet(client kubernetes.Interface) {
	d.client = client
}

// ValidateInitialization implements the InitializationValidator interface.
func (d *DenyExec) ValidateInitialization() error {
	if d.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

// Validate makes an admission decision based on the request attributes
func (d *DenyExec) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	path := a.GetResource().Resource
	if subresource := a.GetSubresource(); subresource != "" {
		path = path + "/" + subresource
	}
	// Only handle exec or attach requests on pods
	if path != "pods/exec" && path != "pods/attach" {
		return nil
	}
	pod, err := d.client.CoreV1().Pods(a.GetNamespace()).Get(context.TODO(), a.GetName(), metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	if d.hostNetwork && pod.Spec.HostNetwork {
		return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a container using host network"))
	}

	if d.hostPID && pod.Spec.HostPID {
		return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a container using host pid"))
	}

	if d.hostIPC && pod.Spec.HostIPC {
		return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a container using host ipc"))
	}

	if d.privileged && isPrivileged(pod) {
		return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a privileged container"))
	}

	return nil
}

// isPrivileged will return true a pod has any privileged containers
func isPrivileged(pod *corev1.Pod) bool {
	var privileged bool
	podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(c *corev1.Container, containerType podutil.ContainerType) bool {
		if c.SecurityContext == nil || c.SecurityContext.Privileged == nil {
			return true
		}
		if *c.SecurityContext.Privileged {
			privileged = true
			return false
		}
		return true
	})
	return privileged
}
