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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	kubelet "k8s.io/kubernetes/pkg/kubelet/types"
)

func init() {
	admission.RegisterPlugin("DenyStaticPodExec", func(config io.Reader) (admission.Interface, error) {
		return NewDenyStaticPodExec(), nil
	})
}

type denyStaticPodExec struct {
	*admission.Handler
	client internalclientset.Interface
}

var _ = kubeapiserveradmission.WantsInternalClientSet(&denyStaticPodExec{})

// NewDenyStaticPodExec is an implementation of admission.Interface which says no to a pod/exec to any
// static pod.  Static pods generally have access to host related information and were set up directly by
// a cluster-admin instead of a daemonset.  They can use container runtime commands instead.
func NewDenyStaticPodExec() admission.Interface {
	return &denyStaticPodExec{
		Handler: admission.NewHandler(admission.Connect),
	}
}

// Admit fulfills the admission interface
func (d *denyStaticPodExec) Admit(a admission.Attributes) (err error) {
	connectRequest, ok := a.GetObject().(*rest.ConnectRequest)
	if !ok {
		return errors.NewBadRequest("a connect request was received, but could not convert the request object.")
	}
	// Only handle exec or attach requests on pods
	if connectRequest.ResourcePath != "pods/exec" && connectRequest.ResourcePath != "pods/attach" {
		return nil
	}
	pod, err := d.client.Core().Pods(a.GetNamespace()).Get(connectRequest.Name, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	if _, isMirrorPod := pod.Annotations[kubelet.ConfigMirrorAnnotationKey]; isMirrorPod {
		return admission.NewForbidden(a, fmt.Errorf("cannot exec into or attach to a static pod"))
	}

	return nil
}

// SetInternalClientSet for initialization
func (d *denyStaticPodExec) SetInternalClientSet(client internalclientset.Interface) {
	d.client = client
}

// Validate to ensure everything is ready
func (d *denyStaticPodExec) Validate() error {
	if d.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
