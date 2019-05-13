/*
Copyright 2017 The Kubernetes Authors.

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

package apiclient

import (
	"net"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// InitDryRunGetter implements the DryRunGetter interface and can be used to GET/LIST values in the dryrun fake clientset
// Need to handle these routes in a special manner:
// - GET /default/services/kubernetes -- must return a valid Service
// - GET /clusterrolebindings/system:nodes -- can safely return a NotFound error
// - GET /kube-system/secrets/bootstrap-token-* -- can safely return a NotFound error
// - GET /nodes/<node-name> -- must return a valid Node
// - ...all other, unknown GETs/LISTs will be logged
type InitDryRunGetter struct {
	controlPlaneName string
	serviceSubnet    string
}

// InitDryRunGetter should implement the DryRunGetter interface
var _ DryRunGetter = &InitDryRunGetter{}

// NewInitDryRunGetter creates a new instance of the InitDryRunGetter struct
func NewInitDryRunGetter(controlPlaneName string, serviceSubnet string) *InitDryRunGetter {
	return &InitDryRunGetter{
		controlPlaneName: controlPlaneName,
		serviceSubnet:    serviceSubnet,
	}
}

// HandleGetAction handles GET actions to the dryrun clientset this interface supports
func (idr *InitDryRunGetter) HandleGetAction(action core.GetAction) (bool, runtime.Object, error) {
	funcs := []func(core.GetAction) (bool, runtime.Object, error){
		idr.handleKubernetesService,
		idr.handleGetNode,
		idr.handleSystemNodesClusterRoleBinding,
		idr.handleGetBootstrapToken,
	}
	for _, f := range funcs {
		handled, obj, err := f(action)
		if handled {
			return handled, obj, err
		}
	}

	return false, nil, nil
}

// HandleListAction handles GET actions to the dryrun clientset this interface supports.
// Currently there are no known LIST calls during kubeadm init this code has to take care of.
func (idr *InitDryRunGetter) HandleListAction(action core.ListAction) (bool, runtime.Object, error) {
	return false, nil, nil
}

// handleKubernetesService returns a faked Kubernetes service in order to be able to continue running kubeadm init.
// The kube-dns addon code GETs the Kubernetes service in order to extract the service subnet
func (idr *InitDryRunGetter) handleKubernetesService(action core.GetAction) (bool, runtime.Object, error) {
	if action.GetName() != "kubernetes" || action.GetNamespace() != metav1.NamespaceDefault || action.GetResource().Resource != "services" {
		// We can't handle this event
		return false, nil, nil
	}

	_, svcSubnet, err := net.ParseCIDR(idr.serviceSubnet)
	if err != nil {
		return true, nil, errors.Wrapf(err, "error parsing CIDR %q", idr.serviceSubnet)
	}

	internalAPIServerVirtualIP, err := ipallocator.GetIndexedIP(svcSubnet, 1)
	if err != nil {
		return true, nil, errors.Wrapf(err, "unable to get first IP address from the given CIDR (%s)", svcSubnet.String())
	}

	// The only used field of this Service object is the ClusterIP, which kube-dns uses to calculate its own IP
	return true, &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kubernetes",
			Namespace: metav1.NamespaceDefault,
			Labels: map[string]string{
				"component": "apiserver",
				"provider":  "kubernetes",
			},
		},
		Spec: v1.ServiceSpec{
			ClusterIP: internalAPIServerVirtualIP.String(),
			Ports: []v1.ServicePort{
				{
					Name:       "https",
					Port:       443,
					TargetPort: intstr.FromInt(6443),
				},
			},
		},
	}, nil
}

// handleGetNode returns a fake node object for the purpose of moving kubeadm init forwards.
func (idr *InitDryRunGetter) handleGetNode(action core.GetAction) (bool, runtime.Object, error) {
	if action.GetName() != idr.controlPlaneName || action.GetResource().Resource != "nodes" {
		// We can't handle this event
		return false, nil, nil
	}

	return true, &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: idr.controlPlaneName,
			Labels: map[string]string{
				"kubernetes.io/hostname": idr.controlPlaneName,
			},
			Annotations: map[string]string{},
		},
	}, nil
}

// handleSystemNodesClusterRoleBinding handles the GET call to the system:nodes clusterrolebinding
func (idr *InitDryRunGetter) handleSystemNodesClusterRoleBinding(action core.GetAction) (bool, runtime.Object, error) {
	if action.GetName() != constants.NodesClusterRoleBinding || action.GetResource().Resource != "clusterrolebindings" {
		// We can't handle this event
		return false, nil, nil
	}
	// We can safely return a NotFound error here as the code will just proceed normally and don't care about modifying this clusterrolebinding
	// This can only happen on an upgrade; and in that case the ClientBackedDryRunGetter impl will be used
	return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), "clusterrolebinding not found")
}

// handleGetBootstrapToken handles the case where kubeadm init creates the default token; and the token code GETs the
// bootstrap token secret first in order to check if it already exists
func (idr *InitDryRunGetter) handleGetBootstrapToken(action core.GetAction) (bool, runtime.Object, error) {
	if !strings.HasPrefix(action.GetName(), "bootstrap-token-") || action.GetNamespace() != metav1.NamespaceSystem || action.GetResource().Resource != "secrets" {
		// We can't handle this event
		return false, nil, nil
	}
	// We can safely return a NotFound error here as the code will just proceed normally and create the Bootstrap Token
	return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), "secret not found")
}
