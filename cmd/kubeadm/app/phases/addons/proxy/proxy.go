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

package proxy

import (
	"fmt"
	"runtime"

	apps "k8s.io/api/apps/v1beta2"
	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

const (
	// KubeProxyClusterRoleName sets the name for the kube-proxy ClusterRole
	// TODO: This k8s-generic, well-known constant should be fetchable from another source, not be in this package
	KubeProxyClusterRoleName = "system:node-proxier"

	// KubeProxyServiceAccountName describes the name of the ServiceAccount for the kube-proxy addon
	KubeProxyServiceAccountName = "kube-proxy"
)

// EnsureProxyAddon creates the kube-proxy addons
func EnsureProxyAddon(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	if err := CreateServiceAccount(client); err != nil {
		return fmt.Errorf("error when creating kube-proxy service account: %v", err)
	}

	// Generate Master Enpoint kubeconfig file
	masterEndpoint, err := kubeadmutil.GetMasterEndpoint(cfg)
	if err != nil {
		return err
	}

	proxyConfigMapBytes, err := kubeadmutil.ParseTemplate(KubeProxyConfigMap, struct{ MasterEndpoint string }{
		// Fetch this value from the kubeconfig file
		MasterEndpoint: masterEndpoint})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy configmap template: %v", err)
	}

	proxyDaemonSetBytes, err := kubeadmutil.ParseTemplate(KubeProxyDaemonSet, struct{ ImageRepository, Arch, Version, ImageOverride, ClusterCIDR, MasterTaintKey, CloudTaintKey string }{
		ImageRepository: cfg.GetControlPlaneImageRepository(),
		Arch:            runtime.GOARCH,
		Version:         kubeadmutil.KubernetesVersionToImageTag(cfg.KubernetesVersion),
		ImageOverride:   cfg.UnifiedControlPlaneImage,
		ClusterCIDR:     getClusterCIDR(cfg.Networking.PodSubnet),
		MasterTaintKey:  kubeadmconstants.LabelNodeRoleMaster,
		CloudTaintKey:   algorithm.TaintExternalCloudProvider,
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy daemonset template: %v", err)
	}

	if err := createKubeProxyAddon(proxyConfigMapBytes, proxyDaemonSetBytes, client); err != nil {
		return err
	}
	if err := CreateRBACRules(client); err != nil {
		return fmt.Errorf("error when creating kube-proxy RBAC rules: %v", err)
	}

	fmt.Println("[addons] Applied essential addon: kube-proxy")
	return nil
}

// CreateServiceAccount creates the necessary serviceaccounts that kubeadm uses/might use, if they don't already exist.
func CreateServiceAccount(client clientset.Interface) error {

	return apiclient.CreateOrUpdateServiceAccount(client, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      KubeProxyServiceAccountName,
			Namespace: metav1.NamespaceSystem,
		},
	})
}

// CreateRBACRules creates the essential RBAC rules for a minimally set-up cluster
func CreateRBACRules(client clientset.Interface) error {
	return createClusterRoleBindings(client)
}

func createKubeProxyAddon(configMapBytes, daemonSetbytes []byte, client clientset.Interface) error {
	kubeproxyConfigMap := &v1.ConfigMap{}
	if err := kuberuntime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), configMapBytes, kubeproxyConfigMap); err != nil {
		return fmt.Errorf("unable to decode kube-proxy configmap %v", err)
	}

	// Create the ConfigMap for kube-proxy or update it in case it already exists
	if err := apiclient.CreateOrUpdateConfigMap(client, kubeproxyConfigMap); err != nil {
		return err
	}

	kubeproxyDaemonSet := &apps.DaemonSet{}
	if err := kuberuntime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), daemonSetbytes, kubeproxyDaemonSet); err != nil {
		return fmt.Errorf("unable to decode kube-proxy daemonset %v", err)
	}

	// Create the DaemonSet for kube-proxy or update it in case it already exists
	return apiclient.CreateOrUpdateDaemonSet(client, kubeproxyDaemonSet)
}

func createClusterRoleBindings(client clientset.Interface) error {
	return apiclient.CreateOrUpdateClusterRoleBinding(client, &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:node-proxier",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     KubeProxyClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind:      rbac.ServiceAccountKind,
				Name:      KubeProxyServiceAccountName,
				Namespace: metav1.NamespaceSystem,
			},
		},
	})
}

func getClusterCIDR(podsubnet string) string {
	if len(podsubnet) == 0 {
		return ""
	}
	return "- --cluster-cidr=" + podsubnet
}
