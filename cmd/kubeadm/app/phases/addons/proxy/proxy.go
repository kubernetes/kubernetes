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

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	rbac "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	apiclientutil "k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/version"
	k8sversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

const (
	// KubeProxyClusterRoleName sets the name for the kube-proxy ClusterRole
	// TODO: This k8s-generic, well-known constant should be fetchable from another source, not be in this package
	KubeProxyClusterRoleName = "system:node-proxier"
)

// EnsureProxyAddon creates the kube-proxy and kube-dns addons
func EnsureProxyAddon(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	if err := CreateServiceAccount(client); err != nil {
		return fmt.Errorf("error when creating kube-proxy service account: %v", err)
	}

	proxyConfigMapBytes, err := kubeadmutil.ParseTemplate(KubeProxyConfigMap, struct{ MasterEndpoint string }{
		// Fetch this value from the kubeconfig file
		MasterEndpoint: fmt.Sprintf("https://%s:%d", cfg.API.AdvertiseAddress, cfg.API.BindPort),
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy configmap template: %v", err)
	}

	proxyDaemonSetBytes, err := kubeadmutil.ParseTemplate(KubeProxyDaemonSet, struct{ ImageRepository, Arch, Version, ImageOverride, ClusterCIDR, MasterTaintKey, CloudTaintKey string }{
		ImageRepository: cfg.ImageRepository,
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

	if err = createKubeProxyAddon(proxyConfigMapBytes, proxyDaemonSetBytes, client); err != nil {
		return err
	}
	fmt.Println("[addons] Applied essential addon: kube-proxy")

	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return fmt.Errorf("couldn't parse kubernetes version %q: %v", cfg.KubernetesVersion, err)
	}
	if err = CreateRBACRules(client, k8sVersion); err != nil {
		return fmt.Errorf("error when creating kube-proxy RBAC rules: %v", err)
	}

	return nil
}

// CreateServiceAccount creates the necessary serviceaccounts that kubeadm uses/might use, if they don't already exist.
func CreateServiceAccount(client clientset.Interface) error {
	sa := v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeProxyServiceAccountName,
			Namespace: metav1.NamespaceSystem,
		},
	}

	if _, err := client.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(&sa); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return err
		}
	}
	return nil
}

// CreateRBACRules creates the essential RBAC rules for a minimally set-up cluster
func CreateRBACRules(client clientset.Interface, k8sVersion *k8sversion.Version) error {
	if err := createClusterRoleBindings(client); err != nil {
		return err
	}
	return nil
}

func createKubeProxyAddon(configMapBytes, daemonSetbytes []byte, client clientset.Interface) error {
	kubeproxyConfigMap := &v1.ConfigMap{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), configMapBytes, kubeproxyConfigMap); err != nil {
		return fmt.Errorf("unable to decode kube-proxy configmap %v", err)
	}

	if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(kubeproxyConfigMap); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create a new kube-proxy configmap: %v", err)
		}

		if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Update(kubeproxyConfigMap); err != nil {
			return fmt.Errorf("unable to update the kube-proxy configmap: %v", err)
		}
	}

	kubeproxyDaemonSet := &extensions.DaemonSet{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), daemonSetbytes, kubeproxyDaemonSet); err != nil {
		return fmt.Errorf("unable to decode kube-proxy daemonset %v", err)
	}

	if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Create(kubeproxyDaemonSet); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create a new kube-proxy daemonset: %v", err)
		}

		if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Update(kubeproxyDaemonSet); err != nil {
			return fmt.Errorf("unable to update the kube-proxy daemonset: %v", err)
		}
	}
	return nil
}

func createClusterRoleBindings(client clientset.Interface) error {
	return apiclientutil.CreateClusterRoleBindingIfNotExists(client, &rbac.ClusterRoleBinding{
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
				Name:      kubeadmconstants.KubeProxyServiceAccountName,
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
