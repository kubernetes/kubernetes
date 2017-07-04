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

package master

import (
	"fmt"

	"k8s.io/api/core/v1"
	ext "k8s.io/api/extensions/v1beta1"
	rbac "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
)

func getKubeProxyDS(cfg *kubeadmapi.MasterConfiguration) ext.DaemonSet {
	privileged := true
	return ext.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "DaemonSet",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeProxy,
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"k8s-app": kubeProxy},
		},
		Spec: ext.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"k8s-app": kubeProxy},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"k8s-app": kubeProxy},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            kubeProxy,
							Image:           images.GetCoreImage(images.KubeProxyImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
							ImagePullPolicy: "IfNotPresent",
							Command: []string{
								"/usr/local/bin/kube-proxy",
								"--config=/var/lib/kube-proxy/kube-proxy.yaml",
							},
							SecurityContext: &v1.SecurityContext{
								Privileged: &privileged,
							},
							VolumeMounts: []v1.VolumeMount{
								{
									MountPath: "/var/lib/kube-proxy",
									Name:      "kube-proxy",
								},
							},
						},
					},
					HostNetwork:        true,
					ServiceAccountName: kubeProxy,
					Tolerations:        []v1.Toleration{kubeadmconstants.MasterToleration},
					Volumes: []v1.Volume{
						{
							Name: "kube-proxy",
							VolumeSource: v1.VolumeSource{
								ConfigMap: &v1.ConfigMapVolumeSource{
									LocalObjectReference: v1.LocalObjectReference{Name: "kube-proxy"},
								},
							},
						},
					},
				},
			},
		},
	}
}

func launchSelfHostedProxy(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {
	sa := getKubeProxyServiceAccount()
	if _, err := client.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(&sa); err != nil {
		return fmt.Errorf("failed to create self-hosted kube-proxy serviceaccount [%v]", err)
	}

	crb := getKubeProxyClusterRoleBinding()
	if _, err := client.RbacV1beta1().ClusterRoleBindings().Create(&crb); err != nil {
		return fmt.Errorf("failed to create self-hosted kube-proxy clusterrolebinding [%v]", err)
	}

	cm := getKubeProxyConfigMap(cfg)
	if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(&cm); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create a new kube-proxy configmap: %v", err)
		}
		if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Update(&cm); err != nil {
			return fmt.Errorf("unable to update the kube-proxy configmap: %v", err)
		}
	}

	ds := getKubeProxyDS(cfg)
	if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Create(&ds); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create a new kube-proxy daemonset: %v", err)
		}
		if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Update(&ds); err != nil {
			return fmt.Errorf("unable to update the kube-proxy daemonset: %v", err)
		}
	}

	return nil
}

func getKubeProxyServiceAccount() v1.ServiceAccount {
	return v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeProxyServiceAccountName,
			Namespace: metav1.NamespaceSystem,
		},
	}
}

func getKubeProxyClusterRoleBinding() rbac.ClusterRoleBinding {
	return rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubeadm:node-proxier",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:node-proxier",
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      kubeadmconstants.KubeProxyServiceAccountName,
				Namespace: metav1.NamespaceSystem,
			},
		},
	}
}

func getKubeProxyConfigMap(cfg *kubeadmapi.MasterConfiguration) v1.ConfigMap {
	kubeConfig := fmt.Sprintf(`apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    server: https://%s:%d
  name: default
contexts:
- context:
    cluster: default
    namespace: default
    user: default
  name: default
current-context: default
users:
- name: default
  user:
    tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
`, cfg.API.AdvertiseAddress, cfg.API.BindPort)

	proxyConfig := fmt.Sprintf(`apiVersion: componentconfig/v1alpha1
kind: KubeProxyConfiguration
clusterCIDR: %s
clientConnection:
  kubeconfig: /var/lib/kube-proxy/kubeconfig
`, cfg.Networking.PodSubnet)

	return v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeProxy,
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"app": kubeProxy},
		},
		Data: map[string]string{
			"kubeconfig": kubeConfig,
			"proxy.conf": proxyConfig,
		},
	}
}
