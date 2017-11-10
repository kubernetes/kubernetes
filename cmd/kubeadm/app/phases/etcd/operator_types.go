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

package etcd

import (
	"fmt"

	"k8s.io/api/core/v1"
	ext "k8s.io/api/extensions/v1beta1"
	rbac "k8s.io/api/rbac/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd/spec"
)

// Label key used for selecting operator-controlled pods
const operatorLabelKey = "k8s-app"

func getEtcdClusterRole() rbac.ClusterRole {
	return rbac.ClusterRole{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "rbac.authorization.k8s.io/v1beta1",
			Kind:       "ClusterRole",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: kubeadmconstants.EtcdOperator,
		},
		Rules: []rbac.PolicyRule{
			{
				APIGroups: []string{"etcd.database.coreos.com"},
				Resources: []string{"etcdclusters"},
				Verbs:     []string{"*"},
			},
			{
				APIGroups: []string{"apiextensions.k8s.io"},
				Resources: []string{"customresourcedefinitions"},
				Verbs:     []string{"*"},
			},
			{
				APIGroups: []string{"storage.k8s.io"},
				Resources: []string{"storageclasses"},
				Verbs:     []string{"*"},
			},
			{
				APIGroups: []string{"apps"},
				Resources: []string{"deployments"},
				Verbs:     []string{"*"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"pods", "events", "services", "secrets", "endpoints", "persistentvolumeclaims"},
				Verbs:     []string{"*"},
			},
		},
	}
}

func getEtcdServiceAccount() v1.ServiceAccount {
	return v1.ServiceAccount{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ServiceAccount",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.EtcdOperator,
			Namespace: metav1.NamespaceSystem,
		},
	}
}

func getEtcdClusterRoleBinding() rbac.ClusterRoleBinding {
	return rbac.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "rbac.authorization.k8s.io/v1beta1",
			Kind:       "ClusterRoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.EtcdOperator,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     kubeadmconstants.EtcdOperator,
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      kubeadmconstants.EtcdOperator,
				Namespace: metav1.NamespaceSystem,
			},
		},
	}
}

func getEtcdOperatorDeployment(cfg *kubeadmapi.MasterConfiguration) ext.Deployment {
	replicas := int32(1)
	return ext.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "Deployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.EtcdOperator,
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{operatorLabelKey: kubeadmconstants.EtcdOperator},
		},
		Spec: ext.DeploymentSpec{
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{operatorLabelKey: kubeadmconstants.EtcdOperator},
				},
				Spec: v1.PodSpec{
					HostNetwork: true,
					DNSPolicy:   v1.DNSClusterFirstWithHostNet,
					Containers: []v1.Container{
						{
							Name:  kubeadmconstants.EtcdOperator,
							Image: images.GetCoreImage(kubeadmconstants.EtcdOperator, "quay.io/coreos", cfg.Etcd.SelfHosted.OperatorVersion, ""),
							Env: []v1.EnvVar{
								getFieldEnv("MY_POD_NAMESPACE", "metadata.namespace"),
								getFieldEnv("MY_POD_NAME", "metadata.name"),
							},
						},
					},
					Tolerations: []v1.Toleration{
						kubeadmconstants.MasterToleration,
						v1.Toleration{
							Key:      "node.kubernetes.io/network-unavailable",
							Operator: "Exists",
							Effect:   "NoSchedule",
						},
					},
					ServiceAccountName: kubeadmconstants.EtcdOperator,
				},
			},
		},
	}
}

func getFieldEnv(name, fieldPath string) v1.EnvVar {
	return v1.EnvVar{
		Name: name,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: fieldPath,
			},
		},
	}
}

func getEtcdCluster(cfg *kubeadmapi.MasterConfiguration) *spec.EtcdCluster {
	return &spec.EtcdCluster{
		TypeMeta: metav1.TypeMeta{
			Kind:       spec.CRDResourceKind,
			APIVersion: spec.SchemeGroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.EtcdClusterName,
			Namespace: metav1.NamespaceSystem,
		},
		Spec: spec.ClusterSpec{
			Size:    1,
			Version: cfg.Etcd.SelfHosted.EtcdVersion,
			SelfHosted: &spec.SelfHostedPolicy{
				BootMemberClientEndpoint: fmt.Sprintf("http://127.0.0.1:12379"),
				SkipBootMemberRemoval:    true,
			},
			Pod: &spec.PodPolicy{
				NodeSelector: map[string]string{kubeadmconstants.LabelNodeRoleMaster: ""},
				Tolerations:  []v1.Toleration{kubeadmconstants.MasterToleration},
			},
		},
	}
}
