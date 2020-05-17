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

package clusterinfo

import (
	"fmt"

	"github.com/pkg/errors"
	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

const (
	// BootstrapSignerClusterRoleName sets the name for the ClusterRole that allows access to ConfigMaps in the kube-public ns
	BootstrapSignerClusterRoleName = "kubeadm:bootstrap-signer-clusterinfo"
)

// CreateBootstrapConfigMapIfNotExists creates the kube-public ConfigMap if it doesn't exist already
func CreateBootstrapConfigMapIfNotExists(client clientset.Interface, file string) error {

	fmt.Printf("[bootstrap-token] Creating the %q ConfigMap in the %q namespace\n", bootstrapapi.ConfigMapClusterInfo, metav1.NamespacePublic)

	klog.V(1).Infoln("[bootstrap-token] loading admin kubeconfig")
	adminConfig, err := clientcmd.LoadFromFile(file)
	if err != nil {
		return errors.Wrap(err, "failed to load admin kubeconfig")
	}

	adminCluster := adminConfig.Contexts[adminConfig.CurrentContext].Cluster
	// Copy the cluster from admin.conf to the bootstrap kubeconfig, contains the CA cert and the server URL
	klog.V(1).Infoln("[bootstrap-token] copying the cluster from admin.conf to the bootstrap kubeconfig")
	bootstrapConfig := &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"": adminConfig.Clusters[adminCluster],
		},
	}
	bootstrapBytes, err := clientcmd.Write(*bootstrapConfig)
	if err != nil {
		return err
	}

	// Create or update the ConfigMap in the kube-public namespace
	klog.V(1).Infoln("[bootstrap-token] creating/updating ConfigMap in kube-public namespace")
	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      bootstrapapi.ConfigMapClusterInfo,
			Namespace: metav1.NamespacePublic,
		},
		Data: map[string]string{
			bootstrapapi.KubeConfigKey: string(bootstrapBytes),
		},
	})
}

// CreateClusterInfoRBACRules creates the RBAC rules for exposing the cluster-info ConfigMap in the kube-public namespace to unauthenticated users
func CreateClusterInfoRBACRules(client clientset.Interface) error {
	klog.V(1).Infoln("creating the RBAC rules for exposing the cluster-info ConfigMap in the kube-public namespace")
	err := apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapSignerClusterRoleName,
			Namespace: metav1.NamespacePublic,
		},
		Rules: []rbac.PolicyRule{
			{
				Verbs:         []string{"get"},
				APIGroups:     []string{""},
				Resources:     []string{"configmaps"},
				ResourceNames: []string{bootstrapapi.ConfigMapClusterInfo},
			},
		},
	})
	if err != nil {
		return err
	}

	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapSignerClusterRoleName,
			Namespace: metav1.NamespacePublic,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     BootstrapSignerClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.UserKind,
				Name: user.Anonymous,
			},
		},
	})
}
