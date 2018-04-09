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

package storage

import (
	"math/rand"
	"time"

	"k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	. "github.com/onsi/ginkgo"
)

const (
	csiExternalAttacherImage    string = "quay.io/k8scsi/csi-attacher:v0.2.0"
	csiExternalProvisionerImage string = "quay.io/k8scsi/csi-provisioner:v0.2.0"
	csiDriverRegistrarImage     string = "quay.io/k8scsi/driver-registrar:v0.2.0"
)

func csiServiceAccount(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
) *v1.ServiceAccount {
	serviceAccountName := config.Prefix + "-service-account"
	serviceAccountClient := client.CoreV1().ServiceAccounts(config.Namespace)
	sa := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name: serviceAccountName,
		},
	}

	serviceAccountClient.Delete(sa.GetName(), &metav1.DeleteOptions{})
	err := wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
		_, err := serviceAccountClient.Get(sa.GetName(), metav1.GetOptions{})
		return apierrs.IsNotFound(err), nil
	})
	framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

	if teardown {
		return nil
	}

	ret, err := serviceAccountClient.Create(sa)
	if err != nil {
		framework.ExpectNoError(err, "Failed to create %s service account: %v", sa.GetName(), err)
	}

	return ret
}

func csiClusterRole(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
) *rbacv1.ClusterRole {
	clusterRoleClient := client.RbacV1().ClusterRoles()
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-cluster-role",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"persistentvolumes"},
				Verbs:     []string{"create", "delete", "get", "list", "watch", "update"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"persistentvolumeclaims"},
				Verbs:     []string{"get", "list", "watch", "update"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"events"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"secrets"},
				Verbs:     []string{"get", "list"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"nodes"},
				Verbs:     []string{"get", "list", "watch", "update"},
			},
			{
				APIGroups: []string{"storage.k8s.io"},
				Resources: []string{"volumeattachments"},
				Verbs:     []string{"get", "list", "watch", "update"},
			},
			{
				APIGroups: []string{"storage.k8s.io"},
				Resources: []string{"storageclasses"},
				Verbs:     []string{"get", "list", "watch"},
			},
		},
	}

	clusterRoleClient.Delete(role.GetName(), &metav1.DeleteOptions{})
	err := wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
		_, err := clusterRoleClient.Get(role.GetName(), metav1.GetOptions{})
		return apierrs.IsNotFound(err), nil
	})
	framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

	if teardown {
		return nil
	}

	ret, err := clusterRoleClient.Create(role)
	if err != nil {
		framework.ExpectNoError(err, "Failed to create %s cluster role: %v", role.GetName(), err)
	}

	return ret
}

func csiClusterRoleBinding(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
	sa *v1.ServiceAccount,
	clusterRole *rbacv1.ClusterRole,
) *rbacv1.ClusterRoleBinding {
	clusterRoleBindingClient := client.RbacV1().ClusterRoleBindings()
	binding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-role-binding",
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      sa.GetName(),
				Namespace: sa.GetNamespace(),
			},
		},
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     clusterRole.GetName(),
			APIGroup: "rbac.authorization.k8s.io",
		},
	}

	clusterRoleBindingClient.Delete(binding.GetName(), &metav1.DeleteOptions{})
	err := wait.Poll(2*time.Second, 10*time.Minute, func() (bool, error) {
		_, err := clusterRoleBindingClient.Get(binding.GetName(), metav1.GetOptions{})
		return apierrs.IsNotFound(err), nil
	})
	framework.ExpectNoError(err, "Timed out waiting for deletion: %v", err)

	if teardown {
		return nil
	}

	ret, err := clusterRoleBindingClient.Create(binding)
	if err != nil {
		framework.ExpectNoError(err, "Failed to create %s role binding: %v", binding.GetName(), err)
	}

	return ret
}

var _ = utils.SIGDescribe("CSI Volumes [Flaky]", func() {
	f := framework.NewDefaultFramework("csi-mock-plugin")

	var (
		cs     clientset.Interface
		ns     *v1.Namespace
		node   v1.Node
		config framework.VolumeTestConfig
	)

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		node = nodes.Items[rand.Intn(len(nodes.Items))]
		config = framework.VolumeTestConfig{
			Namespace:         ns.Name,
			Prefix:            "csi",
			ClientNodeName:    node.Name,
			ServerNodeName:    node.Name,
			WaitForCompletion: true,
		}
	})

	// Create one of these for each of the drivers to be tested
	// CSI hostPath driver test
	Describe("Sanity CSI plugin test using hostPath CSI driver", func() {

		var (
			clusterRole    *rbacv1.ClusterRole
			serviceAccount *v1.ServiceAccount
		)

		BeforeEach(func() {
			By("deploying csi hostpath driver")
			clusterRole = csiClusterRole(cs, config, false)
			serviceAccount = csiServiceAccount(cs, config, false)
			csiClusterRoleBinding(cs, config, false, serviceAccount, clusterRole)
			csiHostPathPod(cs, config, false, f, serviceAccount)
		})

		AfterEach(func() {
			By("uninstalling csi hostpath driver")
			csiHostPathPod(cs, config, true, f, serviceAccount)
			csiClusterRoleBinding(cs, config, true, serviceAccount, clusterRole)
			serviceAccount = csiServiceAccount(cs, config, true)
			clusterRole = csiClusterRole(cs, config, true)
		})

		It("should provision storage with a hostPath CSI driver", func() {
			t := storageClassTest{
				name:         "csi-hostpath",
				provisioner:  "csi-hostpath",
				parameters:   map[string]string{},
				claimSize:    "1Gi",
				expectedSize: "1Gi",
				nodeName:     node.Name,
			}

			claim := newClaim(t, ns.GetName(), "")
			class := newStorageClass(t, ns.GetName(), "")
			claim.Spec.StorageClassName = &class.ObjectMeta.Name
			testDynamicProvisioning(t, cs, claim, class)
		})
	})
})
