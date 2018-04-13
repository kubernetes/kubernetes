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
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	. "github.com/onsi/ginkgo"
)

const (
	csiExternalAttacherImage              string = "quay.io/k8scsi/csi-attacher:v0.2.0"
	csiExternalProvisionerImage           string = "quay.io/k8scsi/csi-provisioner:v0.2.0"
	csiDriverRegistrarImage               string = "quay.io/k8scsi/driver-registrar:v0.2.0"
	csiExternalProvisionerClusterRoleName string = "system:csi-external-provisioner"
	csiExternalAttacherClusterRoleName    string = "system:csi-external-attacher"
	csiDriverRegistrarClusterRoleName     string = "csi-driver-registrar"
)

// Create the driver registrar cluster role if it doesn't exist, no teardown so that tests
// are parallelizable. This role will be shared with many of the CSI tests.
func csiDriverRegistrarClusterRole(
	config framework.VolumeTestConfig,
) *rbacv1.ClusterRole {
	// TODO(Issue: #62237) Remove impersonation workaround and cluster role when issue resolved
	By("Creating an impersonating superuser kubernetes clientset to define cluster role")
	rc, err := framework.LoadConfig()
	framework.ExpectNoError(err)
	rc.Impersonate = restclient.ImpersonationConfig{
		UserName: "superuser",
		Groups:   []string{"system:masters"},
	}
	superuserClientset, err := clientset.NewForConfig(rc)
	By("Creating the CSI driver registrar cluster role")
	clusterRoleClient := superuserClientset.RbacV1().ClusterRoles()
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: csiDriverRegistrarClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{

			{
				APIGroups: []string{""},
				Resources: []string{"events"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"nodes"},
				Verbs:     []string{"get", "update", "patch"},
			},
		},
	}

	ret, err := clusterRoleClient.Create(role)
	if err != nil {
		if apierrs.IsAlreadyExists(err) {
			return ret
		}
		framework.ExpectNoError(err, "Failed to create %s cluster role: %v", role.GetName(), err)
	}

	return ret
}

func csiServiceAccount(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	componentName string,
	teardown bool,
) *v1.ServiceAccount {
	By("Creating a CSI service account")
	serviceAccountName := config.Prefix + "-" + componentName + "-service-account"
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

func csiClusterRoleBindings(
	client clientset.Interface,
	config framework.VolumeTestConfig,
	teardown bool,
	sa *v1.ServiceAccount,
	clusterRolesNames []string,
) {
	By("Binding cluster roles to the CSI service account")
	clusterRoleBindingClient := client.RbacV1().ClusterRoleBindings()
	for _, clusterRoleName := range clusterRolesNames {

		binding := &rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: config.Prefix + "-" + clusterRoleName + "-" + config.Namespace + "-role-binding",
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
				Name:     clusterRoleName,
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
			return
		}

		_, err = clusterRoleBindingClient.Create(binding)
		if err != nil {
			framework.ExpectNoError(err, "Failed to create %s role binding: %v", binding.GetName(), err)
		}
	}
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
		csiDriverRegistrarClusterRole(config)
	})

	// Create one of these for each of the drivers to be tested
	// CSI hostPath driver test
	Describe("Sanity CSI plugin test using hostPath CSI driver", func() {
		var (
			serviceAccount           *v1.ServiceAccount
			combinedClusterRoleNames []string = []string{
				csiExternalAttacherClusterRoleName,
				csiExternalProvisionerClusterRoleName,
				csiDriverRegistrarClusterRoleName,
			}
		)

		BeforeEach(func() {
			By("deploying csi hostpath driver")
			serviceAccount = csiServiceAccount(cs, config, "hostpath", false)
			csiClusterRoleBindings(cs, config, false, serviceAccount, combinedClusterRoleNames)
			csiHostPathPod(cs, config, false, f, serviceAccount)
		})

		AfterEach(func() {
			By("uninstalling csi hostpath driver")
			csiHostPathPod(cs, config, true, f, serviceAccount)
			csiClusterRoleBindings(cs, config, true, serviceAccount, combinedClusterRoleNames)
			csiServiceAccount(cs, config, "hostpath", true)
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
