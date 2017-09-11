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

package upgrade

import (
	"bytes"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	"k8s.io/kubernetes/pkg/api"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	oldClusterInfoRole = "system:bootstrap-signer-clusterinfo"
)

// deleteOldApprovalClusterRoleBindingIfExists exists because the roleRef of the NodeAutoApproveBootstrapClusterRoleBinding changed between
// v1.7 and v1.8, and roleRef updates are not possible. So in order to change that binding's roleRef, we have to delete it if it already exists
// TODO: When the v1.9 cycle starts, we can remove this logic, as the kubeadm v1.9 CLI doesn't support upgrading from v1.7
func deleteOldApprovalClusterRoleBindingIfExists(client clientset.Interface, k8sVersion *version.Version) error {

	// Gate this upgrade behavior for new clusters above v1.9.0-alpha.3 where this change took place
	if k8sVersion.AtLeast(constants.MinimumCSRAutoApprovalClusterRolesVersion) {

		err := client.RbacV1beta1().ClusterRoleBindings().Delete(nodebootstraptoken.NodeAutoApproveBootstrapClusterRoleBinding, &metav1.DeleteOptions{})
		// If the binding was not found, happily continue
		if apierrors.IsNotFound(err) {
			return nil
		}
		// If an unexpected error occurred, return it
		if err != nil {
			return err
		}
	}
	// The binding was successfully deleted
	return nil
}

// deleteWronglyNamedClusterInfoRBACRules exists because the cluster-info Role's name changed from "system:bootstrap-signer-clusterinfo" in v1.7 to
// "kubeadm:bootstrap-signer-clusterinfo" in v1.8. It was incorrectly prefixed "system:" in v1.7
// The old, incorrectly-named Role should be removed and roleRef updates on the binding are not possible. So in order to change that binding's roleRef,
// we have to delete it if it already exists
// TODO: When the v1.9 cycle starts, we can remove this logic, as the kubeadm v1.9 CLI doesn't support upgrading from v1.7
func deleteWronglyNamedClusterInfoRBACRules(client clientset.Interface, k8sVersion *version.Version) error {
	// Gate this upgrade behavior for new clusters above v1.8.0-beta.0 where this change took place
	if k8sVersion.AtLeast(constants.UseEnableBootstrapTokenAuthFlagVersion) {

		if err := removeOldRole(client); err != nil {
			return err
		}
		if err := removeOldRoleBinding(client); err != nil {
			return err
		}
	}
	// The binding was successfully deleted
	return nil
}

func removeOldRole(client clientset.Interface) error {
	err := client.RbacV1beta1().Roles(metav1.NamespacePublic).Delete(oldClusterInfoRole, &metav1.DeleteOptions{})
	// If the binding was not found, happily continue
	if apierrors.IsNotFound(err) {
		return nil
	}
	// If an unexpected error occurred, return it
	if err != nil {
		return err
	}
	// The role was successfully deleted
	return nil
}

func removeOldRoleBinding(client clientset.Interface) error {
	err := client.RbacV1beta1().RoleBindings(metav1.NamespacePublic).Delete(clusterinfo.BootstrapSignerClusterRoleName, &metav1.DeleteOptions{})
	// If the binding was not found, happily continue
	if apierrors.IsNotFound(err) {
		return nil
	}
	// If an unexpected error occurred, return it
	if err != nil {
		return err
	}
	// The binding was successfully removed
	return nil
}

// upgradeBootstrapTokens handles the transition from alpha bootstrap tokens to beta. There isn't much that is changing,
// but the group that a Bootstrap Token authenticates as changes from "system:bootstrappers" (alpha) in v1.7 to
// "system:bootstrappers:kubeadm:default-node-token" (beta). To handle this transition correctly, the RBAC bindings earlier
// bound to "system:bootstrappers" are now bound to "system:bootstrappers:kubeadm:default-node-token". To make v1.7 tokens
// still valid in v1.8; this code makes sure that all tokens that were used for authentication in v1.7 have the right group
// bound to it in v1.8.
// TODO: When the v1.9 cycle starts, we can remove this logic, as the kubeadm v1.9 CLI doesn't support upgrading from v1.7
func upgradeBootstrapTokens(client clientset.Interface, k8sVersion *version.Version) error {

	// Gate this upgrade behavior for new clusters above v1.8.0-beta.0; where this BT change took place
	if k8sVersion.AtLeast(constants.UseEnableBootstrapTokenAuthFlagVersion) {

		tokenSelector := fields.SelectorFromSet(
			map[string]string{
				api.SecretTypeField: string(bootstrapapi.SecretTypeBootstrapToken),
			},
		)
		listOptions := metav1.ListOptions{
			FieldSelector: tokenSelector.String(),
		}

		secrets, err := client.CoreV1().Secrets(metav1.NamespaceSystem).List(listOptions)
		if err != nil {
			return fmt.Errorf("failed to list bootstrap tokens: %v", err)
		}

		errs := []error{}
		for _, secret := range secrets.Items {
			// If this Bootstrap Token is used for authentication, the permissions it had in v1.7 should be preserved
			if bytes.Equal(secret.Data[bootstrapapi.BootstrapTokenUsageAuthentication], []byte("true")) {

				secret.Data[bootstrapapi.BootstrapTokenExtraGroupsKey] = []byte(constants.GetNodeBootstrapTokenAuthGroup(k8sVersion))

				// Update the Bootstrap Token Secret
				if _, err := client.CoreV1().Secrets(metav1.NamespaceSystem).Update(&secret); err != nil {
					errs = append(errs, err)
				}
			}
		}
		return errors.NewAggregate(errs)
	}
	return nil
}
