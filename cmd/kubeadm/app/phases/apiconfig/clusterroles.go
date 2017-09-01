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

package apiconfig

import (
	"fmt"

	rbac "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/util/version"
)

// CreateRBACRules creates the essential RBAC rules for a minimally set-up cluster
// TODO: This function and phase package is DEPRECATED.
// When the v1.9 cycle starts and deletePermissiveNodesBindingWhenUsingNodeAuthorization can be removed, this package will be removed with it.
func CreateRBACRules(client clientset.Interface, k8sVersion *version.Version) error {
	if err := deletePermissiveNodesBindingWhenUsingNodeAuthorization(client, k8sVersion); err != nil {
		return fmt.Errorf("failed to remove the permissive 'system:nodes' Group Subject in the 'system:node' ClusterRoleBinding: %v", err)
	}
	return nil
}

func deletePermissiveNodesBindingWhenUsingNodeAuthorization(client clientset.Interface, k8sVersion *version.Version) error {

	// TODO: When the v1.9 cycle starts (targeting v1.9 at HEAD) and v1.8.0 is the minimum supported version, we can remove this function as the ClusterRoleBinding won't exist
	// or already have no such permissive subject
	nodesRoleBinding, err := client.RbacV1beta1().ClusterRoleBindings().Get(kubeadmconstants.NodesClusterRoleBinding, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Nothing to do; the RoleBinding doesn't exist
			return nil
		}
		return err
	}

	newSubjects := []rbac.Subject{}
	for _, subject := range nodesRoleBinding.Subjects {
		// Skip the subject that binds to the system:nodes group
		if subject.Name == kubeadmconstants.NodesGroup && subject.Kind == "Group" {
			continue
		}
		newSubjects = append(newSubjects, subject)
	}

	nodesRoleBinding.Subjects = newSubjects

	if _, err := client.RbacV1beta1().ClusterRoleBindings().Update(nodesRoleBinding); err != nil {
		return err
	}

	return nil
}
