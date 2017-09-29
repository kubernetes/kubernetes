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
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/pkg/util/version"
)

// PerformPostUpgradeTasks runs nearly the same functions as 'kubeadm init' would do
// Note that the markmaster phase is left out, not needed, and no token is created as that doesn't belong to the upgrade
func PerformPostUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.MasterConfiguration, k8sVersion *version.Version) error {
	errs := []error{}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		errs = append(errs, err)
	}

	// Handle Bootstrap Tokens graduating to from alpha to beta in the v1.7 -> v1.8 upgrade
	// That transition requires two minor changes

	// Remove the old ClusterRoleBinding for approving if it already exists due to the reasons outlined in the comment below
	if err := deleteOldApprovalClusterRoleBindingIfExists(client, k8sVersion); err != nil {
		errs = append(errs, err)
	}
	// Upgrade the Bootstrap Tokens' authentication group
	if err := upgradeBootstrapTokens(client, k8sVersion); err != nil {
		errs = append(errs, err)
	}
	// Upgrade the cluster-info RBAC rules
	if err := deleteWronglyNamedClusterInfoRBACRules(client, k8sVersion); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptoken.AllowBootstrapTokensToPostCSRs(client, k8sVersion); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeBootstrapTokens(client, k8sVersion); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the 1.8.0+ nodes to rotate certificates and get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeCertificateRotation(client, k8sVersion); err != nil {
		errs = append(errs, err)
	}

	// TODO: Is this needed to do here? I think that updating cluster info should probably be separate from a normal upgrade
	// Create the cluster-info ConfigMap with the associated RBAC rules
	// if err := clusterinfo.CreateBootstrapConfigMapIfNotExists(client, kubeadmconstants.GetAdminKubeConfigPath()); err != nil {
	// 	return err
	//}
	// Create/update RBAC rules that makes the cluster-info ConfigMap reachable
	if err := clusterinfo.CreateClusterInfoRBACRules(client); err != nil {
		errs = append(errs, err)
	}

	// Upgrade kube-dns and kube-proxy
	if err := dns.EnsureDNSAddon(cfg, client); err != nil {
		errs = append(errs, err)
	}
	if err := proxy.EnsureProxyAddon(cfg, client); err != nil {
		errs = append(errs, err)
	}
	return errors.NewAggregate(errs)
}
