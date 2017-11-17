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
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/util/version"
)

// PerformPostUpgradeTasks runs nearly the same functions as 'kubeadm init' would do
// Note that the markmaster phase is left out, not needed, and no token is created as that doesn't belong to the upgrade
func PerformPostUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.MasterConfiguration, newK8sVer *version.Version) error {
	errs := []error{}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptoken.AllowBootstrapTokensToPostCSRs(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeBootstrapTokens(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the 1.8.0+ nodes to rotate certificates and get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeCertificateRotation(client); err != nil {
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

	certAndKeyDir := kubeadmapiext.DefaultCertificatesDir
	shouldBackup, err := shouldBackupAPIServerCertAndKey(certAndKeyDir, newK8sVer)
	// Don't fail the upgrade phase if failing to determine to backup kube-apiserver cert and key.
	if err != nil {
		fmt.Printf("[postupgrade] WARNING: failed to determine to backup kube-apiserver cert and key: %v", err)
	} else if shouldBackup {
		// Don't fail the upgrade phase if failing to backup kube-apiserver cert and key.
		if err := backupAPIServerCertAndKey(certAndKeyDir); err != nil {
			fmt.Printf("[postupgrade] WARNING: failed to backup kube-apiserver cert and key: %v", err)
		}
		if err := certsphase.CreateAPIServerCertAndKeyFiles(cfg); err != nil {
			errs = append(errs, err)
		}
	}

	// Upgrade kube-dns and kube-proxy
	if err := dns.EnsureDNSAddon(cfg, client); err != nil {
		errs = append(errs, err)
	}

	if err := coreDNSDeployment(cfg, client); err != nil {
		errs = append(errs, err)
	}

	if err := proxy.EnsureProxyAddon(cfg, client); err != nil {
		errs = append(errs, err)
	}
	return errors.NewAggregate(errs)
}

func coreDNSDeployment(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	if features.Enabled(cfg.FeatureGates, features.CoreDNS) {
		return apiclient.TryRunCommand(func() error {
			getCoreDNS, err := client.AppsV1beta2().Deployments(metav1.NamespaceSystem).Get(kubeadmconstants.CoreDNS, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if getCoreDNS.Status.ReadyReplicas == 0 {
				return fmt.Errorf("the CodeDNS deployment isn't ready yet")
			}
			err = client.AppsV1beta2().Deployments(metav1.NamespaceSystem).Delete(kubeadmconstants.KubeDNS, nil)
			if err != nil {
				return err
			}
			return nil
		}, 5)
	}
	return nil
}
