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
	"os"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
)

// PerformPostUpgradeTasks runs nearly the same functions as 'kubeadm init' would do
// Note that the mark-control-plane phase is left out, not needed, and no token is created as that doesn't belong to the upgrade
func PerformPostUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, newK8sVer *version.Version, dryRun bool) error {
	errs := []error{}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		errs = append(errs, err)
	}

	// Create the new, version-branched kubelet ComponentConfig ConfigMap
	if err := kubeletphase.CreateConfigMap(&cfg.ClusterConfiguration, client); err != nil {
		errs = append(errs, errors.Wrap(err, "error creating kubelet configuration ConfigMap"))
	}

	// Write the new kubelet config down to disk and the env file if needed
	if err := writeKubeletConfigFiles(client, cfg, newK8sVer, dryRun); err != nil {
		errs = append(errs, err)
	}

	// Annotate the node with the crisocket information, sourced either from the InitConfiguration struct or
	// --cri-socket.
	// TODO: In the future we want to use something more official like NodeStatus or similar for detecting this properly
	if err := patchnodephase.AnnotateCRISocket(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.CRISocket); err != nil {
		errs = append(errs, errors.Wrap(err, "error uploading crisocket"))
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptoken.AllowBootstrapTokensToPostCSRs(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeBootstrapTokens(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
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

	// Upgrade kube-dns/CoreDNS and kube-proxy
	if err := dns.EnsureDNSAddon(&cfg.ClusterConfiguration, client); err != nil {
		errs = append(errs, err)
	}
	// Remove the old DNS deployment if a new DNS service is now used (kube-dns to CoreDNS or vice versa)
	if err := removeOldDNSDeploymentIfAnotherDNSIsUsed(&cfg.ClusterConfiguration, client, dryRun); err != nil {
		errs = append(errs, err)
	}

	if err := proxy.EnsureProxyAddon(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, client); err != nil {
		errs = append(errs, err)
	}
	return errorsutil.NewAggregate(errs)
}

func removeOldDNSDeploymentIfAnotherDNSIsUsed(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, dryRun bool) error {
	return apiclient.TryRunCommand(func() error {
		installedDeploymentName := kubeadmconstants.KubeDNSDeploymentName
		deploymentToDelete := kubeadmconstants.CoreDNSDeploymentName

		if cfg.DNS.Type == kubeadmapi.CoreDNS {
			installedDeploymentName = kubeadmconstants.CoreDNSDeploymentName
			deploymentToDelete = kubeadmconstants.KubeDNSDeploymentName
		}

		// If we're dry-running, we don't need to wait for the new DNS addon to become ready
		if !dryRun {
			dnsDeployment, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Get(installedDeploymentName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if dnsDeployment.Status.ReadyReplicas == 0 {
				return errors.New("the DNS deployment isn't ready yet")
			}
		}

		// We don't want to wait for the DNS deployment above to become ready when dryrunning (as it never will)
		// but here we should execute the DELETE command against the dryrun clientset, as it will only be logged
		err := apiclient.DeleteDeploymentForeground(client, metav1.NamespaceSystem, deploymentToDelete)
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		return nil
	}, 10)
}

func writeKubeletConfigFiles(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, newK8sVer *version.Version, dryRun bool) error {
	kubeletDir, err := GetKubeletDir(dryRun)
	if err != nil {
		// The error here should never occur in reality, would only be thrown if /tmp doesn't exist on the machine.
		return err
	}
	errs := []error{}
	// Write the configuration for the kubelet down to disk so the upgraded kubelet can start with fresh config
	if err := kubeletphase.DownloadConfig(client, newK8sVer, kubeletDir); err != nil {
		// Tolerate the error being NotFound when dryrunning, as there is a pretty common scenario: the dryrun process
		// *would* post the new kubelet-config-1.X configmap that doesn't exist now when we're trying to download it
		// again.
		if !(apierrors.IsNotFound(err) && dryRun) {
			errs = append(errs, errors.Wrap(err, "error downloading kubelet configuration from the ConfigMap"))
		}
	}

	if dryRun { // Print what contents would be written
		dryrunutil.PrintDryRunFile(kubeadmconstants.KubeletConfigurationFileName, kubeletDir, kubeadmconstants.KubeletRunDirectory, os.Stdout)
	}
	return errorsutil.NewAggregate(errs)
}

// GetKubeletDir gets the kubelet directory based on whether the user is dry-running this command or not.
func GetKubeletDir(dryRun bool) (string, error) {
	if dryRun {
		return kubeadmconstants.CreateTempDirForKubeadm("", "kubeadm-upgrade-dryrun")
	}
	return kubeadmconstants.KubeletRunDirectory, nil
}

// moveFiles moves files from one directory to another.
func moveFiles(files map[string]string) error {
	filesToRecover := map[string]string{}
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			return rollbackFiles(filesToRecover, err)
		}
		filesToRecover[to] = from
	}
	return nil
}

// rollbackFiles moves the files back to the original directory.
func rollbackFiles(files map[string]string, originalErr error) error {
	errs := []error{originalErr}
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Errorf("couldn't move these files: %v. Got errors: %v", files, errorsutil.NewAggregate(errs))
}
