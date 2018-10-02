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
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
)

var expiry = 180 * 24 * time.Hour

// PerformPostUpgradeTasks runs nearly the same functions as 'kubeadm init' would do
// Note that the markmaster phase is left out, not needed, and no token is created as that doesn't belong to the upgrade
func PerformPostUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, newK8sVer *version.Version, dryRun bool) error {
	errs := []error{}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		errs = append(errs, err)
	}

	// Create the new, version-branched kubelet ComponentConfig ConfigMap
	if err := kubeletphase.CreateConfigMap(cfg, client); err != nil {
		errs = append(errs, fmt.Errorf("error creating kubelet configuration ConfigMap: %v", err))
	}

	// Write the new kubelet config down to disk and the env file if needed
	if err := writeKubeletConfigFiles(client, cfg, newK8sVer, dryRun); err != nil {
		errs = append(errs, err)
	}

	// Annotate the node with the crisocket information, sourced either from the InitConfiguration struct or
	// --cri-socket.
	// TODO: In the future we want to use something more official like NodeStatus or similar for detecting this properly
	if err := patchnodephase.AnnotateCRISocket(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.CRISocket); err != nil {
		errs = append(errs, fmt.Errorf("error uploading crisocket: %v", err))
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

	// Upgrade to a self-hosted control plane if possible
	if err := upgradeToSelfHosting(client, cfg, dryRun); err != nil {
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

	// Rotate the kube-apiserver cert and key if needed
	if err := BackupAPIServerCertIfNeeded(cfg, dryRun); err != nil {
		errs = append(errs, err)
	}

	// Upgrade kube-dns/CoreDNS and kube-proxy
	if err := dns.EnsureDNSAddon(cfg, client); err != nil {
		errs = append(errs, err)
	}
	// Remove the old DNS deployment if a new DNS service is now used (kube-dns to CoreDNS or vice versa)
	if err := removeOldDNSDeploymentIfAnotherDNSIsUsed(cfg, client, dryRun); err != nil {
		errs = append(errs, err)
	}

	if err := proxy.EnsureProxyAddon(cfg, client); err != nil {
		errs = append(errs, err)
	}
	return errors.NewAggregate(errs)
}

func removeOldDNSDeploymentIfAnotherDNSIsUsed(cfg *kubeadmapi.InitConfiguration, client clientset.Interface, dryRun bool) error {
	return apiclient.TryRunCommand(func() error {
		installedDeploymentName := kubeadmconstants.KubeDNS
		deploymentToDelete := kubeadmconstants.CoreDNS

		if features.Enabled(cfg.FeatureGates, features.CoreDNS) {
			installedDeploymentName = kubeadmconstants.CoreDNS
			deploymentToDelete = kubeadmconstants.KubeDNS
		}

		// If we're dry-running, we don't need to wait for the new DNS addon to become ready
		if !dryRun {
			dnsDeployment, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Get(installedDeploymentName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if dnsDeployment.Status.ReadyReplicas == 0 {
				return fmt.Errorf("the DNS deployment isn't ready yet")
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

func upgradeToSelfHosting(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, dryRun bool) error {
	if features.Enabled(cfg.FeatureGates, features.SelfHosting) && !IsControlPlaneSelfHosted(client) {

		waiter := getWaiter(dryRun, client)

		// kubeadm will now convert the static Pod-hosted control plane into a self-hosted one
		fmt.Println("[self-hosted] Creating self-hosted control plane.")
		if err := selfhosting.CreateSelfHostedControlPlane(kubeadmconstants.GetStaticPodDirectory(), kubeadmconstants.KubernetesDir, cfg, client, waiter, dryRun); err != nil {
			return fmt.Errorf("error creating self hosted control plane: %v", err)
		}
	}
	return nil
}

// BackupAPIServerCertIfNeeded rotates the kube-apiserver certificate if older than 180 days
func BackupAPIServerCertIfNeeded(cfg *kubeadmapi.InitConfiguration, dryRun bool) error {
	certAndKeyDir := kubeadmapiv1alpha3.DefaultCertificatesDir
	shouldBackup, err := shouldBackupAPIServerCertAndKey(certAndKeyDir)
	if err != nil {
		// Don't fail the upgrade phase if failing to determine to backup kube-apiserver cert and key.
		return fmt.Errorf("[postupgrade] WARNING: failed to determine to backup kube-apiserver cert and key: %v", err)
	}

	if !shouldBackup {
		return nil
	}

	// If dry-running, just say that this would happen to the user and exit
	if dryRun {
		fmt.Println("[postupgrade] Would rotate the API server certificate and key.")
		return nil
	}

	// Don't fail the upgrade phase if failing to backup kube-apiserver cert and key, just continue rotating the cert
	// TODO: We might want to reconsider this choice.
	if err := backupAPIServerCertAndKey(certAndKeyDir); err != nil {
		fmt.Printf("[postupgrade] WARNING: failed to backup kube-apiserver cert and key: %v", err)
	}
	return certsphase.CreateCertAndKeyFilesWithCA(
		&certsphase.KubeadmCertAPIServer,
		&certsphase.KubeadmCertRootCA,
		cfg,
	)
}

func writeKubeletConfigFiles(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, newK8sVer *version.Version, dryRun bool) error {
	kubeletDir, err := getKubeletDir(dryRun)
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
			errs = append(errs, fmt.Errorf("error downloading kubelet configuration from the ConfigMap: %v", err))
		}
	}

	if dryRun { // Print what contents would be written
		dryrunutil.PrintDryRunFile(kubeadmconstants.KubeletConfigurationFileName, kubeletDir, kubeadmconstants.KubeletRunDirectory, os.Stdout)
	}

	envFilePath := filepath.Join(kubeadmconstants.KubeletRunDirectory, kubeadmconstants.KubeletEnvFileName)
	if _, err := os.Stat(envFilePath); os.IsNotExist(err) {
		// Write env file with flags for the kubelet to use. We do not need to write the --register-with-taints for the master,
		// as we handle that ourselves in the markmaster phase
		// TODO: Maybe we want to do that some time in the future, in order to remove some logic from the markmaster phase?
		if err := kubeletphase.WriteKubeletDynamicEnvFile(&cfg.NodeRegistration, cfg.FeatureGates, false, kubeletDir); err != nil {
			errs = append(errs, fmt.Errorf("error writing a dynamic environment file for the kubelet: %v", err))
		}

		if dryRun { // Print what contents would be written
			dryrunutil.PrintDryRunFile(kubeadmconstants.KubeletEnvFileName, kubeletDir, kubeadmconstants.KubeletRunDirectory, os.Stdout)
		}
	}
	return errors.NewAggregate(errs)
}

// getWaiter gets the right waiter implementation for the right occasion
// TODO: Consolidate this with what's in init.go?
func getWaiter(dryRun bool, client clientset.Interface) apiclient.Waiter {
	if dryRun {
		return dryrunutil.NewWaiter()
	}
	return apiclient.NewKubeWaiter(client, 30*time.Minute, os.Stdout)
}

// getKubeletDir gets the kubelet directory based on whether the user is dry-running this command or not.
// TODO: Consolidate this with similar funcs?
func getKubeletDir(dryRun bool) (string, error) {
	if dryRun {
		return ioutil.TempDir("", "kubeadm-upgrade-dryrun")
	}
	return kubeadmconstants.KubeletRunDirectory, nil
}

// backupAPIServerCertAndKey backups the old cert and key of kube-apiserver to a specified directory.
func backupAPIServerCertAndKey(certAndKeyDir string) error {
	subDir := filepath.Join(certAndKeyDir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		return fmt.Errorf("failed to created backup directory %s: %v", subDir, err)
	}

	filesToMove := map[string]string{
		filepath.Join(certAndKeyDir, kubeadmconstants.APIServerCertName): filepath.Join(subDir, kubeadmconstants.APIServerCertName),
		filepath.Join(certAndKeyDir, kubeadmconstants.APIServerKeyName):  filepath.Join(subDir, kubeadmconstants.APIServerKeyName),
	}
	return moveFiles(filesToMove)
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
	return fmt.Errorf("couldn't move these files: %v. Got errors: %v", files, errors.NewAggregate(errs))
}

// shouldBackupAPIServerCertAndKey checks if the cert of kube-apiserver will be expired in 180 days.
func shouldBackupAPIServerCertAndKey(certAndKeyDir string) (bool, error) {
	apiServerCert := filepath.Join(certAndKeyDir, kubeadmconstants.APIServerCertName)
	certs, err := certutil.CertsFromFile(apiServerCert)
	if err != nil {
		return false, fmt.Errorf("couldn't load the certificate file %s: %v", apiServerCert, err)
	}
	if len(certs) == 0 {
		return false, fmt.Errorf("no certificate data found")
	}

	if time.Now().Sub(certs[0].NotBefore) > expiry {
		return true, nil
	}

	return false, nil
}
