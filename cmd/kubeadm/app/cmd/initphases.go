/*
Copyright 2018 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/audit"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/utils/exec"
)

// runInitStart prints kubeadm init startup messages
func (c *initContext) runInitStart(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()

	fmt.Printf("[init] Using Kubernetes version: %s\n", cfg.KubernetesVersion)
	fmt.Printf("[init] Using Authorization modes: %v\n", cfg.AuthorizationModes)
	if cfg.CloudProvider != "" {
		fmt.Println("[init] WARNING: For cloudprovider integrations to work --cloud-provider must be set for all kubelets in the cluster.")
		fmt.Println("\t(/etc/systemd/system/kubelet.service.d/10-kubeadm.conf should be edited for this purpose)")
	}
	return nil
}

// runMasterPreflight executes prefights checks and try to start kubelet in case it's inactive
func (c *initContext) runMasterPreflight(cmd *cobra.Command, args []string) error {
	fmt.Println("[preflight] Running pre-flight checks.")
	if err := preflight.RunInitMasterChecks(exec.New(), c.MasterConfiguration(), c.ignorePreflightErrors); err != nil {
		return err
	}

	preflight.TryStartKubelet(c.ignorePreflightErrors)
	return nil
}

// buildRunCertsFor creates an anonymous RunCerts function that executes the one of the create cert logic.
// NB. By using this builder we avoid to implement a RunCerts function for each certificate
func (c *initContext) buildRunCertsFor(certFunc func(cfg *kubeadmapi.MasterConfiguration) error) func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		return certFunc(c.MasterConfiguration())
	}
}

// buildRunKubeconfigFor creates an anonymous RunKubeconfig function that executes the one of the create kubeconfig logic.
// NB. By using this builder we avoid to implement a RunKubeconfig function for each kubeconfig file
func (c *initContext) buildRunKubeconfigFor(kubeconfigFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error) func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		return kubeconfigFunc(c.KubeConfigDir(), c.MasterConfiguration())
	}
}

// runAuditPolicy Setup the AuditPolicy (either it was passed in and exists or it wasn't passed in and generate a default policy)
func (c *initContext) runAuditPolicy(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()

	if cfg.AuditPolicyConfiguration.Path != "" {
		// TODO(chuckha) ensure passed in audit policy is valid so users don't have to find the error in the api server log.
		if _, err := os.Stat(cfg.AuditPolicyConfiguration.Path); err != nil {
			return fmt.Errorf("error getting file info for audit policy file %q [%v]", cfg.AuditPolicyConfiguration.Path, err)
		}
		fmt.Println("[audit-policy] Using the existing audit policy.")
	} else {
		// TODO: provide better support for dry running > generate the file in temp dir, but create the manifest
		//       considering the file in its final location; print the file
		cfg.AuditPolicyConfiguration.Path = filepath.Join(c.KubeConfigDir(), constants.AuditPolicyDir, constants.AuditPolicyFile)
		if err := audit.CreateDefaultAuditLogPolicy(cfg.AuditPolicyConfiguration.Path); err != nil {
			return fmt.Errorf("error creating default audit policy %q [%v]", cfg.AuditPolicyConfiguration.Path, err)
		}
		fmt.Printf("[audit-policy] Wrote KubeConfig file to disk:: %q.\n", cfg.AuditPolicyConfiguration.Path)
	}
	return nil
}

// buildRunControlPlaneFor creates an anonymous RunControlplane function that executes one of the create manifest logic for control plane components.
// NB. By using this builder we avoid to implement a RunControlplane function for each controlplane manifest file
func (c *initContext) buildRunControlPlaneFor(controlplaneFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error) func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		cfg := c.MasterConfiguration()

		// Temporarily set cfg.CertificatesDir to the "real value" when writing controlplane manifests
		// This is needed for writing the right kind of manifests
		cfg.CertificatesDir = c.CertsDir()

		// Bootstrap the control plane
		if err := controlplaneFunc(c.ManifestDir(), cfg); err != nil {
			return fmt.Errorf("error creating init static pod manifest files: %v", err)
		}

		// Revert the earlier CertificatesDir assignment to the directory that can be written to
		cfg.CertificatesDir = c.CertsDirToWriteTo()
		return nil
	}
}

// runEtcd creates the manifest for a local etcd instance
func (c *initContext) runEtcd(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()

	// Temporarily set cfg.CertificatesDir to the "real value" when writing controlplane manifests
	// This is needed for writing the right kind of manifests
	cfg.CertificatesDir = c.CertsDir()

	// Add etcd static pod spec only if external etcd is not configured
	if len(cfg.Etcd.Endpoints) == 0 {
		if err := etcd.CreateLocalEtcdStaticPodManifestFile(c.ManifestDir(), cfg); err != nil {
			return fmt.Errorf("error creating local etcd static pod manifest file: %v", err)
		}
	}

	// Revert the earlier CertificatesDir assignment to the directory that can be written to
	cfg.CertificatesDir = c.CertsDirToWriteTo()
	return nil
}

// runPrintFiles prints the generated manifests files only if we're dry-running,
func (c *initContext) runPrintFiles(cmd *cobra.Command, args []string) error {
	if !c.dryRun {
		return nil
	}

	fmt.Printf("[dryrun] Wrote certificates, kubeconfig files and control plane manifests to the %q directory.\n", c.ManifestDir())
	fmt.Println("[dryrun] The certificates or kubeconfig files would not be printed due to their sensitive nature.")
	fmt.Printf("[dryrun] Please examine the %q directory for details about what would be written.\n", c.ManifestDir())

	// Print the contents of the upgraded manifests and pretend like they were in /etc/kubernetes/manifests
	files := []dryrun.FileToPrint{}
	for _, component := range constants.MasterComponents {
		realPath := constants.GetStaticPodFilepath(component, c.ManifestDir())
		outputPath := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		files = append(files, dryrun.NewFileToPrint(realPath, outputPath))
	}

	if err := dryrun.PrintDryRunFiles(files, c.out); err != nil {
		return fmt.Errorf("error printing files on dryrun: %v", err)
	}

	return nil
}

// runInitKubeletConfig writes a base kubelet configuration for dynamic kubelet configuration feature.
func (c *initContext) runInitKubeletConfig(cmd *cobra.Command, args []string) error {
	if err := kubelet.WriteInitKubeletConfigToDiskOnMaster(c.MasterConfiguration()); err != nil {
		return fmt.Errorf("error writing base kubelet configuration to disk: %v", err)
	}
	return nil
}

// runInitWait waits for the control plane start
func (c *initContext) runInitWait(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}
	waiter := c.Waiter(cfg, client, c.dryRun)

	if err := waitForAPIAndKubelet(waiter); err != nil {
		ctx := map[string]string{
			"Error":                  fmt.Sprintf("%v", err),
			"APIServerImage":         images.GetCoreImage(constants.KubeAPIServer, cfg.GetControlPlaneImageRepository(), cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			"ControllerManagerImage": images.GetCoreImage(constants.KubeControllerManager, cfg.GetControlPlaneImageRepository(), cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			"SchedulerImage":         images.GetCoreImage(constants.KubeScheduler, cfg.GetControlPlaneImageRepository(), cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			"EtcdImage":              images.GetCoreImage(constants.Etcd, cfg.ImageRepository, cfg.KubernetesVersion, cfg.Etcd.Image),
		}

		kubeletFailTemplate.Execute(c.out, ctx)

		return fmt.Errorf("couldn't initialize a Kubernetes cluster")
	}
	return nil
}

// waitForAPIAndKubelet waits primarily for the API server to come up. If that takes a long time, and the kubelet
// /healthz and /healthz/syncloop endpoints continuously are unhealthy, kubeadm will error out after a period of
// backoffing exponentially
func waitForAPIAndKubelet(waiter apiclient.Waiter) error {
	errorChan := make(chan error)

	fmt.Printf("[init] Waiting for the kubelet to boot up the control plane as Static Pods from directory %q.\n", constants.GetStaticPodDirectory())
	fmt.Println("[init] This might take a minute or longer if the control plane images have to be pulled.")

	go func(errC chan error, waiter apiclient.Waiter) {
		// This goroutine can only make kubeadm init fail. If this check succeeds, it won't do anything special
		if err := waiter.WaitForHealthyKubelet(40*time.Second, "http://localhost:10255/healthz"); err != nil {
			errC <- err
		}
	}(errorChan, waiter)

	go func(errC chan error, waiter apiclient.Waiter) {
		// This goroutine can only make kubeadm init fail. If this check succeeds, it won't do anything special
		if err := waiter.WaitForHealthyKubelet(60*time.Second, "http://localhost:10255/healthz/syncloop"); err != nil {
			errC <- err
		}
	}(errorChan, waiter)

	go func(errC chan error, waiter apiclient.Waiter) {
		// This main goroutine sends whatever WaitForAPI returns (error or not) to the channel
		// This in order to continue on success (nil error), or just fail if
		errC <- waiter.WaitForAPI()
	}(errorChan, waiter)

	// This call is blocking until one of the goroutines sends to errorChan
	return <-errorChan
}

// runUploadKubeletConfig uploads the Kubelet Config to a config map and use it as a config source for this node
func (c *initContext) runUploadKubeletConfig(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if err := kubelet.CreateBaseKubeletConfiguration(cfg, client); err != nil {
		return fmt.Errorf("error creating base kubelet configuration: %v", err)
	}
	return nil
}

// runUploadConfig Upload currently used configuration to the kubeam-config configMap in the kube-system namespace
// This is done right in the beginning of cluster initialization; as we might want to make other phases
// depend on centralized information from this source in the future
func (c *initContext) runUploadConfig(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		return fmt.Errorf("error uploading configuration: %v", err)
	}
	return nil
}

// runMarkMaster sets master label and taints on this node
func (c *initContext) runMarkMaster(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if err := markmaster.MarkMaster(client, cfg.NodeName, cfg.NoTaintMaster); err != nil {
		return fmt.Errorf("error marking master: %v", err)
	}
	return nil
}

// runCreateToken creates the default node bootstrap token
func (c *initContext) runToken(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if !c.skipTokenPrint {
		fmt.Printf("[bootstraptoken] Using token: %s\n", cfg.Token)
	}
	fmt.Println("[bootstraptoken] Bootstrap token created")

	tokenDescription := "The default bootstrap token generated by 'kubeadm init'."
	if err := node.UpdateOrCreateToken(client, cfg.Token, false, cfg.TokenTTL.Duration, cfg.TokenUsages, cfg.TokenGroups, tokenDescription); err != nil {
		return fmt.Errorf("error updating or creating token: %v", err)
	}
	return nil
}

// runClusterInfo uploads kubeam-public configMap in the kube-public namespace and
// sets RBAC rules for allowing bootstrap tokens to access this during join
func (c *initContext) runClusterInfo(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if err := clusterinfo.CreateBootstrapConfigMapIfNotExists(client, c.AdminKubeConfigPath()); err != nil {
		return fmt.Errorf("error creating bootstrap configmap: %v", err)
	}
	if err := clusterinfo.CreateClusterInfoRBACRules(client); err != nil {
		return fmt.Errorf("error creating clusterinfo RBAC rules: %v", err)
	}
	return nil
}

// runAllowPostCSR create RBAC rules that makes the bootstrap tokens able to post CSRs
// for the joining nodes.
func (c *initContext) runAllowPostCSR(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if err := node.AllowBootstrapTokensToPostCSRs(client); err != nil {
		return fmt.Errorf("error allowing bootstrap tokens to post CSRs: %v", err)
	}
	return nil
}

// runAllowAutoApproveCSR create RBAC rules that makes CSRs for the joining nodes automatically
// approved; it also creates RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
func (c *initContext) runAllowAutoApproveCSR(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}

	if err := node.AutoApproveNodeBootstrapTokens(client); err != nil {
		return fmt.Errorf("error auto-approving node bootstrap tokens: %v", err)
	}

	return node.AutoApproveNodeCertificateRotation(client)
}

// buildRunAddonFor creates an anonymous RunAddon function that executes one of the create addon logic.
// NB. By using this builder we avoid to implement a RunAddon function for each addon
func (c *initContext) buildRunAddonFor(addonFunc func(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error) func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		cfg := c.MasterConfiguration()
		client, err := c.Client(cfg, c.dryRun)
		if err != nil {
			return err
		}

		if err := addonFunc(cfg, client); err != nil {
			return fmt.Errorf("error ensuring proxy addon: %v", err)
		}
		return nil
	}
}

// runSelfhosting transform the static pod controlplane into a self hosted controlplane
func (c *initContext) runSelfhosting(cmd *cobra.Command, args []string) error {
	cfg := c.MasterConfiguration()
	client, err := c.Client(cfg, c.dryRun)
	if err != nil {
		return err
	}
	waiter := c.Waiter(cfg, client, c.dryRun)

	fmt.Println("[self-hosted] Creating self-hosted control plane.")
	if err := selfhosting.CreateSelfHostedControlPlane(c.ManifestDir(), c.KubeConfigDir(), cfg, client, waiter, c.dryRun); err != nil {
		return fmt.Errorf("error creating self hosted control plane: %v", err)
	}

	return nil
}

// runInitComplete prints the init complete message
func (c *initContext) runInitComplete(cmd *cobra.Command, args []string) error {
	adminKubeConfigPath := c.AdminKubeConfigPath()

	if c.dryRun {
		fmt.Println("[dryrun] Finished dry-running successfully. Above are the resources that would be created.")
		return nil
	}

	joinCommand, err := cmdutil.GetJoinCommand(adminKubeConfigPath, c.MasterConfiguration().Token, c.skipTokenPrint)
	if err != nil {
		return fmt.Errorf("failed to get join command: %v", err)
	}

	ctx := map[string]string{
		"KubeConfigPath": adminKubeConfigPath,
		"joinCommand":    joinCommand,
	}
	return initDoneTemplate.Execute(c.out, ctx)
}
