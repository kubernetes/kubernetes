/*
Copyright 2016 The Kubernetes Authors.

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
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	uploadconfig "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	utilstaticpod "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	"k8s.io/kubernetes/pkg/util/initsystem"
	utilsexec "k8s.io/utils/exec"
)

// resetOptions defines all the options exposed via flags by kubeadm reset.
type resetOptions struct {
	certificatesDir       string
	criSocketPath         string
	forceReset            bool
	ignorePreflightErrors []string
	kubeconfigPath        string
}

// resetData defines all the runtime information used when running the kubeadm reset worklow;
// this data is shared across all the phases that are included in the workflow.
type resetData struct {
	certificatesDir       string
	client                clientset.Interface
	criSocketPath         string
	forceReset            bool
	ignorePreflightErrors sets.String
	inputReader           io.Reader
	outputWriter          io.Writer
	cfg                   *kubeadmapi.InitConfiguration
}

// newResetOptions returns a struct ready for being used for creating cmd join flags.
func newResetOptions() *resetOptions {
	return &resetOptions{
		certificatesDir: kubeadmapiv1beta2.DefaultCertificatesDir,
		forceReset:      false,
		kubeconfigPath:  kubeadmconstants.GetAdminKubeConfigPath(),
	}
}

// newResetData returns a new resetData struct to be used for the execution of the kubeadm reset workflow.
func newResetData(cmd *cobra.Command, options *resetOptions, in io.Reader, out io.Writer) (*resetData, error) {
	var cfg *kubeadmapi.InitConfiguration
	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(options.ignorePreflightErrors)
	if err != nil {
		return nil, err
	}

	client, err := getClientset(options.kubeconfigPath, false)
	if err == nil {
		klog.V(1).Infof("[reset] Loaded client set from kubeconfig file: %s", options.kubeconfigPath)
		cfg, err = configutil.FetchInitConfigurationFromCluster(client, out, "reset", false)
		if err != nil {
			klog.Warningf("[reset] Unable to fetch the kubeadm-config ConfigMap from cluster: %v", err)
		}
	} else {
		klog.V(1).Infof("[reset] Could not obtain a client set from the kubeconfig file: %s", options.kubeconfigPath)
	}

	var criSocketPath string
	if options.criSocketPath == "" {
		criSocketPath, err = resetDetectCRISocket(cfg)
		if err != nil {
			return nil, err
		}
		klog.V(1).Infof("[reset] Detected and using CRI socket: %s", criSocketPath)
	}

	return &resetData{
		certificatesDir:       options.certificatesDir,
		client:                client,
		criSocketPath:         criSocketPath,
		forceReset:            options.forceReset,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		inputReader:           in,
		outputWriter:          out,
		cfg:                   cfg,
	}, nil
}

// AddResetFlags adds reset flags
func AddResetFlags(flagSet *flag.FlagSet, resetOptions *resetOptions) {
	flagSet.StringVar(
		&resetOptions.certificatesDir, options.CertificatesDir, resetOptions.certificatesDir,
		`The path to the directory where the certificates are stored. If specified, clean this directory.`,
	)
	flagSet.BoolVarP(
		&resetOptions.forceReset, options.ForceReset, "f", false,
		"Reset the node without prompting for confirmation.",
	)

	options.AddKubeConfigFlag(flagSet, &resetOptions.kubeconfigPath)
	options.AddIgnorePreflightErrorsFlag(flagSet, &resetOptions.ignorePreflightErrors)
	cmdutil.AddCRISocketFlag(flagSet, &resetOptions.criSocketPath)
}

// NewCmdReset returns the "kubeadm reset" command
func NewCmdReset(in io.Reader, out io.Writer, resetOptions *resetOptions) *cobra.Command {
	if resetOptions == nil {
		resetOptions = newResetOptions()
	}
	resetRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'",
		Run: func(cmd *cobra.Command, args []string) {
			c, err := resetRunner.InitData(args)
			kubeadmutil.CheckErr(err)

			err = resetRunner.Run(args)
			kubeadmutil.CheckErr(err)
			// TODO: remove this once we have all phases in place.
			// the method joinData.Run() itself should be removed too.
			data := c.(*resetData)
			kubeadmutil.CheckErr(data.Run())
		},
	}

	AddResetFlags(cmd.Flags(), resetOptions)

	// initialize the workflow runner with the list of phases
	// TODO: append phases here

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	resetRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newResetData(cmd, resetOptions, in, out)
	})

	// binds the Runner to kubeadm init command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	resetRunner.BindToCommand(cmd)

	return cmd
}

// Cfg returns the InitConfiguration.
func (r *resetData) Cfg() *kubeadmapi.InitConfiguration {
	return r.cfg
}

// CertificatesDir returns the CertificatesDir.
func (r *resetData) CertificatesDir() string {
	return r.certificatesDir
}

// Client returns the Client for accessing the cluster.
func (r *resetData) Client() clientset.Interface {
	return r.client
}

// ForceReset returns the forceReset flag.
func (r *resetData) ForceReset() bool {
	return r.forceReset
}

// InputReader returns the io.reader used to read messages.
func (r *resetData) InputReader() io.Reader {
	return r.inputReader
}

// IgnorePreflightErrors returns the list of preflight errors to ignore.
func (r *resetData) IgnorePreflightErrors() sets.String {
	return r.ignorePreflightErrors
}

func (r *resetData) preflight() error {
	if !r.ForceReset() {
		fmt.Println("[reset] WARNING: Changes made to this host by 'kubeadm init' or 'kubeadm join' will be reverted.")
		fmt.Print("[reset] Are you sure you want to proceed? [y/N]: ")
		s := bufio.NewScanner(r.InputReader())
		s.Scan()
		if err := s.Err(); err != nil {
			return err
		}
		if strings.ToLower(s.Text()) != "y" {
			return errors.New("Aborted reset operation")
		}
	}

	fmt.Println("[preflight] Running pre-flight checks")
	if err := preflight.RunRootCheckOnly(r.IgnorePreflightErrors()); err != nil {
		return err
	}

	return nil
}

// Run reverts any changes made to this host by "kubeadm init" or "kubeadm join".
func (r *resetData) Run() error {
	var dirsToClean []string
	cfg := r.Cfg()
	certsDir := r.CertificatesDir()
	client := r.Client()

	err := r.preflight()
	if err != nil {
		return err
	}

	// Reset the ClusterStatus for a given control-plane node.
	if isControlPlane() && cfg != nil {
		uploadconfig.ResetClusterStatusForNode(cfg.NodeRegistration.Name, client)
	}

	// Only clear etcd data when using local etcd.
	klog.V(1).Infoln("[reset] Checking for etcd config")
	etcdManifestPath := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ManifestsSubDirName, "etcd.yaml")
	etcdDataDir, err := getEtcdDataDir(etcdManifestPath, cfg)
	if err == nil {
		dirsToClean = append(dirsToClean, etcdDataDir)
		if cfg != nil {
			if err := etcdphase.RemoveStackedEtcdMemberFromCluster(client, cfg); err != nil {
				klog.Warningf("[reset] failed to remove etcd member: %v\n.Please manually remove this etcd member using etcdctl", err)
			}
		}
	} else {
		fmt.Println("[reset] No etcd config found. Assuming external etcd")
		fmt.Println("[reset] Please manually reset etcd to prevent further issues")
	}

	// Try to stop the kubelet service
	klog.V(1).Infoln("[reset] Getting init system")
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		klog.Warningln("[reset] The kubelet service could not be stopped by kubeadm. Unable to detect a supported init system!")
		klog.Warningln("[reset] Please ensure kubelet is stopped manually")
	} else {
		fmt.Println("[reset] Stopping the kubelet service")
		if err := initSystem.ServiceStop("kubelet"); err != nil {
			klog.Warningf("[reset] The kubelet service could not be stopped by kubeadm: [%v]\n", err)
			klog.Warningln("[reset] Please ensure kubelet is stopped manually")
		}
	}

	// Try to unmount mounted directories under kubeadmconstants.KubeletRunDirectory in order to be able to remove the kubeadmconstants.KubeletRunDirectory directory later
	fmt.Printf("[reset] Unmounting mounted directories in %q\n", kubeadmconstants.KubeletRunDirectory)
	umountDirsCmd := fmt.Sprintf("awk '$2 ~ path {print $2}' path=%s/ /proc/mounts | xargs -r umount", kubeadmconstants.KubeletRunDirectory)

	klog.V(1).Infof("[reset] Executing command %q", umountDirsCmd)
	umountOutputBytes, err := exec.Command("sh", "-c", umountDirsCmd).Output()
	if err != nil {
		klog.Errorf("[reset] Failed to unmount mounted directories in %s: %s\n", kubeadmconstants.KubeletRunDirectory, string(umountOutputBytes))
	}

	klog.V(1).Info("[reset] Removing Kubernetes-managed containers")
	if err := removeContainers(utilsexec.New(), r.criSocketPath); err != nil {
		klog.Errorf("[reset] Failed to remove containers: %v", err)
	}

	dirsToClean = append(dirsToClean, []string{kubeadmconstants.KubeletRunDirectory, "/etc/cni/net.d", "/var/lib/dockershim", "/var/run/kubernetes"}...)

	// Then clean contents from the stateful kubelet, etcd and cni directories
	fmt.Printf("[reset] Deleting contents of stateful directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		klog.V(1).Infof("[reset] Deleting content of %s", dir)
		cleanDir(dir)
	}

	// Remove contents from the config and pki directories
	klog.V(1).Infoln("[reset] Removing contents from the config and pki directories")
	if certsDir != kubeadmapiv1beta2.DefaultCertificatesDir {
		klog.Warningf("[reset] WARNING: Cleaning a non-default certificates directory: %q\n", certsDir)
	}
	resetConfigDir(kubeadmconstants.KubernetesDir, certsDir)

	// Output help text instructing user how to remove iptables rules
	msg := dedent.Dedent(`
		The reset process does not reset or clean up iptables rules or IPVS tables.
		If you wish to reset iptables, you must do so manually.
		For example:
		iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X

		If your cluster was setup to utilize IPVS, run ipvsadm --clear (or similar)
		to reset your system's IPVS tables.
		
		The reset process does not clean your kubeconfig files and you must remove them manually.
		Please, check the contents of the $HOME/.kube/config file.
	`)
	fmt.Print(msg)

	return nil
}

func getEtcdDataDir(manifestPath string, cfg *kubeadmapi.InitConfiguration) (string, error) {
	const etcdVolumeName = "etcd-data"
	var dataDir string

	if cfg != nil && cfg.Etcd.Local != nil {
		return cfg.Etcd.Local.DataDir, nil
	}
	klog.Warningln("[reset] No kubeadm config, using etcd pod spec to get data directory")

	etcdPod, err := utilstaticpod.ReadStaticPodFromDisk(manifestPath)
	if err != nil {
		return "", err
	}

	for _, volumeMount := range etcdPod.Spec.Volumes {
		if volumeMount.Name == etcdVolumeName {
			dataDir = volumeMount.HostPath.Path
			break
		}
	}
	if dataDir == "" {
		return dataDir, errors.New("invalid etcd pod manifest")
	}
	return dataDir, nil
}

func removeContainers(execer utilsexec.Interface, criSocketPath string) error {
	containerRuntime, err := utilruntime.NewContainerRuntime(execer, criSocketPath)
	if err != nil {
		return err
	}
	containers, err := containerRuntime.ListKubeContainers()
	if err != nil {
		return err
	}
	return containerRuntime.RemoveContainers(containers)
}

// cleanDir removes everything in a directory, but not the directory itself
func cleanDir(filePath string) error {
	// If the directory doesn't even exist there's nothing to do, and we do
	// not consider this an error
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil
	}

	d, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		if err = os.RemoveAll(filepath.Join(filePath, name)); err != nil {
			return err
		}
	}
	return nil
}

// resetConfigDir is used to cleanup the files kubeadm writes in /etc/kubernetes/.
func resetConfigDir(configPathDir, pkiPathDir string) {
	dirsToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.ManifestsSubDirName),
		pkiPathDir,
	}
	fmt.Printf("[reset] Deleting contents of config directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		if err := cleanDir(dir); err != nil {
			klog.Errorf("[reset] Failed to remove directory: %q [%v]\n", dir, err)
		}
	}

	filesToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.AdminKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletBootstrapKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}
	fmt.Printf("[reset] Deleting files: %v\n", filesToClean)
	for _, path := range filesToClean {
		if err := os.RemoveAll(path); err != nil {
			klog.Errorf("[reset] Failed to remove file: %q [%v]\n", path, err)
		}
	}
}

func resetDetectCRISocket(cfg *kubeadmapi.InitConfiguration) (string, error) {
	if cfg != nil {
		// first try to get the CRI socket from the cluster configuration
		return cfg.NodeRegistration.CRISocket, nil
	}

	// if this fails, try to detect it
	return utilruntime.DetectCRISocket()
}

// isControlPlane checks if a node is a control-plane node by looking up
// the kube-apiserver manifest file
func isControlPlane() bool {
	filepath := kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.KubeAPIServer, kubeadmconstants.GetStaticPodDirectory())
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return false
	}
	return true
}
