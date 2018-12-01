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

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	utilstaticpod "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	"k8s.io/kubernetes/pkg/util/initsystem"
	utilsexec "k8s.io/utils/exec"
)

// NewCmdReset returns the "kubeadm reset" command
func NewCmdReset(in io.Reader, out io.Writer) *cobra.Command {
	var certsDir string
	var criSocketPath string
	var ignorePreflightErrors []string
	var forceReset bool
	var client clientset.Interface
	kubeConfigFile := kubeadmconstants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.",
		Run: func(cmd *cobra.Command, args []string) {
			ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(ignorePreflightErrors)
			kubeadmutil.CheckErr(err)

			kubeConfigFile = cmdutil.FindExistingKubeConfig(kubeConfigFile)
			if _, err := os.Stat(kubeConfigFile); !os.IsNotExist(err) {
				client, err = getClientset(kubeConfigFile, false)
				kubeadmutil.CheckErr(err)
			}

			r, err := NewReset(in, ignorePreflightErrorsSet, forceReset, certsDir, criSocketPath)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(r.Run(out, client))
		},
	}

	options.AddIgnorePreflightErrorsFlag(cmd.PersistentFlags(), &ignorePreflightErrors)
	options.AddKubeConfigFlag(cmd.PersistentFlags(), &kubeConfigFile)

	cmd.PersistentFlags().StringVar(
		&certsDir, "cert-dir", kubeadmapiv1beta1.DefaultCertificatesDir,
		"The path to the directory where the certificates are stored. If specified, clean this directory.",
	)

	cmd.PersistentFlags().StringVar(
		&criSocketPath, "cri-socket", kubeadmapiv1beta1.DefaultCRISocket,
		"The path to the CRI socket to use with crictl when cleaning up containers.",
	)

	cmd.PersistentFlags().BoolVarP(
		&forceReset, "force", "f", false,
		"Reset the node without prompting for confirmation.",
	)

	return cmd
}

// Reset defines struct used for kubeadm reset command
type Reset struct {
	certsDir      string
	criSocketPath string
}

// NewReset instantiate Reset struct
func NewReset(in io.Reader, ignorePreflightErrors sets.String, forceReset bool, certsDir, criSocketPath string) (*Reset, error) {
	if !forceReset {
		fmt.Println("[reset] WARNING: changes made to this host by 'kubeadm init' or 'kubeadm join' will be reverted.")
		fmt.Print("[reset] are you sure you want to proceed? [y/N]: ")
		s := bufio.NewScanner(in)
		s.Scan()
		if err := s.Err(); err != nil {
			return nil, err
		}
		if strings.ToLower(s.Text()) != "y" {
			return nil, errors.New("Aborted reset operation")
		}
	}

	fmt.Println("[preflight] running pre-flight checks")
	if err := preflight.RunRootCheckOnly(ignorePreflightErrors); err != nil {
		return nil, err
	}

	return &Reset{
		certsDir:      certsDir,
		criSocketPath: criSocketPath,
	}, nil
}

// Run reverts any changes made to this host by "kubeadm init" or "kubeadm join".
func (r *Reset) Run(out io.Writer, client clientset.Interface) error {
	var dirsToClean []string
	// Only clear etcd data when using local etcd.
	etcdManifestPath := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ManifestsSubDirName, "etcd.yaml")

	klog.V(1).Infof("[reset] checking for etcd config")
	etcdDataDir, err := getEtcdDataDir(etcdManifestPath, client)
	if err == nil {
		dirsToClean = append(dirsToClean, etcdDataDir)
	} else {
		fmt.Println("[reset] no etcd config found. Assuming external etcd")
		fmt.Println("[reset] please manually reset etcd to prevent further issues")
	}

	// Try to stop the kubelet service
	klog.V(1).Infof("[reset] getting init system")
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		klog.Warningln("[reset] the kubelet service could not be stopped by kubeadm. Unable to detect a supported init system!")
		klog.Warningln("[reset] please ensure kubelet is stopped manually")
	} else {
		fmt.Println("[reset] stopping the kubelet service")
		if err := initSystem.ServiceStop("kubelet"); err != nil {
			klog.Warningf("[reset] the kubelet service could not be stopped by kubeadm: [%v]\n", err)
			klog.Warningln("[reset] please ensure kubelet is stopped manually")
		}
	}

	// Try to unmount mounted directories under kubeadmconstants.KubeletRunDirectory in order to be able to remove the kubeadmconstants.KubeletRunDirectory directory later
	fmt.Printf("[reset] unmounting mounted directories in %q\n", kubeadmconstants.KubeletRunDirectory)
	umountDirsCmd := fmt.Sprintf("awk '$2 ~ path {print $2}' path=%s /proc/mounts | xargs -r umount", kubeadmconstants.KubeletRunDirectory)

	klog.V(1).Infof("[reset] executing command %q", umountDirsCmd)
	umountOutputBytes, err := exec.Command("sh", "-c", umountDirsCmd).Output()
	if err != nil {
		klog.Errorf("[reset] failed to unmount mounted directories in %s: %s\n", kubeadmconstants.KubeletRunDirectory, string(umountOutputBytes))
	}

	klog.V(1).Info("[reset] removing Kubernetes-managed containers")
	if err := removeContainers(utilsexec.New(), r.criSocketPath); err != nil {
		klog.Errorf("[reset] failed to remove containers: %+v", err)
	}

	dirsToClean = append(dirsToClean, []string{kubeadmconstants.KubeletRunDirectory, "/etc/cni/net.d", "/var/lib/dockershim", "/var/run/kubernetes"}...)

	// Then clean contents from the stateful kubelet, etcd and cni directories
	fmt.Printf("[reset] deleting contents of stateful directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		klog.V(1).Infof("[reset] deleting content of %s", dir)
		cleanDir(dir)
	}

	// Remove contents from the config and pki directories
	klog.V(1).Infoln("[reset] removing contents from the config and pki directories")
	if r.certsDir != kubeadmapiv1beta1.DefaultCertificatesDir {
		klog.Warningf("[reset] WARNING: cleaning a non-default certificates directory: %q\n", r.certsDir)
	}
	resetConfigDir(kubeadmconstants.KubernetesDir, r.certsDir)

	// Output help text instructing user how to remove iptables rules
	msg := `
The reset process does not reset or clean up iptables rules or IPVS tables.
If you wish to reset iptables, you must do so manually.
For example: 
iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X

If your cluster was setup to utilize IPVS, run ipvsadm --clear (or similar)
to reset your system's IPVS tables.

`
	fmt.Print(msg)

	return nil
}

func getEtcdDataDir(manifestPath string, client clientset.Interface) (string, error) {
	const etcdVolumeName = "etcd-data"
	var dataDir string

	if client != nil {
		cfg, err := configutil.FetchConfigFromFileOrCluster(client, os.Stdout, "reset", "", false)
		if err == nil && cfg.Etcd.Local != nil {
			return cfg.Etcd.Local.DataDir, nil
		}
		klog.Warningf("[reset] Unable to fetch the kubeadm-config ConfigMap, using etcd pod spec as fallback: %v", err)
	}

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
		return dataDir, fmt.Errorf("invalid etcd pod manifest")
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
	if err := containerRuntime.RemoveContainers(containers); err != nil {
		return err
	}
	return nil
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
	fmt.Printf("[reset] deleting contents of config directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		if err := cleanDir(dir); err != nil {
			klog.Errorf("[reset] failed to remove directory: %q [%v]\n", dir, err)
		}
	}

	filesToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.AdminKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletBootstrapKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}
	fmt.Printf("[reset] deleting files: %v\n", filesToClean)
	for _, path := range filesToClean {
		if err := os.RemoveAll(path); err != nil {
			klog.Errorf("[reset] failed to remove file: %q [%v]\n", path, err)
		}
	}
}
