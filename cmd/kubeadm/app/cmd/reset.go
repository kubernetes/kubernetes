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
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/spf13/cobra"

	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/util/initsystem"
)

// NewCmdReset returns the "kubeadm reset" command
func NewCmdReset(out io.Writer) *cobra.Command {
	var skipPreFlight bool
	var certsDir string
	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.",
		Run: func(cmd *cobra.Command, args []string) {
			r, err := NewReset(skipPreFlight, certsDir)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(r.Run(out))
		},
	}

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"Skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().StringVar(
		&certsDir, "cert-dir", kubeadmapiext.DefaultCertificatesDir,
		"The path to the directory where the certificates are stored. If specified, clean this directory.",
	)

	return cmd
}

type Reset struct {
	certsDir string
}

func NewReset(skipPreFlight bool, certsDir string) (*Reset, error) {
	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks")

		if err := preflight.RunRootCheckOnly(); err != nil {
			return nil, err
		}
	} else {
		fmt.Println("[preflight] Skipping pre-flight checks")
	}

	return &Reset{
		certsDir: certsDir,
	}, nil
}

// Run reverts any changes made to this host by "kubeadm init" or "kubeadm join".
func (r *Reset) Run(out io.Writer) error {

	// Try to stop the kubelet service
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Println("[reset] WARNING: The kubelet service couldn't be stopped by kubeadm because no supported init system was detected.")
		fmt.Println("[reset] WARNING: Please ensure kubelet is stopped manually.")
	} else {
		fmt.Println("[reset] Stopping the kubelet service")
		if err := initSystem.ServiceStop("kubelet"); err != nil {
			fmt.Printf("[reset] WARNING: The kubelet service couldn't be stopped by kubeadm: [%v]\n", err)
			fmt.Println("[reset] WARNING: Please ensure kubelet is stopped manually.")
		}
	}

	// Try to unmount mounted directories under /var/lib/kubelet in order to be able to remove the /var/lib/kubelet directory later
	fmt.Printf("[reset] Unmounting mounted directories in %q\n", "/var/lib/kubelet")
	umountDirsCmd := "awk '$2 ~ path {print $2}' path=/var/lib/kubelet /proc/mounts | xargs -r umount"
	umountOutputBytes, err := exec.Command("sh", "-c", umountDirsCmd).Output()
	if err != nil {
		fmt.Printf("[reset] Failed to unmount mounted directories in /var/lib/kubelet: %s\n", string(umountOutputBytes))
	}

	dockerCheck := preflight.ServiceCheck{Service: "docker", CheckIfActive: true}
	if _, errors := dockerCheck.Check(); len(errors) == 0 {
		fmt.Println("[reset] Removing kubernetes-managed containers")
		if err := exec.Command("sh", "-c", "docker ps -a --filter name=k8s_ -q | xargs -r docker rm --force --volumes").Run(); err != nil {
			fmt.Println("[reset] Failed to stop the running containers")
		}
	} else {
		fmt.Println("[reset] docker doesn't seem to be running, skipping the removal of running kubernetes containers")
	}

	dirsToClean := []string{"/var/lib/kubelet", "/etc/cni/net.d", "/var/lib/dockershim"}

	// Only clear etcd data when the etcd manifest is found. In case it is not found, we must assume that the user
	// provided external etcd endpoints. In that case, it is his own responsibility to reset etcd
	etcdManifestPath := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ManifestsSubDirName, "etcd.yaml")
	if _, err := os.Stat(etcdManifestPath); err == nil {
		dirsToClean = append(dirsToClean, "/var/lib/etcd")
	} else {
		fmt.Printf("[reset] No etcd manifest found in %q, assuming external etcd.\n", etcdManifestPath)
	}

	// Then clean contents from the stateful kubelet, etcd and cni directories
	fmt.Printf("[reset] Deleting contents of stateful directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		cleanDir(dir)
	}

	// Remove contents from the config and pki directories
	resetConfigDir(kubeadmconstants.KubernetesDir, r.certsDir)

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
	fmt.Printf("[reset] Deleting contents of config directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		if err := cleanDir(dir); err != nil {
			fmt.Printf("[reset] Failed to remove directory: %q [%v]\n", dir, err)
		}
	}

	filesToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.AdminKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}
	fmt.Printf("[reset] Deleting files: %v\n", filesToClean)
	for _, path := range filesToClean {
		if err := os.RemoveAll(path); err != nil {
			fmt.Printf("[reset] Failed to remove file: %q [%v]\n", path, err)
		}
	}
}
