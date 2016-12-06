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
	"path"
	"strings"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/util/initsystem"
)

// NewCmdReset returns the "kubeadm reset" command
func NewCmdReset(out io.Writer) *cobra.Command {
	var skipPreFlight, removeNode bool
	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.",
		Run: func(cmd *cobra.Command, args []string) {
			r, err := NewReset(skipPreFlight, removeNode)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(r.Run(out))
		},
	}

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"Skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().BoolVar(
		&removeNode, "remove-node", true,
		"Remove this node from the pool of nodes in this cluster",
	)

	return cmd
}

type Reset struct {
	removeNode bool
}

func NewReset(skipPreFlight, removeNode bool) (*Reset, error) {
	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks...")

		if err := preflight.RunResetCheck(); err != nil {
			return nil, &preflight.PreFlightError{Msg: err.Error()}
		}
	} else {
		fmt.Println("[preflight] Skipping pre-flight checks...")
	}

	return &Reset{
		removeNode: removeNode,
	}, nil
}

// Run reverts any changes made to this host by "kubeadm init" or "kubeadm join".
func (r *Reset) Run(out io.Writer) error {

	// Drain and maybe remove the node from the cluster
	err := drainAndRemoveNode(r.removeNode)
	if err != nil {
		fmt.Printf("[reset] Failed to cleanup node: [%v]\n", err)
	}

	serviceToStop := "kubelet"
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Printf("[reset] Failed to detect init system and stop the kubelet service: %v\n", err)
	} else {
		fmt.Printf("[reset] Stopping the %s service...\n", serviceToStop)
		if err := initSystem.ServiceStop(serviceToStop); err != nil {
			fmt.Printf("[reset] Failed to stop the %s service\n", serviceToStop)
		}
	}

	fmt.Println("[reset] Unmounting directories in /var/lib/kubelet...")
	umountDirsCmd := "cat /proc/mounts | awk '{print $2}' | grep '/var/lib/kubelet' | xargs -r umount"
	umountOutputBytes, err := exec.Command("sh", "-c", umountDirsCmd).Output()
	if err != nil {
		fmt.Printf("[reset] Failed to unmount directories in /var/lib/kubelet: %s\n", string(umountOutputBytes))
	}

	// Remove contents from the config and pki directories
	resetConfigDir(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmapi.GlobalEnvParams.HostPKIPath)

	dirsToClean := []string{"/var/lib/kubelet", "/etc/cni/net.d"}

	// Only clear etcd data when the etcd manifest is found. In case it is not found, we must assume that the user
	// provided external etcd endpoints. In that case, it is his own responsibility to reset etcd
	if _, err := os.Stat("/etc/kubernetes/manifests/etcd.json"); os.IsNotExist(err) {
		dirsToClean = append(dirsToClean, "/var/lib/etcd")
	} else {
		fmt.Printf("[reset] No etcd manifest found in %q, assuming external etcd.\n", "/etc/kubernetes/manifests/etcd.json")
	}

	fmt.Printf("[reset] Deleting contents of stateful directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		cleanDir(dir)
	}

	dockerCheck := preflight.ServiceCheck{Service: "docker"}
	if warnings, errors := dockerCheck.Check(); len(warnings) == 0 && len(errors) == 0 {
		fmt.Println("[reset] Stopping all running docker containers...")
		if err := exec.Command("sh", "-c", "docker ps | grep 'k8s_' | awk '{print $1}' | xargs -r docker rm --force --volumes").Run(); err != nil {
			fmt.Println("[reset] Failed to stop the running containers")
		}
	} else {
		fmt.Println("[reset] docker doesn't seem to be running, skipping the removal of running kubernetes containers")
	}

	return nil
}

func drainAndRemoveNode(removeNode bool) error {

	hostname, err := os.Hostname()
	if err != nil {
		return fmt.Errorf("failed to detect node hostname")
	}
	hostname = strings.ToLower(hostname)

	// TODO: Use the "native" k8s client for this once we're confident the versioned is working
	kubeConfigPath := path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "kubelet.conf")

	getNodesCmd := fmt.Sprintf("kubectl --kubeconfig %s get nodes | grep %s", kubeConfigPath, hostname)
	output, err := exec.Command("sh", "-c", getNodesCmd).Output()
	if err != nil || len(output) == 0 {
		// kubeadm shouldn't drain and/or remove the node when it doesn't exist anymore
		return nil
	}

	fmt.Printf("[reset] Draining node: %q\n", hostname)

	output, err = exec.Command("kubectl", "--kubeconfig", kubeConfigPath, "drain", hostname, "--delete-local-data", "--force", "--ignore-daemonsets").Output()
	if err != nil {
		return fmt.Errorf("failed to drain node %q [%s]", hostname, output)
	}

	if removeNode {
		fmt.Printf("[reset] Removing node: %q\n", hostname)

		output, err = exec.Command("kubectl", "--kubeconfig", kubeConfigPath, "delete", "node", hostname).Output()
		if err != nil {
			return fmt.Errorf("failed to remove node %q [%s]", hostname, output)
		}
	}

	return nil
}

// cleanDir removes everything in a directory, but not the directory itself
func cleanDir(filepath string) error {
	// If the directory doesn't even exist there's nothing to do, and we do
	// not consider this an error
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return nil
	}

	d, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		err = os.RemoveAll(path.Join(filepath, name))
		if err != nil {
			return err
		}
	}
	return nil
}

// resetConfigDir is used to cleanup the files kubeadm writes in /etc/kubernetes/.
func resetConfigDir(configPathDir, pkiPathDir string) {
	dirsToClean := []string{
		path.Join(configPathDir, "manifests"),
		pkiPathDir,
	}
	fmt.Printf("[reset] Deleting contents of config directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		err := cleanDir(dir)
		if err != nil {
			fmt.Printf("[reset] Failed to remove directory: %q [%v]\n", dir, err)
		}
	}

	filesToClean := []string{
		path.Join(configPathDir, "admin.conf"),
		path.Join(configPathDir, "kubelet.conf"),
	}
	fmt.Printf("[reset] Deleting files: %v\n", filesToClean)
	for _, path := range filesToClean {
		err := os.RemoveAll(path)
		if err != nil {
			fmt.Printf("[reset] Failed to remove file: %q [%v]\n", path, err)
		}
	}
}
