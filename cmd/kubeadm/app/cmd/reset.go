/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/util/initsystem"
)

// NewCmdReset returns "kubeadm reset" command.
func NewCmdReset(out io.Writer) *cobra.Command {
	var skipPreFlight bool
	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.",
		Run: func(cmd *cobra.Command, args []string) {
			r, err := NewReset(skipPreFlight)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(r.Run(out))
		},
	}

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"skip preflight checks normally run before modifying the system",
	)

	return cmd
}

type Reset struct{}

func NewReset(skipPreFlight bool) (*Reset, error) {
	if !skipPreFlight {
		fmt.Println("Running pre-flight checks")
		err := preflight.RunResetCheck()
		if err != nil {
			return nil, &preflight.PreFlightError{Msg: err.Error()}
		}
	} else {
		fmt.Println("Skipping pre-flight checks")
	}

	return &Reset{}, nil
}

// cleanDir removes everything in a directory, but not the directory itself:
func cleanDir(path string) {
	// If the directory doesn't even exist there's nothing to do, and we do
	// not consider this an error:
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return
	}

	d, err := os.Open(path)
	if err != nil {
		fmt.Printf("failed to remove directory: [%v]\n", err)
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		fmt.Printf("failed to remove directory: [%v]\n", err)
	}
	for _, name := range names {
		err = os.RemoveAll(filepath.Join(path, name))
		if err != nil {
			fmt.Printf("failed to remove directory: [%v]\n", err)
		}
	}
}

// resetConfigDir is used to cleanup the files kubeadm writes in /etc/kubernetes/.
func resetConfigDir(configDirPath string) {
	dirsToClean := []string{
		filepath.Join(configDirPath, "manifests"),
		filepath.Join(configDirPath, "pki"),
	}
	fmt.Printf("Deleting contents of config directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		cleanDir(dir)
	}

	filesToClean := []string{
		filepath.Join(configDirPath, "admin.conf"),
		filepath.Join(configDirPath, "kubelet.conf"),
	}
	fmt.Printf("Deleting files: %v\n", filesToClean)
	for _, path := range filesToClean {
		err := os.RemoveAll(path)
		if err != nil {
			fmt.Printf("failed to remove file: [%v]\n", err)
		}
	}
}

// Run reverts any changes made to this host by "kubeadm init" or "kubeadm join".
func (r *Reset) Run(out io.Writer) error {
	serviceToStop := "kubelet"
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Printf("%v", err)
	} else {
		fmt.Printf("Stopping the %s service...\n", serviceToStop)
		initSystem.ServiceStop(serviceToStop)
	}

	fmt.Printf("Unmounting directories in /var/lib/kubelet...\n")
	// Don't check for errors here, since umount will return a non-zero exit code if there is no directories to umount
	exec.Command("sh", "-c", "cat /proc/mounts | awk '{print $2}' | grep '/var/lib/kubelet' | xargs umount").Run()

	resetConfigDir("/etc/kubernetes/")

	dirsToClean := []string{"/var/lib/kubelet", "/var/lib/etcd"}
	fmt.Printf("Deleting contents of stateful directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		cleanDir(dir)
	}

	dockerCheck := preflight.ServiceCheck{Service: "docker"}
	if warnings, errors := dockerCheck.Check(); len(warnings) == 0 && len(errors) == 0 {
		fmt.Println("Stopping all running docker containers...")
		if err := exec.Command("sh", "-c", "docker ps | grep 'k8s_' | awk '{print $1}' | xargs docker rm --force --volumes").Run(); err != nil {
			fmt.Println("failed to stop the running containers")
		}
	} else {
		fmt.Println("docker doesn't seem to be running, skipping the removal of kubernetes containers")
	}

	return nil
}
