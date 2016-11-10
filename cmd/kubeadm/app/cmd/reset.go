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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/initsystem"
)

func NewCmdReset(out io.Writer) *cobra.Command {
	var skipPreFlight bool
	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Revert the actions kubeadm init or join made to the machine",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunReset(out, cmd, skipPreFlight)
			cmdutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"skip preflight checks normally run before modifying the system",
	)

	return cmd
}

func RunReset(out io.Writer, cmd *cobra.Command, skipPreFlight bool) error {
	if !skipPreFlight {
		fmt.Println("Running pre-flight checks")
		err := preflight.RunResetCheck()
		if err != nil {
			return &preflight.PreFlightError{Msg: err.Error()}
		}
	} else {
		fmt.Println("Skipping pre-flight checks")
	}

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

	dirsToRemove := []string{"/var/lib/kubelet", "/var/lib/etcd", "/etc/kubernetes"}
	fmt.Printf("Deleting the stateful directories: %v\n", dirsToRemove)
	for _, dir := range dirsToRemove {
		err := os.RemoveAll(dir)
		if err != nil {
			fmt.Printf("failed to remove directory: [%v]\n", err)
		}
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
