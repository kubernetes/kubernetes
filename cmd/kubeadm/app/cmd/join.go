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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubechecks "k8s.io/kubernetes/cmd/kubeadm/app/checks"
	kubenode "k8s.io/kubernetes/cmd/kubeadm/app/node"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	joinDoneMsgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this machine join.
		`)
)

// NewCmdJoin returns "kubeadm join" command.
func NewCmdJoin(out io.Writer) *cobra.Command {
	cfg := &kubeadmapi.NodeConfiguration{}
	var skipChecks bool
	cmd := &cobra.Command{
		Use:   "join",
		Short: "Run this on any machine you wish to join an existing cluster.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunJoin(out, cmd, args, cfg, skipChecks)
			cmdutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().StringVar(
		&cfg.Secrets.GivenToken, "token", "",
		"(required) Shared secret used to secure bootstrap. Must match the output of 'kubeadm init'",
	)

	cmd.PersistentFlags().BoolVar(
		&skipChecks, "skip-checks", false,
		"skip checks normally run before modifying the system",
	)

	return cmd
}

// RunJoin executes worked node provisioning and tries to join an existing cluster.
func RunJoin(out io.Writer, cmd *cobra.Command, args []string, s *kubeadmapi.NodeConfiguration, skipChecks bool) error {

	if !skipChecks {
		fmt.Println("Running pre-flight checks")
		kubechecks.RunJoinNodeChecks()
	} else {
		fmt.Println("Skipping pre-flight checks")
	}

	// TODO(phase1+) this we are missing args from the help text, there should be a way to tell cobra about it
	if len(args) == 0 {
		return fmt.Errorf("<cmd/join> must specify master IP address (see --help)")
	}
	s.MasterAddresses = append(s.MasterAddresses, args...)

	ok, err := kubeadmutil.UseGivenTokenIfValid(&s.Secrets)
	if !ok {
		if err != nil {
			return fmt.Errorf("<cmd/join> %v (see --help)\n", err)
		}
		return fmt.Errorf("Must specify --token (see --help)\n")
	}

	clusterInfo, err := kubenode.RetrieveTrustedClusterInfo(s)
	if err != nil {
		return err
	}

	connectionDetails, err := kubenode.EstablishMasterConnection(s, clusterInfo)
	if err != nil {
		return err
	}

	kubeconfig, err := kubenode.PerformTLSBootstrap(connectionDetails)
	if err != nil {
		return err
	}

	err = kubeadmutil.WriteKubeconfigIfNotExists("kubelet", kubeconfig)
	if err != nil {
		return err
	}

	fmt.Fprintf(out, joinDoneMsgf)
	return nil
}
