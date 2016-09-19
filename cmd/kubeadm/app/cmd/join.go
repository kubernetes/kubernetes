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
	"net"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
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
func NewCmdJoin(out io.Writer, s *kubeadmapi.KubeadmConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "join",
		Short: "Run this on any machine you wish to join an existing cluster.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunJoin(out, cmd, args, s)
			cmdutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().StringVarP(
		&s.Secrets.GivenToken, "token", "", "",
		`Shared secret used to secure bootstrap. Must match the output of 'kubeadm init'.`,
	)

	return cmd
}

// RunJoin executes worked node provisioning and tries to join an existing cluster.
func RunJoin(out io.Writer, cmd *cobra.Command, args []string, s *kubeadmapi.KubeadmConfig) error {
	// TODO this we are missing args from the help text, there should be a way to tell cobra about it
	if len(args) == 0 {
		return fmt.Errorf("<cmd/join> must specify master IP address (see --help)")
	}
	for _, i := range args {
		addr := net.ParseIP(i) // TODO(phase1+) should allow resolvable names too
		if addr == nil {
			return fmt.Errorf("<cmd/join> failed to parse argument (%q) as an IP address", i)
		}
		s.JoinFlags.MasterAddrs = append(s.JoinFlags.MasterAddrs, addr)
	}

	ok, err := kubeadmutil.UseGivenTokenIfValid(s)
	if !ok {
		if err != nil {
			return fmt.Errorf("<cmd/join> %s (see --help)\n", err)
		}
		return fmt.Errorf("Must specify --token (see --help)\n")
	}

	kubeconfig, err := kubenode.RetrieveTrustedClusterInfo(s)
	if err != nil {
		return err
	}

	err = kubeadmutil.WriteKubeconfigIfNotExists(s, "kubelet", kubeconfig)
	if err != nil {
		return err
	}

	fmt.Fprintf(out, joinDoneMsgf)
	return nil
}
