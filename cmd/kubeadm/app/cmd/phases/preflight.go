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

package phases

import (
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/util/normalizer"
	utilsexec "k8s.io/utils/exec"
)

var (
	masterPreflightLongDesc = normalizer.LongDesc(`
		Run master pre-flight checks, functionally equivalent to what implemented by kubeadm init.
		` + cmdutil.AlphaDisclaimer)

	masterPreflightExample = normalizer.Examples(`
		# Run master pre-flight checks.
		kubeadm alpha phase preflight master
		`)

	nodePreflightLongDesc = normalizer.LongDesc(`
		Run node pre-flight checks, functionally equivalent to what implemented by kubeadm join.
		` + cmdutil.AlphaDisclaimer)

	nodePreflightExample = normalizer.Examples(`
		# Run node pre-flight checks.
		kubeadm alpha phase preflight node
	`)
)

// NewCmdPreFlight calls cobra.Command for preflight checks
func NewCmdPreFlight() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "preflight",
		Short: "Run pre-flight checks",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdPreFlightMaster())
	cmd.AddCommand(NewCmdPreFlightNode())
	return cmd
}

// NewCmdPreFlightMaster calls cobra.Command for master preflight checks
func NewCmdPreFlightMaster() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "master",
		Short:   "Run master pre-flight checks",
		Long:    masterPreflightLongDesc,
		Example: masterPreflightExample,
		Run: func(cmd *cobra.Command, args []string) {
			cfg := &kubeadmapi.MasterConfiguration{}
			criSocket := ""
			err := preflight.RunInitMasterChecks(utilsexec.New(), cfg, criSocket, sets.NewString())
			kubeadmutil.CheckErr(err)
		},
	}

	return cmd
}

// NewCmdPreFlightNode calls cobra.Command for node preflight checks
func NewCmdPreFlightNode() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "node",
		Short:   "Run node pre-flight checks",
		Long:    nodePreflightLongDesc,
		Example: nodePreflightExample,
		Run: func(cmd *cobra.Command, args []string) {
			cfg := &kubeadmapi.NodeConfiguration{}
			criSocket := ""
			err := preflight.RunJoinNodeChecks(utilsexec.New(), cfg, criSocket, sets.NewString())
			kubeadmutil.CheckErr(err)
		},
	}

	return cmd
}
