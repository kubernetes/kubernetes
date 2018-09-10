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
	"errors"
	"fmt"

	"github.com/spf13/cobra"

	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
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

	errorMissingConfigFlag = errors.New("the --config flag is mandatory")
)

// NewCmdPreFlight calls cobra.Command for preflight checks
func NewCmdPreFlight() *cobra.Command {
	var cfgPath string
	var ignorePreflightErrors []string

	cmd := &cobra.Command{
		Use:   "preflight",
		Short: "Run pre-flight checks",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	options.AddConfigFlag(cmd.PersistentFlags(), &cfgPath)
	options.AddIgnorePreflightErrorsFlag(cmd.PersistentFlags(), &ignorePreflightErrors)

	cmd.AddCommand(NewCmdPreFlightMaster(&cfgPath, &ignorePreflightErrors))
	cmd.AddCommand(NewCmdPreFlightNode(&cfgPath, &ignorePreflightErrors))

	return cmd
}

// NewCmdPreFlightMaster calls cobra.Command for master preflight checks
func NewCmdPreFlightMaster(cfgPath *string, ignorePreflightErrors *[]string) *cobra.Command {

	cmd := &cobra.Command{
		Use:     "master",
		Short:   "Run master pre-flight checks",
		Long:    masterPreflightLongDesc,
		Example: masterPreflightExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(*cfgPath) == 0 {
				kubeadmutil.CheckErr(errorMissingConfigFlag)
			}
			ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(*ignorePreflightErrors)
			kubeadmutil.CheckErr(err)

			cfg := &kubeadmapiv1alpha3.InitConfiguration{}
			kubeadmscheme.Scheme.Default(cfg)

			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			err = configutil.VerifyAPIServerBindAddress(internalcfg.APIEndpoint.AdvertiseAddress)
			kubeadmutil.CheckErr(err)

			fmt.Println("[preflight] running pre-flight checks")

			err = preflight.RunInitMasterChecks(utilsexec.New(), internalcfg, ignorePreflightErrorsSet)
			kubeadmutil.CheckErr(err)

			fmt.Println("[preflight] pre-flight checks passed")
		},
	}

	return cmd
}

// NewCmdPreFlightNode calls cobra.Command for node preflight checks
func NewCmdPreFlightNode(cfgPath *string, ignorePreflightErrors *[]string) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "node",
		Short:   "Run node pre-flight checks",
		Long:    nodePreflightLongDesc,
		Example: nodePreflightExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(*cfgPath) == 0 {
				kubeadmutil.CheckErr(errorMissingConfigFlag)
			}
			ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(*ignorePreflightErrors)
			kubeadmutil.CheckErr(err)

			cfg := &kubeadmapiv1alpha3.JoinConfiguration{}
			kubeadmscheme.Scheme.Default(cfg)

			internalcfg, err := configutil.NodeConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			err = configutil.VerifyAPIServerBindAddress(internalcfg.APIEndpoint.AdvertiseAddress)
			kubeadmutil.CheckErr(err)

			fmt.Println("[preflight] running pre-flight checks")

			err = preflight.RunJoinNodeChecks(utilsexec.New(), internalcfg, ignorePreflightErrorsSet)
			kubeadmutil.CheckErr(err)

			fmt.Println("[preflight] pre-flight checks passed")
		},
	}

	return cmd
}
