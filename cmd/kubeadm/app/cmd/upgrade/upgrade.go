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

package upgrade

import (
	"io"

	"github.com/spf13/cobra"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
)

// cmdUpgradeFlags holds the values for the common flags in `kubeadm upgrade`
type cmdUpgradeFlags struct {
	kubeConfigPath            string
	cfgPath                   string
	allowExperimentalUpgrades bool
	allowRCUpgrades           bool
	printConfig               bool
	skipPreFlight             bool
}

// NewCmdUpgrade returns the cobra command for `kubeadm upgrade`
func NewCmdUpgrade(out io.Writer) *cobra.Command {
	flags := &cmdUpgradeFlags{
		kubeConfigPath:            "/etc/kubernetes/admin.conf",
		cfgPath:                   "",
		allowExperimentalUpgrades: false,
		allowRCUpgrades:           false,
		printConfig:               false,
		skipPreFlight:             false,
	}

	cmd := &cobra.Command{
		Use:   "upgrade",
		Short: "Upgrade your cluster smoothly to a newer version with this command.",
		RunE:  cmdutil.SubCmdRunE("upgrade"),
	}

	cmd.PersistentFlags().StringVar(&flags.kubeConfigPath, "kubeconfig", flags.kubeConfigPath, "The KubeConfig file to use for talking to the cluster.")
	cmd.PersistentFlags().StringVar(&flags.cfgPath, "config", flags.cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental).")
	cmd.PersistentFlags().BoolVar(&flags.allowExperimentalUpgrades, "allow-experimental-upgrades", flags.allowExperimentalUpgrades, "Show unstable versions of Kubernetes as an upgrade alternative and allow upgrading to an alpha/beta/release candidate versions of Kubernetes.")
	cmd.PersistentFlags().BoolVar(&flags.allowRCUpgrades, "allow-release-candidate-upgrades", flags.allowRCUpgrades, "Show release candidate versions of Kubernetes as an upgrade alternative and allow upgrading to a release candidate versions of Kubernetes.")
	cmd.PersistentFlags().BoolVar(&flags.printConfig, "print-config", flags.printConfig, "Whether the configuration file that will be used in the upgrade should be printed or not.")
	cmd.PersistentFlags().BoolVar(&flags.skipPreFlight, "skip-preflight-checks", flags.skipPreFlight, "Skip preflight checks normally run before modifying the system")

	cmd.AddCommand(NewCmdApply(flags))
	cmd.AddCommand(NewCmdPlan(flags))

	return cmd
}
