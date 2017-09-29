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
	"fmt"
	"io"
	"os"
	"sort"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// NewCmdPlan returns the cobra command for `kubeadm upgrade plan`
func NewCmdPlan(parentFlags *cmdUpgradeFlags) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "plan",
		Short: "Check which versions are available to upgrade to and validate whether your current cluster is upgradeable",
		Run: func(_ *cobra.Command, _ []string) {
			// Ensure the user is root
			err := runPreflightChecks(parentFlags.skipPreFlight)
			kubeadmutil.CheckErr(err)

			err = RunPlan(parentFlags)
			kubeadmutil.CheckErr(err)
		},
	}

	return cmd
}

// RunPlan takes care of outputting available versions to upgrade to for the user
func RunPlan(parentFlags *cmdUpgradeFlags) error {

	// Start with the basics, verify that the cluster is healthy, build a client and a versionGetter. Never set dry-run for plan.
	upgradeVars, err := enforceRequirements(parentFlags.kubeConfigPath, parentFlags.cfgPath, parentFlags.printConfig, false)
	if err != nil {
		return err
	}

	// Compute which upgrade possibilities there are
	availUpgrades, err := upgrade.GetAvailableUpgrades(upgradeVars.versionGetter, parentFlags.allowExperimentalUpgrades, parentFlags.allowRCUpgrades)
	if err != nil {
		return fmt.Errorf("[upgrade/versions] FATAL: %v", err)
	}

	// Tell the user which upgrades are available
	printAvailableUpgrades(availUpgrades, os.Stdout)
	return nil
}

// printAvailableUpgrades prints a UX-friendly overview of what versions are available to upgrade to
// TODO look into columnize or some other formatter when time permits instead of using the tabwriter
func printAvailableUpgrades(upgrades []upgrade.Upgrade, w io.Writer) {

	// Return quickly if no upgrades can be made
	if len(upgrades) == 0 {
		fmt.Fprintln(w, "Awesome, you're up-to-date! Enjoy!")
		return
	}
	// The tab writer writes to the "real" writer w
	tabw := tabwriter.NewWriter(w, 10, 4, 3, ' ', 0)

	// Loop through the upgrade possibilities and output text to the command line
	for _, upgrade := range upgrades {

		if upgrade.CanUpgradeKubelets() {
			fmt.Fprintln(w, "Components that must be upgraded manually after you've upgraded the control plane with 'kubeadm upgrade apply':")
			fmt.Fprintln(tabw, "COMPONENT\tCURRENT\tAVAILABLE")
			firstPrinted := false

			// The map is of the form <old-version>:<node-count>. Here all the keys are put into a slice and sorted
			// in order to always get the right order. Then the map value is extracted separately
			for _, oldVersion := range sortedSliceFromStringIntMap(upgrade.Before.KubeletVersions) {
				nodeCount := upgrade.Before.KubeletVersions[oldVersion]
				if !firstPrinted {
					// Output the Kubelet header only on the first version pair
					fmt.Fprintf(tabw, "Kubelet\t%d x %s\t%s\n", nodeCount, oldVersion, upgrade.After.KubeVersion)
					firstPrinted = true
					continue
				}
				fmt.Fprintf(tabw, "\t\t%d x %s\t%s\n", nodeCount, oldVersion, upgrade.After.KubeVersion)
			}
			// We should flush the writer here at this stage; as the columns will now be of the right size, adjusted to the above content
			tabw.Flush()
			fmt.Fprintln(w, "")
		}

		fmt.Fprintf(w, "Upgrade to the latest %s:\n", upgrade.Description)
		fmt.Fprintln(w, "")
		fmt.Fprintln(tabw, "COMPONENT\tCURRENT\tAVAILABLE")
		fmt.Fprintf(tabw, "API Server\t%s\t%s\n", upgrade.Before.KubeVersion, upgrade.After.KubeVersion)
		fmt.Fprintf(tabw, "Controller Manager\t%s\t%s\n", upgrade.Before.KubeVersion, upgrade.After.KubeVersion)
		fmt.Fprintf(tabw, "Scheduler\t%s\t%s\n", upgrade.Before.KubeVersion, upgrade.After.KubeVersion)
		fmt.Fprintf(tabw, "Kube Proxy\t%s\t%s\n", upgrade.Before.KubeVersion, upgrade.After.KubeVersion)
		fmt.Fprintf(tabw, "Kube DNS\t%s\t%s\n", upgrade.Before.DNSVersion, upgrade.After.DNSVersion)

		// The tabwriter should be flushed at this stage as we have now put in all the required content for this time. This is required for the tabs' size to be correct.
		tabw.Flush()
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "You can now apply the upgrade by executing the following command:")
		fmt.Fprintln(w, "")
		fmt.Fprintf(w, "\tkubeadm upgrade apply %s\n", upgrade.After.KubeVersion)
		fmt.Fprintln(w, "")

		if upgrade.Before.KubeadmVersion != upgrade.After.KubeadmVersion {
			fmt.Fprintf(w, "Note: Before you do can perform this upgrade, you have to update kubeadm to %s\n", upgrade.After.KubeadmVersion)
			fmt.Fprintln(w, "")
		}

		fmt.Fprintln(w, "_____________________________________________________________________")
		fmt.Fprintln(w, "")
	}
}

// sortedSliceFromStringIntMap returns a slice of the keys in the map sorted alphabetically
func sortedSliceFromStringIntMap(strMap map[string]uint16) []string {
	strSlice := []string{}
	for k := range strMap {
		strSlice = append(strSlice, k)
	}
	sort.Strings(strSlice)
	return strSlice
}
