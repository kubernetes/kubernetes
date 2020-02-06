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
	"strings"
	"text/tabwriter"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
)

type planFlags struct {
	*applyPlanFlags
}

// NewCmdPlan returns the cobra command for `kubeadm upgrade plan`
func NewCmdPlan(apf *applyPlanFlags) *cobra.Command {
	flags := &planFlags{
		applyPlanFlags: apf,
	}

	cmd := &cobra.Command{
		Use:   "plan [version] [flags]",
		Short: "Check which versions are available to upgrade to and validate whether your current cluster is upgradeable. To skip the internet check, pass in the optional [version] parameter",
		RunE: func(_ *cobra.Command, args []string) error {
			userVersion, err := getK8sVersionFromUserInput(flags.applyPlanFlags, args, false)
			if err != nil {
				return err
			}

			return runPlan(flags, userVersion)
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	return cmd
}

// runPlan takes care of outputting available versions to upgrade to for the user
func runPlan(flags *planFlags, userVersion string) error {
	// Start with the basics, verify that the cluster is healthy, build a client and a versionGetter. Never dry-run when planning.
	klog.V(1).Infoln("[upgrade/plan] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/plan] retrieving configuration from cluster")
	client, versionGetter, cfg, err := enforceRequirements(flags.applyPlanFlags, false, userVersion)
	if err != nil {
		return err
	}

	var etcdClient etcdutil.ClusterInterrogator

	// Currently this is the only method we have for distinguishing
	// external etcd vs static pod etcd
	isExternalEtcd := cfg.Etcd.External != nil
	if isExternalEtcd {
		etcdClient, err = etcdutil.New(
			cfg.Etcd.External.Endpoints,
			cfg.Etcd.External.CAFile,
			cfg.Etcd.External.CertFile,
			cfg.Etcd.External.KeyFile)
	} else {
		// Connects to local/stacked etcd existing in the cluster
		etcdClient, err = etcdutil.NewFromCluster(client, cfg.CertificatesDir)
	}
	if err != nil {
		return err
	}

	// Compute which upgrade possibilities there are
	klog.V(1).Infoln("[upgrade/plan] computing upgrade possibilities")
	availUpgrades, err := upgrade.GetAvailableUpgrades(versionGetter, flags.allowExperimentalUpgrades, flags.allowRCUpgrades, etcdClient, cfg.DNS.Type, client)
	if err != nil {
		return errors.Wrap(err, "[upgrade/versions] FATAL")
	}

	// Tell the user which upgrades are available
	printAvailableUpgrades(availUpgrades, os.Stdout, isExternalEtcd)
	return nil
}

// printAvailableUpgrades prints a UX-friendly overview of what versions are available to upgrade to
// TODO look into columnize or some other formatter when time permits instead of using the tabwriter
func printAvailableUpgrades(upgrades []upgrade.Upgrade, w io.Writer, isExternalEtcd bool) {

	// Return quickly if no upgrades can be made
	if len(upgrades) == 0 {
		fmt.Fprintln(w, "Awesome, you're up-to-date! Enjoy!")
		return
	}
	// The tab writer writes to the "real" writer w
	tabw := tabwriter.NewWriter(w, 10, 4, 3, ' ', 0)

	// Loop through the upgrade possibilities and output text to the command line
	for _, upgrade := range upgrades {

		newK8sVersion, err := version.ParseSemantic(upgrade.After.KubeVersion)
		if err != nil {
			fmt.Fprintf(w, "Unable to parse normalized version %q as a semantic version\n", upgrade.After.KubeVersion)
			continue
		}

		UnstableVersionFlag := ""
		if len(newK8sVersion.PreRelease()) != 0 {
			if strings.HasPrefix(newK8sVersion.PreRelease(), "rc") {
				UnstableVersionFlag = " --allow-release-candidate-upgrades"
			} else {
				UnstableVersionFlag = " --allow-experimental-upgrades"
			}
		}

		if isExternalEtcd && upgrade.CanUpgradeEtcd() {
			fmt.Fprintln(w, "External components that should be upgraded manually before you upgrade the control plane with 'kubeadm upgrade apply':")
			fmt.Fprintln(tabw, "COMPONENT\tCURRENT\tAVAILABLE")
			fmt.Fprintf(tabw, "Etcd\t%s\t%s\n", upgrade.Before.EtcdVersion, upgrade.After.EtcdVersion)

			// We should flush the writer here at this stage; as the columns will now be of the right size, adjusted to the above content
			tabw.Flush()
			fmt.Fprintln(w, "")
		}

		if upgrade.CanUpgradeKubelets() {
			fmt.Fprintln(w, "Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':")
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
				fmt.Fprintf(tabw, "\t%d x %s\t%s\n", nodeCount, oldVersion, upgrade.After.KubeVersion)
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

		// TODO There is currently no way to cleanly output upgrades that involve adding, removing, or changing components
		// https://github.com/kubernetes/kubeadm/issues/810 was created to track addressing this.
		printCoreDNS, printKubeDNS := false, false
		coreDNSBeforeVersion, coreDNSAfterVersion, kubeDNSBeforeVersion, kubeDNSAfterVersion := "", "", "", ""

		switch upgrade.Before.DNSType {
		case kubeadmapi.CoreDNS:
			printCoreDNS = true
			coreDNSBeforeVersion = upgrade.Before.DNSVersion
		case kubeadmapi.KubeDNS:
			printKubeDNS = true
			kubeDNSBeforeVersion = upgrade.Before.DNSVersion
		}

		switch upgrade.After.DNSType {
		case kubeadmapi.CoreDNS:
			printCoreDNS = true
			coreDNSAfterVersion = upgrade.After.DNSVersion
		case kubeadmapi.KubeDNS:
			printKubeDNS = true
			kubeDNSAfterVersion = upgrade.After.DNSVersion
		}

		if printCoreDNS {
			fmt.Fprintf(tabw, "CoreDNS\t%s\t%s\n", coreDNSBeforeVersion, coreDNSAfterVersion)
		}
		if printKubeDNS {
			fmt.Fprintf(tabw, "Kube DNS\t%s\t%s\n", kubeDNSBeforeVersion, kubeDNSAfterVersion)
		}

		if !isExternalEtcd {
			fmt.Fprintf(tabw, "Etcd\t%s\t%s\n", upgrade.Before.EtcdVersion, upgrade.After.EtcdVersion)
		}

		// The tabwriter should be flushed at this stage as we have now put in all the required content for this time. This is required for the tabs' size to be correct.
		tabw.Flush()
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "You can now apply the upgrade by executing the following command:")
		fmt.Fprintln(w, "")
		fmt.Fprintf(w, "\tkubeadm upgrade apply %s%s\n", upgrade.After.KubeVersion, UnstableVersionFlag)
		fmt.Fprintln(w, "")

		if upgrade.Before.KubeadmVersion != upgrade.After.KubeadmVersion {
			fmt.Fprintf(w, "Note: Before you can perform this upgrade, you have to update kubeadm to %s.\n", upgrade.After.KubeadmVersion)
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
