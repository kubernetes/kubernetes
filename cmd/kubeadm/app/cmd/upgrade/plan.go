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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	"k8s.io/kubernetes/pkg/util/version"
)

type planFlags struct {
	*applyPlanFlags

	newK8sVersionStr string
}

// NewCmdPlan returns the cobra command for `kubeadm upgrade plan`
func NewCmdPlan(apf *applyPlanFlags) *cobra.Command {
	flags := &planFlags{
		applyPlanFlags: apf,
	}

	cmd := &cobra.Command{
		Use:   "plan [version] [flags]",
		Short: "Check which versions are available to upgrade to and validate whether your current cluster is upgradeable. To skip the internet check, pass in the optional [version] parameter.",
		Run: func(_ *cobra.Command, args []string) {
			var err error
			flags.ignorePreflightErrorsSet, err = validation.ValidateIgnorePreflightErrors(flags.ignorePreflightErrors)
			kubeadmutil.CheckErr(err)
			// Ensure the user is root
			err = runPreflightChecks(flags.ignorePreflightErrorsSet)
			kubeadmutil.CheckErr(err)

			// If the version is specified in config file, pick up that value.
			if flags.cfgPath != "" {
				glog.V(1).Infof("fetching configuration from file %s", flags.cfgPath)
				cfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(flags.cfgPath, &kubeadmapiv1alpha3.InitConfiguration{})
				kubeadmutil.CheckErr(err)

				if cfg.KubernetesVersion != "" {
					flags.newK8sVersionStr = cfg.KubernetesVersion
				}
			}
			// If option was specified in both args and config file, args will overwrite the config file.
			if len(args) == 1 {
				flags.newK8sVersionStr = args[0]
			}

			err = RunPlan(flags)
			kubeadmutil.CheckErr(err)
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	return cmd
}

// RunPlan takes care of outputting available versions to upgrade to for the user
func RunPlan(flags *planFlags) error {
	// Start with the basics, verify that the cluster is healthy, build a client and a versionGetter. Never dry-run when planning.
	glog.V(1).Infof("[upgrade/plan] verifying health of cluster")
	glog.V(1).Infof("[upgrade/plan] retrieving configuration from cluster")
	upgradeVars, err := enforceRequirements(flags.applyPlanFlags, false, flags.newK8sVersionStr)
	if err != nil {
		return err
	}

	var etcdClient etcdutil.ClusterInterrogator

	// Currently this is the only method we have for distinguishing
	// external etcd vs static pod etcd
	isExternalEtcd := upgradeVars.cfg.Etcd.External != nil
	if isExternalEtcd {
		client, err := etcdutil.New(
			upgradeVars.cfg.Etcd.External.Endpoints,
			upgradeVars.cfg.Etcd.External.CAFile,
			upgradeVars.cfg.Etcd.External.CertFile,
			upgradeVars.cfg.Etcd.External.KeyFile)
		if err != nil {
			return err
		}
		etcdClient = client
	} else {
		client, err := etcdutil.NewFromStaticPod(
			[]string{"localhost:2379"},
			constants.GetStaticPodDirectory(),
			upgradeVars.cfg.CertificatesDir,
		)
		if err != nil {
			return err
		}
		etcdClient = client
	}

	// Compute which upgrade possibilities there are
	glog.V(1).Infof("[upgrade/plan] computing upgrade possibilities")
	availUpgrades, err := upgrade.GetAvailableUpgrades(upgradeVars.versionGetter, flags.allowExperimentalUpgrades, flags.allowRCUpgrades, etcdClient, upgradeVars.cfg.FeatureGates, upgradeVars.client)
	if err != nil {
		return fmt.Errorf("[upgrade/versions] FATAL: %v", err)
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
		case constants.CoreDNS:
			printCoreDNS = true
			coreDNSBeforeVersion = upgrade.Before.DNSVersion
		case constants.KubeDNS:
			printKubeDNS = true
			kubeDNSBeforeVersion = upgrade.Before.DNSVersion
		}

		switch upgrade.After.DNSType {
		case constants.CoreDNS:
			printCoreDNS = true
			coreDNSAfterVersion = upgrade.After.DNSVersion
		case constants.KubeDNS:
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
