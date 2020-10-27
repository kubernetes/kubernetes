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
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	outputapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/output"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

type planFlags struct {
	*applyPlanFlags
}

// newCmdPlan returns the cobra command for `kubeadm upgrade plan`
func newCmdPlan(apf *applyPlanFlags) *cobra.Command {
	flags := &planFlags{
		applyPlanFlags: apf,
	}

	cmd := &cobra.Command{
		Use:   "plan [version] [flags]",
		Short: "Check which versions are available to upgrade to and validate whether your current cluster is upgradeable. To skip the internet check, pass in the optional [version] parameter",
		RunE: func(_ *cobra.Command, args []string) error {
			return runPlan(flags, args)
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	return cmd
}

// runPlan takes care of outputting available versions to upgrade to for the user
func runPlan(flags *planFlags, args []string) error {
	// Start with the basics, verify that the cluster is healthy, build a client and a versionGetter. Never dry-run when planning.
	klog.V(1).Infoln("[upgrade/plan] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/plan] retrieving configuration from cluster")
	client, versionGetter, cfg, err := enforceRequirements(flags.applyPlanFlags, args, false, false)
	if err != nil {
		return err
	}

	// Currently this is the only method we have for distinguishing
	// external etcd vs static pod etcd
	isExternalEtcd := cfg.Etcd.External != nil

	// Compute which upgrade possibilities there are
	klog.V(1).Infoln("[upgrade/plan] computing upgrade possibilities")
	availUpgrades, err := upgrade.GetAvailableUpgrades(versionGetter, flags.allowExperimentalUpgrades, flags.allowRCUpgrades, isExternalEtcd, cfg.DNS.Type, client, constants.GetStaticPodDirectory())
	if err != nil {
		return errors.Wrap(err, "[upgrade/versions] FATAL")
	}

	// Fetch the current state of the component configs
	klog.V(1).Infoln("[upgrade/plan] analysing component config version states")
	configVersionStates, err := getComponentConfigVersionStates(&cfg.ClusterConfiguration, client, flags.cfgPath)
	if err != nil {
		return errors.WithMessage(err, "[upgrade/versions] FATAL")
	}

	// No upgrades available
	if len(availUpgrades) == 0 {
		klog.V(1).Infoln("[upgrade/plan] Awesome, you're up-to-date! Enjoy!")
		return nil
	}

	// Generate and print upgrade plans
	for _, up := range availUpgrades {
		plan, unstableVersionFlag, err := genUpgradePlan(&up, isExternalEtcd)
		if err != nil {
			return err
		}

		// Actually, this is needed for machine readable output only.
		// printUpgradePlan won't output the configVersionStates as it will simply print the same table several times
		// in the human readable output if it did so
		plan.ConfigVersions = configVersionStates

		printUpgradePlan(&up, plan, unstableVersionFlag, isExternalEtcd, os.Stdout)
	}

	// Finally, print the component config state table
	printComponentConfigVersionStates(configVersionStates, os.Stdout)

	return nil
}

// newComponentUpgradePlan helper creates outputapi.ComponentUpgradePlan object
func newComponentUpgradePlan(name, currentVersion, newVersion string) outputapi.ComponentUpgradePlan {
	return outputapi.ComponentUpgradePlan{
		Name:           name,
		CurrentVersion: currentVersion,
		NewVersion:     newVersion,
	}
}

// TODO There is currently no way to cleanly output upgrades that involve adding, removing, or changing components
// https://github.com/kubernetes/kubeadm/issues/810 was created to track addressing this.
func appendDNSComponent(components []outputapi.ComponentUpgradePlan, up *upgrade.Upgrade, DNSType kubeadmapi.DNSAddOnType, name string) []outputapi.ComponentUpgradePlan {
	beforeVersion, afterVersion := "", ""
	if up.Before.DNSType == DNSType {
		beforeVersion = up.Before.DNSVersion
	}
	if up.After.DNSType == DNSType {
		afterVersion = up.After.DNSVersion
	}

	if beforeVersion != "" || afterVersion != "" {
		components = append(components, newComponentUpgradePlan(name, beforeVersion, afterVersion))
	}
	return components
}

// genUpgradePlan generates output-friendly upgrade plan out of upgrade.Upgrade structure
func genUpgradePlan(up *upgrade.Upgrade, isExternalEtcd bool) (*outputapi.UpgradePlan, string, error) {
	newK8sVersion, err := version.ParseSemantic(up.After.KubeVersion)
	if err != nil {
		return nil, "", errors.Wrapf(err, "Unable to parse normalized version %q as a semantic version", up.After.KubeVersion)
	}

	unstableVersionFlag := ""
	if len(newK8sVersion.PreRelease()) != 0 {
		if strings.HasPrefix(newK8sVersion.PreRelease(), "rc") {
			unstableVersionFlag = " --allow-release-candidate-upgrades"
		} else {
			unstableVersionFlag = " --allow-experimental-upgrades"
		}
	}

	components := []outputapi.ComponentUpgradePlan{}

	if up.CanUpgradeKubelets() {
		// The map is of the form <old-version>:<node-count>. Here all the keys are put into a slice and sorted
		// in order to always get the right order. Then the map value is extracted separately
		for _, oldVersion := range sortedSliceFromStringIntMap(up.Before.KubeletVersions) {
			nodeCount := up.Before.KubeletVersions[oldVersion]
			components = append(components, newComponentUpgradePlan(constants.Kubelet, fmt.Sprintf("%d x %s", nodeCount, oldVersion), up.After.KubeVersion))
		}
	}

	components = append(components, newComponentUpgradePlan(constants.KubeAPIServer, up.Before.KubeVersion, up.After.KubeVersion))
	components = append(components, newComponentUpgradePlan(constants.KubeControllerManager, up.Before.KubeVersion, up.After.KubeVersion))
	components = append(components, newComponentUpgradePlan(constants.KubeScheduler, up.Before.KubeVersion, up.After.KubeVersion))
	components = append(components, newComponentUpgradePlan(constants.KubeProxy, up.Before.KubeVersion, up.After.KubeVersion))

	components = appendDNSComponent(components, up, kubeadmapi.CoreDNS, constants.CoreDNS)
	components = appendDNSComponent(components, up, kubeadmapi.KubeDNS, constants.KubeDNS)

	if !isExternalEtcd {
		components = append(components, newComponentUpgradePlan(constants.Etcd, up.Before.EtcdVersion, up.After.EtcdVersion))
	}

	return &outputapi.UpgradePlan{Components: components}, unstableVersionFlag, nil
}

func getComponentConfigVersionStates(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, cfgPath string) ([]outputapi.ComponentConfigVersionState, error) {
	docmap := kubeadmapi.DocumentMap{}

	if cfgPath != "" {
		bytes, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, errors.Wrapf(err, "unable to read config file %q", cfgPath)
		}

		docmap, err = kubeadmutil.SplitYAMLDocuments(bytes)
		if err != nil {
			return nil, err
		}
	}

	return componentconfigs.GetVersionStates(cfg, client, docmap)
}

// printUpgradePlan prints a UX-friendly overview of what versions are available to upgrade to
func printUpgradePlan(up *upgrade.Upgrade, plan *outputapi.UpgradePlan, unstableVersionFlag string, isExternalEtcd bool, w io.Writer) {
	// The tab writer writes to the "real" writer w
	tabw := tabwriter.NewWriter(w, 10, 4, 3, ' ', 0)

	// endOfTable helper function flashes table writer
	endOfTable := func() {
		tabw.Flush()
		fmt.Fprintln(w, "")
	}

	printHeader := true
	printManualUpgradeHeader := true
	for _, component := range plan.Components {
		if isExternalEtcd && component.Name == constants.Etcd {
			// Don't print etcd if it's external
			continue
		} else if component.Name == constants.Kubelet {
			if printManualUpgradeHeader {
				fmt.Fprintln(w, "Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':")
				fmt.Fprintln(tabw, "COMPONENT\tCURRENT\tAVAILABLE")
				fmt.Fprintf(tabw, "%s\t%s\t%s\n", component.Name, component.CurrentVersion, component.NewVersion)
				printManualUpgradeHeader = false
			} else {
				fmt.Fprintf(tabw, "%s\t%s\t%s\n", "", component.CurrentVersion, component.NewVersion)
			}
		} else {
			if printHeader {
				// End of manual upgrades table
				endOfTable()

				fmt.Fprintf(w, "Upgrade to the latest %s:\n", up.Description)
				fmt.Fprintln(w, "")
				fmt.Fprintln(tabw, "COMPONENT\tCURRENT\tAVAILABLE")
				printHeader = false
			}
			fmt.Fprintf(tabw, "%s\t%s\t%s\n", component.Name, component.CurrentVersion, component.NewVersion)
		}
	}
	// End of control plane table
	endOfTable()

	//fmt.Fprintln(w, "")
	fmt.Fprintln(w, "You can now apply the upgrade by executing the following command:")
	fmt.Fprintln(w, "")
	fmt.Fprintf(w, "\tkubeadm upgrade apply %s%s\n", up.After.KubeVersion, unstableVersionFlag)
	fmt.Fprintln(w, "")

	if up.Before.KubeadmVersion != up.After.KubeadmVersion {
		fmt.Fprintf(w, "Note: Before you can perform this upgrade, you have to update kubeadm to %s.\n", up.After.KubeadmVersion)
		fmt.Fprintln(w, "")
	}

	printLineSeparator(w)
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

func strOrDash(s string) string {
	if s != "" {
		return s
	}
	return "-"
}

func yesOrNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func printLineSeparator(w io.Writer) {
	fmt.Fprintln(w, "_____________________________________________________________________")
	fmt.Fprintln(w, "")
}

func printComponentConfigVersionStates(versionStates []outputapi.ComponentConfigVersionState, w io.Writer) {
	if len(versionStates) == 0 {
		fmt.Fprintln(w, "No information available on component configs.")
		return
	}

	fmt.Fprintln(w, dedent.Dedent(`
		The table below shows the current state of component configs as understood by this version of kubeadm.
		Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
		resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
		upgrade to is denoted in the "PREFERRED VERSION" column.
	`))

	tabw := tabwriter.NewWriter(w, 10, 4, 3, ' ', 0)
	fmt.Fprintln(tabw, "API GROUP\tCURRENT VERSION\tPREFERRED VERSION\tMANUAL UPGRADE REQUIRED")

	for _, state := range versionStates {
		fmt.Fprintf(tabw,
			"%s\t%s\t%s\t%s\n",
			state.Group,
			strOrDash(state.CurrentVersion),
			strOrDash(state.PreferredVersion),
			yesOrNo(state.ManualUpgradeRequired),
		)
	}

	tabw.Flush()
	printLineSeparator(w)
}
