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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	outputapiv1alpha1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

type planFlags struct {
	*applyPlanFlags
}

// NewCmdPlan returns the cobra command for `kubeadm upgrade plan`
func NewCmdPlan(apf *applyPlanFlags) *cobra.Command {
	flags := &planFlags{
		applyPlanFlags: apf,
	}

	outputFlags := newUpgradePlanPrintFlags(output.TextOutput)

	cmd := &cobra.Command{
		Use:   "plan [version] [flags]",
		Short: "Check which versions are available to upgrade to and validate whether your current cluster is upgradeable. To skip the internet check, pass in the optional [version] parameter",
		RunE: func(_ *cobra.Command, args []string) error {
			userVersion, err := getK8sVersionFromUserInput(flags.applyPlanFlags, args, false)
			if err != nil {
				return err
			}

			printer, err := outputFlags.ToPrinter()
			if err != nil {
				return err
			}

			return runPlan(flags, userVersion, printer)
		},
	}

	outputFlags.AddFlags(cmd)

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	return cmd
}

// newComponentUpgradePlan helper creates outputapiv1alpha1.ComponentUpgradePlan object
func newComponentUpgradePlan(name, currentVersion, newVersion string) *outputapiv1alpha1.ComponentUpgradePlan {
	return &outputapiv1alpha1.ComponentUpgradePlan{
		Name:           name,
		CurrentVersion: currentVersion,
		NewVersion:     newVersion,
	}
}

// upgradePlanPrintFlags defines printer flag structure for the
// upgrade plan kubeadm command and provides a method
// of retrieving a known printer based on flag values provided.
type upgradePlanPrintFlags struct {
	// JSONYamlPrintFlags provides default flags necessary for json/yaml printing.
	JSONYamlPrintFlags *upgradePlanJSONYamlPrintFlags
	// TextPrintFlags provides default flags necessary for text printing.
	TextPrintFlags *upgradePlanTextPrintFlags
	// TypeSetterPrinter is an implementation of ResourcePrinter that wraps another printer with types set on the objects
	TypeSetterPrinter *printers.TypeSetterPrinter
	// OutputFormat contains currently set output format
	OutputFormat string
}

func newUpgradePlanPrintFlags(outputFormat string) *upgradePlanPrintFlags {
	return &upgradePlanPrintFlags{
		JSONYamlPrintFlags: &upgradePlanJSONYamlPrintFlags{},
		TextPrintFlags:     &upgradePlanTextPrintFlags{},
		TypeSetterPrinter:  printers.NewTypeSetter(outputapischeme.Scheme),
		OutputFormat:       strings.ToLower(outputFormat),
	}
}

// AllowedFormats returns list of allowed output formats
func (pf *upgradePlanPrintFlags) AllowedFormats() []string {
	ret := pf.TextPrintFlags.AllowedFormats()
	return append(ret, pf.JSONYamlPrintFlags.AllowedFormats()...)
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to Kubeadm printing to it
func (pf *upgradePlanPrintFlags) AddFlags(cmd *cobra.Command) {
	pf.TextPrintFlags.AddFlags(cmd)
	pf.JSONYamlPrintFlags.AddFlags(cmd)
	cmd.Flags().StringVarP(&pf.OutputFormat, "experimental-output", "o", pf.OutputFormat, fmt.Sprintf("Output format. One of: %s.", strings.Join(pf.AllowedFormats(), "|")))
}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling format printing.
// Returns error if the specified outputFormat does not match supported formats.
func (pf *upgradePlanPrintFlags) ToPrinter() (output.Printer, error) {
	switch pf.OutputFormat {
	case output.TextOutput:
		return pf.TextPrintFlags.ToPrinter(pf.OutputFormat)
	case output.JSONOutput:
		return newUpgradePlanJSONYAMLPrinter(pf.TypeSetterPrinter.WrapToPrinter(pf.JSONYamlPrintFlags.ToPrinter(output.JSONOutput)))
	case output.YAMLOutput:
		return newUpgradePlanJSONYAMLPrinter(pf.TypeSetterPrinter.WrapToPrinter(pf.JSONYamlPrintFlags.ToPrinter(output.YAMLOutput)))
	default:
		return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &pf.OutputFormat, AllowedFormats: pf.AllowedFormats()}
	}
}

type upgradePlanJSONYamlPrintFlags struct {
	genericclioptions.JSONYamlPrintFlags
}

// AllowedFormats returns list of allowed output formats
func (pf *upgradePlanJSONYamlPrintFlags) AllowedFormats() []string {
	return []string{output.JSONOutput, output.YAMLOutput}
}

// upgradePlanJSONYAMLPrinter prints upgrade plan in a JSON or YAML format
type upgradePlanJSONYAMLPrinter struct {
	output.ResourcePrinterWrapper
	Buffer []outputapiv1alpha1.ComponentUpgradePlan
}

// newUpgradePlanJSONYAMLPrinter creates new upgradePlanJSONYAMLPrinter object
func newUpgradePlanJSONYAMLPrinter(resourcePrinter printers.ResourcePrinter, err error) (output.Printer, error) {
	if err != nil {
		return nil, err
	}
	return &upgradePlanJSONYAMLPrinter{ResourcePrinterWrapper: output.ResourcePrinterWrapper{Printer: resourcePrinter}}, nil
}

// PrintObj is an implementation of ResourcePrinter.PrintObj that adds object to the buffer
func (p *upgradePlanJSONYAMLPrinter) PrintObj(obj runtime.Object, writer io.Writer) error {
	item, ok := obj.(*outputapiv1alpha1.ComponentUpgradePlan)
	if !ok {
		return fmt.Errorf("expected ComponentUpgradePlan, but got %+v", obj)
	}
	p.Buffer = append(p.Buffer, *item)
	return nil
}

// Close writes any buffered data and empties list of buffered components
func (p *upgradePlanJSONYAMLPrinter) Close(writer io.Writer) {
	plan := &outputapiv1alpha1.UpgradePlan{Components: p.Buffer}
	p.Printer.PrintObj(plan, writer)
	p.Buffer = []outputapiv1alpha1.ComponentUpgradePlan{}
}

// upgradePlanTextPrinter prints upgrade plan in a text form
type upgradePlanTextPrinter struct {
	output.TextPrinter
	columns   []string
	tabwriter *tabwriter.Writer
}

// Flush writes any buffered data
func (p *upgradePlanTextPrinter) Flush(writer io.Writer) {
	if p.tabwriter != nil {
		p.tabwriter.Flush()
		p.tabwriter = nil
	}
}

// PrintObj is an implementation of ResourcePrinter.PrintObj for upgrade plan plain text output
func (p *upgradePlanTextPrinter) PrintObj(obj runtime.Object, writer io.Writer) error {
	if p.tabwriter == nil {
		p.tabwriter = tabwriter.NewWriter(writer, 10, 4, 3, ' ', 0)
		// Print header
		fmt.Fprintln(p.tabwriter, strings.Join(p.columns, "\t"))
	}

	item, ok := obj.(*outputapiv1alpha1.ComponentUpgradePlan)
	if !ok {
		return fmt.Errorf("expected ComponentUpgradePlan, but got %+v", obj)
	}

	// Print item
	fmt.Fprintf(p.tabwriter, "%s\t%s\t%s\n", item.Name, item.CurrentVersion, item.NewVersion)

	return nil
}

// upgradePlanTextPrintFlags provides flags necessary for printing upgrade plan in a text form.
type upgradePlanTextPrintFlags struct{}

func (pf *upgradePlanTextPrintFlags) AddFlags(cmd *cobra.Command) {}

// AllowedFormats returns list of allowed output formats
func (pf *upgradePlanTextPrintFlags) AllowedFormats() []string {
	return []string{output.TextOutput}
}

// ToPrinter returns kubeadm printer for the text output format
func (pf *upgradePlanTextPrintFlags) ToPrinter(outputFormat string) (output.Printer, error) {
	if outputFormat == output.TextOutput {
		return &upgradePlanTextPrinter{columns: []string{"COMPONENT", "CURRENT", "AVAILABLE"}}, nil
	}
	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: []string{output.TextOutput}}
}

// runPlan takes care of outputting available versions to upgrade to for the user
func runPlan(flags *planFlags, userVersion string, printer output.Printer) error {
	// Start with the basics, verify that the cluster is healthy, build a client and a versionGetter. Never dry-run when planning.
	klog.V(1).Infoln("[upgrade/plan] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/plan] retrieving configuration from cluster")
	client, versionGetter, cfg, err := enforceRequirements(flags.applyPlanFlags, false, userVersion, printer)
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
	klog.V(1).Infoln("[upgrade/plan] Fetching available versions to upgrade to")
	availUpgrades, err := upgrade.GetAvailableUpgrades(versionGetter, flags.allowExperimentalUpgrades, flags.allowRCUpgrades, etcdClient, cfg.DNS.Type, client)
	if err != nil {
		return errors.Wrap(err, "[upgrade/versions] FATAL")
	}
	// Add a newline in the end of this output to leave some space to the next output section
	klog.V(1).Infoln()

	// Tell the user which upgrades are available
	return printAvailableUpgrades(availUpgrades, os.Stdout, isExternalEtcd, printer)
}

// printAvailableUpgrades prints a UX-friendly overview of what versions are available to upgrade to
func printAvailableUpgrades(upgrades []upgrade.Upgrade, writer io.Writer, isExternalEtcd bool, printer output.Printer) error {
	// Return quickly if no upgrades can be made
	if len(upgrades) == 0 {
		printer.Fprintf(writer, "Awesome, you're up-to-date! Enjoy!\n")
		return nil
	}

	// Loop through the upgrade possibilities
	for _, upgrade := range upgrades {
		newK8sVersion, err := version.ParseSemantic(upgrade.After.KubeVersion)
		if err != nil {
			printer.Fprintf(writer, "Unable to parse normalized version %q as a semantic version\n", upgrade.After.KubeVersion)
			continue
		}

		if isExternalEtcd && upgrade.CanUpgradeEtcd() {
			printer.Fprintf(writer, "\nExternal components that should be upgraded manually before you upgrade the control plane with 'kubeadm upgrade apply':\n\n")
			printer.PrintObj(newComponentUpgradePlan(constants.Etcd, upgrade.Before.EtcdVersion, upgrade.After.EtcdVersion), writer)
			printer.Flush(writer)
			printer.Fprintf(writer, "\n")
		}

		if upgrade.CanUpgradeKubelets() {
			printer.Fprintf(writer, "\nComponents that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':\n\n")

			name := constants.Kubelet
			for _, oldVersion := range sortedSliceFromStringIntMap(upgrade.Before.KubeletVersions) {
				nodeCount := upgrade.Before.KubeletVersions[oldVersion]
				item := newComponentUpgradePlan(name, fmt.Sprintf("%d x %s", nodeCount, oldVersion), upgrade.After.KubeVersion)
				err = printer.PrintObj(item, writer)
				if err != nil {
					return errors.Wrapf(err, "Cannot print upgrade plan item %v", item)
				}
				name = ""
			}
		}

		printer.Flush(writer)
		printer.Fprintf(writer, "\nUpgrade to the latest %s:\n\n", upgrade.Description)

		items := []*outputapiv1alpha1.ComponentUpgradePlan{
			newComponentUpgradePlan(constants.KubeAPIServer, upgrade.Before.KubeVersion, upgrade.After.KubeVersion),
			newComponentUpgradePlan(constants.KubeControllerManager, upgrade.Before.KubeVersion, upgrade.After.KubeVersion),
			newComponentUpgradePlan(constants.KubeScheduler, upgrade.Before.KubeVersion, upgrade.After.KubeVersion),
			newComponentUpgradePlan(constants.KubeProxy, upgrade.Before.KubeVersion, upgrade.After.KubeVersion),
		}

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
			items = append(items, newComponentUpgradePlan(constants.CoreDNS, coreDNSBeforeVersion, coreDNSAfterVersion))
		}
		if printKubeDNS {
			items = append(items, newComponentUpgradePlan(constants.KubeDNS, kubeDNSBeforeVersion, kubeDNSAfterVersion))
		}

		if !isExternalEtcd {
			items = append(items, newComponentUpgradePlan(constants.Etcd, upgrade.Before.EtcdVersion, upgrade.After.EtcdVersion))
		}

		UnstableVersionFlag := ""
		if len(newK8sVersion.PreRelease()) != 0 {
			if strings.HasPrefix(newK8sVersion.PreRelease(), "rc") {
				UnstableVersionFlag = " --allow-release-candidate-upgrades"
			} else {
				UnstableVersionFlag = " --allow-experimental-upgrades"
			}
		}

		for _, item := range items {
			err = printer.PrintObj(item, writer)
			if err != nil {
				return errors.Wrapf(err, "Cannot print upgrade plan item %v", item)
			}
		}
		printer.Flush(writer)

		printer.Fprintf(writer, "\nYou can now apply the upgrade by executing the following command:\n\n")
		printer.Fprintf(writer, "\tkubeadm upgrade apply %s%s\n\n", upgrade.After.KubeVersion, UnstableVersionFlag)

		if upgrade.Before.KubeadmVersion != upgrade.After.KubeadmVersion {
			printer.Fprintf(writer, "Note: Before you can perform this upgrade, you have to update kubeadm to %s.\n\n", upgrade.After.KubeadmVersion)
		}
		printer.Fprintf(writer, "_____________________________________________________________________\n\n")
	}

	printer.Close(writer)

	return nil
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
