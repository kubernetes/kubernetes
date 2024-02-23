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

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/klog/v2"

	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	outputapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/v1alpha2"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

type planFlags struct {
	*applyPlanFlags
}

var upgradePlanLongDesc = cmdutil.LongDesc(`
	Check which versions are available to upgrade to and validate whether your current cluster is upgradeable.
	This command can only run on the control plane nodes where the kubeconfig file "admin.conf" exists.
	To skip the internet check, pass in the optional [version] parameter.
`)

// newCmdPlan returns the cobra command for `kubeadm upgrade plan`
func newCmdPlan(apf *applyPlanFlags) *cobra.Command {
	flags := &planFlags{
		applyPlanFlags: apf,
	}

	outputFlags := newUpgradePlanPrintFlags(output.TextOutput)

	cmd := &cobra.Command{
		Use:   "plan [version] [flags]",
		Short: "Check which versions are available to upgrade to and validate whether your current cluster is upgradeable.",
		Long:  upgradePlanLongDesc,
		RunE: func(_ *cobra.Command, args []string) error {
			printer, err := outputFlags.ToPrinter()
			if err != nil {
				return errors.Wrap(err, "could not construct output printer")
			}

			return runPlan(flags, args, printer)
		},
	}

	outputFlags.AddFlags(cmd)

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	return cmd
}

// newComponentUpgradePlan helper creates outputapiv1alpha2.ComponentUpgradePlan object
func newComponentUpgradePlan(name, currentVersion, newVersion string) outputapiv1alpha2.ComponentUpgradePlan {
	return outputapiv1alpha2.ComponentUpgradePlan{
		Name:           name,
		CurrentVersion: currentVersion,
		NewVersion:     newVersion,
	}
}

// upgradePlanPrintFlags defines a printer flag structure for the
// upgrade plan kubeadm command and provides a method
// of retrieving a known printer based on flag values provided.
type upgradePlanPrintFlags struct {
	// JSONYamlPrintFlags provides default flags necessary for json/yaml printing
	JSONYamlPrintFlags *upgradePlanJSONYamlPrintFlags
	// TextPrintFlags provides default flags necessary for text printing
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

// AllowedFormats returns a list of allowed output formats
func (pf *upgradePlanPrintFlags) AllowedFormats() []string {
	ret := pf.TextPrintFlags.AllowedFormats()
	return append(ret, pf.JSONYamlPrintFlags.AllowedFormats()...)
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to Kubeadm printing to it
func (pf *upgradePlanPrintFlags) AddFlags(cmd *cobra.Command) {
	pf.TextPrintFlags.AddFlags(cmd)
	pf.JSONYamlPrintFlags.AddFlags(cmd)
	// TODO: once we are confident the feature is graduated we should remove the EXPERIMENTAL text below:
	// https://github.com/kubernetes/kubeadm/issues/494
	cmd.Flags().StringVarP(&pf.OutputFormat, "output", "o", pf.OutputFormat, fmt.Sprintf("EXPERIMENTAL: Output format. One of: %s.", strings.Join(pf.AllowedFormats(), "|")))
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

// AllowedFormats returns a list of allowed output formats
func (pf *upgradePlanJSONYamlPrintFlags) AllowedFormats() []string {
	return []string{output.JSONOutput, output.YAMLOutput}
}

// upgradePlanJSONYAMLPrinter prints upgrade plan in a JSON or YAML format
type upgradePlanJSONYAMLPrinter struct {
	output.ResourcePrinterWrapper
	Components     []outputapiv1alpha2.ComponentUpgradePlan
	ConfigVersions []outputapiv1alpha2.ComponentConfigVersionState
}

// newUpgradePlanJSONYAMLPrinter creates a new upgradePlanJSONYAMLPrinter object
func newUpgradePlanJSONYAMLPrinter(resourcePrinter printers.ResourcePrinter, err error) (output.Printer, error) {
	if err != nil {
		return nil, err
	}
	return &upgradePlanJSONYAMLPrinter{ResourcePrinterWrapper: output.ResourcePrinterWrapper{Printer: resourcePrinter}}, nil
}

// PrintObj is an implementation of ResourcePrinter.PrintObj that adds object to the buffer
func (p *upgradePlanJSONYAMLPrinter) PrintObj(obj runtime.Object, writer io.Writer) error {
	item, ok := obj.(*outputapiv1alpha2.ComponentUpgradePlan)
	if !ok {
		return errors.Errorf("expected ComponentUpgradePlan, but got %+v", obj)
	}
	p.Components = append(p.Components, *item)
	return nil
}

// Flush writes any buffered data once last object is added
func (p *upgradePlanJSONYAMLPrinter) Flush(writer io.Writer, last bool) {
	if !last {
		return
	}
	if len(p.Components) == 0 && len(p.ConfigVersions) == 0 {
		return
	}
	plan := &outputapiv1alpha2.UpgradePlan{Components: p.Components, ConfigVersions: p.ConfigVersions}
	if err := p.Printer.PrintObj(plan, writer); err != nil {
		fmt.Fprintf(os.Stderr, "could not flush output buffer: %v\n", err)
	}
	p.Components = p.Components[:0]
}

// Close does nothing.
func (p *upgradePlanJSONYAMLPrinter) Close(writer io.Writer) {}

// upgradePlanTextPrinter prints upgrade plan in a text form
type upgradePlanTextPrinter struct {
	output.TextPrinter
	columns   []string
	tabwriter *tabwriter.Writer
}

// Flush writes any buffered data
func (p *upgradePlanTextPrinter) Flush(writer io.Writer, last bool) {
	if p.tabwriter != nil {
		p.tabwriter.Flush()
		p.tabwriter = nil
		p.Fprintln(writer, "")
	}
}

// PrintObj is an implementation of ResourcePrinter.PrintObj for upgrade plan plain text output
func (p *upgradePlanTextPrinter) PrintObj(obj runtime.Object, writer io.Writer) error {
	if p.tabwriter == nil {
		p.tabwriter = tabwriter.NewWriter(writer, 10, 4, 3, ' ', 0)
		// Print header
		fmt.Fprintln(p.tabwriter, strings.Join(p.columns, "\t"))
	}

	item, ok := obj.(*outputapiv1alpha2.ComponentUpgradePlan)
	if !ok {
		return errors.Errorf("expected ComponentUpgradePlan, but got %+v", obj)
	}

	// Print item
	fmt.Fprintf(p.tabwriter, "%s\t%s\t%s\n", item.Name, item.CurrentVersion, item.NewVersion)
	return nil
}

// upgradePlanTextPrintFlags provides flags necessary for printing upgrade plan in a text form
type upgradePlanTextPrintFlags struct{}

func (pf *upgradePlanTextPrintFlags) AddFlags(cmd *cobra.Command) {}

// AllowedFormats returns a list of allowed output formats
func (pf *upgradePlanTextPrintFlags) AllowedFormats() []string {
	return []string{output.TextOutput}
}

// ToPrinter returns a kubeadm printer for the text output format
func (pf *upgradePlanTextPrintFlags) ToPrinter(outputFormat string) (output.Printer, error) {
	if outputFormat == output.TextOutput {
		return &upgradePlanTextPrinter{columns: []string{"COMPONENT", "CURRENT", "TARGET"}}, nil
	}
	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: []string{output.JSONOutput, output.YAMLOutput, output.TextOutput}}
}

// runPlan takes care of outputting available versions to upgrade to for the user
func runPlan(flags *planFlags, args []string, printer output.Printer) error {
	// Start with the basics, verify that the cluster is healthy, build a client and a versionGetter. Never dry-run when planning.
	klog.V(1).Infoln("[upgrade/plan] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/plan] retrieving configuration from cluster")
	client, versionGetter, cfg, err := enforceRequirements(flags.applyPlanFlags, args, false, false, printer, loadConfig)
	if err != nil {
		return err
	}

	// Currently this is the only method we have for distinguishing
	// external etcd vs static pod etcd
	isExternalEtcd := cfg.Etcd.External != nil

	// Compute which upgrade possibilities there are
	klog.V(1).Infoln("[upgrade/plan] computing upgrade possibilities")
	availUpgrades, err := upgrade.GetAvailableUpgrades(versionGetter, flags.allowExperimentalUpgrades, flags.allowRCUpgrades, isExternalEtcd, client, constants.GetStaticPodDirectory(), printer)
	if err != nil {
		return errors.Wrap(err, "[upgrade/versions] FATAL")
	}

	// Fetch the current state of the component configs
	klog.V(1).Infoln("[upgrade/plan] analysing component config version states")
	configVersionStates, err := componentconfigs.GetVersionStates(&cfg.ClusterConfiguration, client)
	if err != nil {
		return errors.WithMessage(err, "[upgrade/versions] FATAL")
	}

	// No upgrades available
	if len(availUpgrades) == 0 {
		klog.V(1).Infoln("[upgrade/plan] Awesome, you're up-to-date! Enjoy!")
		return nil
	}

	// A workaround to set the configVersionStates in the printer
	if p, ok := printer.(*upgradePlanJSONYAMLPrinter); ok {
		p.ConfigVersions = configVersionStates
	}

	// Generate and print upgrade plans
	for _, up := range availUpgrades {
		plan, unstableVersionFlag, err := genUpgradePlan(&up, isExternalEtcd)
		if err != nil {
			return err
		}

		// Actually, this is needed for machine-readable output only.
		// printUpgradePlan won't output the configVersionStates as it will simply print the same table several times
		// in the human-readable output if it did so
		plan.ConfigVersions = configVersionStates

		printUpgradePlan(&up, plan, unstableVersionFlag, isExternalEtcd, os.Stdout, printer)
	}

	// Finally, print the component config state table
	printComponentConfigVersionStates(configVersionStates, os.Stdout, printer)
	return nil
}

// TODO There is currently no way to cleanly output upgrades that involve adding, removing, or changing components
// https://github.com/kubernetes/kubeadm/issues/810 was created to track addressing this.
func appendDNSComponent(components []outputapiv1alpha2.ComponentUpgradePlan, up *upgrade.Upgrade, name string) []outputapiv1alpha2.ComponentUpgradePlan {
	beforeVersion := up.Before.DNSVersion
	afterVersion := up.After.DNSVersion

	if beforeVersion != "" || afterVersion != "" {
		components = append(components, newComponentUpgradePlan(name, beforeVersion, afterVersion))
	}
	return components
}

// genUpgradePlan generates output-friendly upgrade plan out of upgrade.Upgrade structure
func genUpgradePlan(up *upgrade.Upgrade, isExternalEtcd bool) (*outputapiv1alpha2.UpgradePlan, string, error) {
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

	components := []outputapiv1alpha2.ComponentUpgradePlan{}

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

	components = appendDNSComponent(components, up, constants.CoreDNS)

	if !isExternalEtcd {
		components = append(components, newComponentUpgradePlan(constants.Etcd, up.Before.EtcdVersion, up.After.EtcdVersion))
	}

	return &outputapiv1alpha2.UpgradePlan{Components: components}, unstableVersionFlag, nil
}

// printUpgradePlan prints a UX-friendly overview of what versions are available to upgrade to
func printUpgradePlan(up *upgrade.Upgrade, plan *outputapiv1alpha2.UpgradePlan, unstableVersionFlag string, isExternalEtcd bool, writer io.Writer, printer output.Printer) {
	printHeader := true
	printManualUpgradeHeader := true
	for _, component := range plan.Components {
		if isExternalEtcd && component.Name == constants.Etcd {
			// Don't print etcd if it's external
			continue
		} else if component.Name == constants.Kubelet {
			if printManualUpgradeHeader {
				printer.Fprintln(writer, "Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':")
				plan := newComponentUpgradePlan(component.Name, component.CurrentVersion, component.NewVersion)
				printer.PrintObj(&plan, writer)
				printManualUpgradeHeader = false
			} else {
				plan := newComponentUpgradePlan("", component.CurrentVersion, component.NewVersion)
				printer.PrintObj(&plan, writer)
			}
		} else {
			if printHeader {
				// End of manual upgrades table
				printer.Flush(writer, false)
				printer.Fprintf(writer, "Upgrade to the latest %s:\n", up.Description)
				printer.Fprintln(writer, "")
				printHeader = false
			}
			plan := newComponentUpgradePlan(component.Name, component.CurrentVersion, component.NewVersion)
			printer.PrintObj(&plan, writer)
		}
	}

	printer.Flush(writer, true)

	printer.Fprintln(writer, "You can now apply the upgrade by executing the following command:")
	printer.Fprintln(writer, "")
	printer.Fprintf(writer, "\tkubeadm upgrade apply %s%s\n", up.After.KubeVersion, unstableVersionFlag)
	printer.Fprintln(writer, "")

	if up.Before.KubeadmVersion != up.After.KubeadmVersion {
		printer.Fprintf(writer, "Note: Before you can perform this upgrade, you have to update kubeadm to %s.\n", up.After.KubeadmVersion)
		printer.Fprintln(writer, "")
	}

	printLineSeparator(writer, printer)
	printer.Close(writer)
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

func printLineSeparator(w io.Writer, printer output.Printer) {
	printer.Fprintf(w, "_____________________________________________________________________\n\n")
}

func printComponentConfigVersionStates(versionStates []outputapiv1alpha2.ComponentConfigVersionState, w io.Writer, printer output.Printer) {
	if len(versionStates) == 0 {
		printer.Fprintln(w, "No information available on component configs.")
		return
	}

	printer.Fprintln(w, dedent.Dedent(`
		The table below shows the current state of component configs as understood by this version of kubeadm.
		Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
		resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
		upgrade to is denoted in the "PREFERRED VERSION" column.
	`))

	tabw := tabwriter.NewWriter(w, 10, 4, 3, ' ', 0)
	printer.Fprintln(tabw, "API GROUP\tCURRENT VERSION\tPREFERRED VERSION\tMANUAL UPGRADE REQUIRED")

	for _, state := range versionStates {
		printer.Fprintf(tabw,
			"%s\t%s\t%s\t%s\n",
			state.Group,
			strOrDash(state.CurrentVersion),
			strOrDash(state.PreferredVersion),
			yesOrNo(state.ManualUpgradeRequired),
		)
	}

	tabw.Flush()
	printLineSeparator(w, printer)
}
