/*
Copyright 2025 The Kubernetes Authors.

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

package kuberc

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"k8s.io/kubectl/pkg/kuberc"
	"sigs.k8s.io/yaml"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	viewLong = templates.LongDesc(i18n.T(`
		Display the current kuberc configuration.

		The kuberc file contains command aliases and default flag values
		for kubectl commands. This command displays the contents of the
		kuberc file in the specified output format.`))

	viewExample = templates.Examples(i18n.T(`
		# View kuberc configuration in YAML format (default)
		kubectl kuberc view

		# View kuberc configuration in JSON format
		kubectl kuberc view --output json

		# View a specific kuberc file
		kubectl kuberc view --kuberc /path/to/kuberc`))
)

// ViewOptions contains the options for viewing kuberc configuration
type ViewOptions struct {
	KubeRCFile string
	PrintFlags *genericclioptions.PrintFlags

	genericiooptions.IOStreams
}

// NewCmdKubeRCView returns a Command instance for 'kuberc view' sub command
func NewCmdKubeRCView(streams genericiooptions.IOStreams) *cobra.Command {
	o := &ViewOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput("yaml"),
		IOStreams:  streams,
	}

	cmd := &cobra.Command{
		Use:                   "view",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display the current kuberc configuration"),
		Long:                  viewLong,
		Example:               viewExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	cmd.Flags().StringVar(&o.KubeRCFile, "kuberc", o.KubeRCFile, "Path to the kuberc file to view")
	o.PrintFlags.AddFlags(cmd)

	return cmd
}

// Complete sets default values for ViewOptions
func (o *ViewOptions) Complete(cmd *cobra.Command) error {
	if env := os.Getenv("KUBERC"); env != "" && env != "off" {
		o.KubeRCFile = env
	} else if env != "off" {
		if o.KubeRCFile == "" {
			o.KubeRCFile = kuberc.RecommendedKubeRCFile
		}
	}
	return nil
}

// Validate validates the ViewOptions
func (o *ViewOptions) Validate() error {
	return nil
}

// Run executes the view command
func (o *ViewOptions) Run() error {
	if o.KubeRCFile == "" {
		return fmt.Errorf("KUBERC is disabled via KUBERC=off environment variable")
	}

	data, err := os.ReadFile(o.KubeRCFile)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("kuberc file not found at %s", o.KubeRCFile)
		}
		return fmt.Errorf("error reading kuberc file: %w", err)
	}

	var pref *v1beta1.Preference
	if err := yaml.Unmarshal(data, &pref); err != nil {
		return fmt.Errorf("error parsing kuberc file: %w", err)
	}

	if pref.Aliases == nil {
		pref.Aliases = []v1beta1.AliasOverride{}
	}
	if pref.Defaults == nil {
		pref.Defaults = []v1beta1.CommandDefaults{}
	}

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	return printer.PrintObj(pref, o.Out)
}
