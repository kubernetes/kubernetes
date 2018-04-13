/*
Copyright 2018 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	psp "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	clientgo "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	PSPLong = templates.LongDesc(i18n.T(`
		Create a PodSecurityPolicy.`))

	pspExample = templates.Examples(i18n.T(`

		# Create a PodSecurityPolicy named "example" that simply uses default values.
		kubectl create podsecuritypolicy example
		`))
)

type createPSPOptions struct {
	Name               string
	SELinux            string
	RunAsUser          string
	SupplementalGroups string
	FSGroup            string

	DryRun       bool
	OutputFormat string
	Client       clientgo.ExtensionsV1beta1Interface
	Mapper       meta.RESTMapper
	Out          io.Writer
	PrintObject  func(obj runtime.Object) error
}

func NewCmdCreatePSP(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &createPSPOptions{
		Out: cmdOut,
	}
	cmd := &cobra.Command{
		Use: "podsecuritypolicy NAME --selinux=rule --supplemental-groups=rule --run-as-user=rule --fs-group=rule [other options] [--dry-run]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"psp"},
		Short:                 PSPLong,
		Long:                  PSPLong,
		Example:               pspExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(f, cmd, args))
			cmdutil.CheckErr(c.Validate())
			cmdutil.CheckErr(c.Run())
		},
	}

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().String("selinux", "RunAsAny", "define the strategy that will dictate the allowable labels that may be set.Legal values are RunAsAny/MustRunAs.")
	cmd.Flags().String("run-as-user", "RunAsAny", "control what user ID containers can run as.Legal values are RunAsAny/MustRunAs/MustRunAsNonRoot.")
	cmd.Flags().String("supplemental-groups", "RunAsAny", "control which group IDs containers can add.Legal values are RunAsAny/MustRunAs")
	cmd.Flags().String("fs-group", "RunAsAny", "Controls the supplemental group applied to some volumes.Legal values are RunAsAny/MustRunAs")

	return cmd
}

func (c *createPSPOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {

	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	c.Name = name
	c.SELinux = cmdutil.GetFlagString(cmd, "selinux")
	c.RunAsUser = cmdutil.GetFlagString(cmd, "run-as-user")
	c.SupplementalGroups = cmdutil.GetFlagString(cmd, "supplemental-groups")
	c.FSGroup = cmdutil.GetFlagString(cmd, "fs-group")
	// Complete other options for Run.
	c.Mapper, _ = f.Object()

	c.DryRun = cmdutil.GetDryRunFlag(cmd)
	c.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	c.PrintObject = func(obj runtime.Object) error {
		return cmdutil.PrintObject(cmd, obj, c.Out)
	}
	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	c.Client = clientset.ExtensionsV1beta1()

	return nil
}

func (c *createPSPOptions) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}
	if c.SELinux != string(extensions.SELinuxStrategyMustRunAs) && c.SELinux != string(extensions.SELinuxStrategyRunAsAny) {
		return fmt.Errorf("illegal --selinux rule, must be one of RunAsAny MustRunAs")
	}
	if c.RunAsUser != string(extensions.RunAsUserStrategyMustRunAs) && c.RunAsUser != string(extensions.RunAsUserStrategyMustRunAsNonRoot) && c.RunAsUser != string(extensions.RunAsUserStrategyRunAsAny) {
		return fmt.Errorf("illegal --run-as-user rule, must be one of RunAsAny MustRunAs MustRunAsNonRoot")
	}
	if c.SupplementalGroups != string(extensions.SupplementalGroupsStrategyMustRunAs) && c.SupplementalGroups != string(extensions.SupplementalGroupsStrategyRunAsAny) {
		return fmt.Errorf("illegal --supplemental-groups rule, must be one of RunAsAny MustRunAs")
	}
	if c.FSGroup != string(extensions.FSGroupStrategyMustRunAs) && c.FSGroup != string(extensions.FSGroupStrategyRunAsAny) {
		return fmt.Errorf("illegal --fs-group rule, must be one of RunAsAny MustRunAs")
	}
	return nil
}

func (c *createPSPOptions) Run() error {
	createdPSP := &psp.PodSecurityPolicy{}
	createdPSP.Name = c.Name

	createdPSP.Spec.SELinux = psp.SELinuxStrategyOptions{
		Rule: psp.SELinuxStrategy(c.SELinux),
	}
	createdPSP.Spec.RunAsUser = psp.RunAsUserStrategyOptions{
		Rule: psp.RunAsUserStrategy(c.RunAsUser),
	}
	createdPSP.Spec.SupplementalGroups = psp.SupplementalGroupsStrategyOptions{
		Rule: psp.SupplementalGroupsStrategyType(c.SupplementalGroups),
	}
	createdPSP.Spec.FSGroup = psp.FSGroupStrategyOptions{
		Rule: psp.FSGroupStrategyType(c.FSGroup),
	}

	if !c.DryRun {
		var err error
		createdPSP, err = c.Client.PodSecurityPolicies().Create(createdPSP)
		if err != nil {
			return err
		}
	}

	if useShortOutput := c.OutputFormat == "name"; useShortOutput || len(c.OutputFormat) == 0 {
		cmdutil.PrintSuccess(useShortOutput, c.Out, createdPSP, c.DryRun, "created")
		return nil
	}

	return c.PrintObject(createdPSP)
}
