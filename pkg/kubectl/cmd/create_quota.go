/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	quotaLong = templates.LongDesc(`
		Create a resourcequota with the specified name, hard limits and optional scopes`)

	quotaExample = templates.Examples(`
		# Create a new resourcequota named my-quota
		$ kubectl create quota my-quota --hard=cpu=1,memory=1G,pods=2,services=3,replicationcontrollers=2,resourcequotas=1,secrets=5,persistentvolumeclaims=10

		# Create a new resourcequota named best-effort
		$ kubectl create quota best-effort --hard=pods=100 --scopes=BestEffort`)
)

// NewCmdCreateQuota is a macro command to create a new quota
func NewCmdCreateQuota(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "quota NAME [--hard=key1=value1,key2=value2] [--scopes=Scope1,Scope2] [--dry-run=bool]",
		Aliases: []string{"resourcequota"},
		Short:   "Create a quota with the specified name.",
		Long:    quotaLong,
		Example: quotaExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateQuota(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ResourceQuotaV1GeneratorName)
	cmd.Flags().String("hard", "", "A comma-delimited set of resource=quantity pairs that define a hard limit.")
	cmd.Flags().String("scopes", "", "A comma-delimited set of quota scopes that must all match each object tracked by the quota.")
	return cmd
}

// CreateQuota implements the behavior to run the create quota command
func CreateQuota(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ResourceQuotaV1GeneratorName:
		generator = &kubectl.ResourceQuotaGeneratorV1{
			Name:   name,
			Hard:   cmdutil.GetFlagString(cmd, "hard"),
			Scopes: cmdutil.GetFlagString(cmd, "scopes"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetFlagBool(cmd, "dry-run"),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
