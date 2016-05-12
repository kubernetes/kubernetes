/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	processTemplateLong = `
Process the template by passing the parameterized values as literal commandline flags.`

	processTemplateExample = `  # Process the template named template-name using only parameter defaults
  kubectl process template template-name

  # Process the template named template-name using the literal parameter substitutions
  kubectl create configmap my-config --from-literal=PNAME1=value --from-literal=PNAME2=2 --from-literal=PNAME3=true`
)

// NewCmdProcessTemplate is a macro command to create a new namespace
func NewCmdProcessTemplate(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "template NAME [--from-literal=PARAM_NAME=value]",
		Short:   "Process the template with the specified name.",
		Long:    processTemplateLong,
		Example: processTemplateExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ProcessTemplate(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringP("output", "o", "yaml", "Output format. One of: json|yaml.")
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.TemplateParametersV1GeneratorName)
	cmd.Flags().StringSlice("from-literal", []string{}, "Specify a named parameter and literal value to substitute (i.e. PARAM=somevalue)")
	return cmd
}

// ProcessTemplate implements the behavior to run the process template command
func ProcessTemplate(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.TemplateParametersV1GeneratorName:
		generator = &kubectl.TemplateParametersGeneratorV1Beta1{
			Name:           name,
			LiteralSources: cmdutil.GetFlagStringSlice(cmd, "from-literal"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunProcessSubcommand(f, cmd, cmdOut, &ProcessSubcommandOptions{
		StructuredGenerator: generator,
		Output:              cmdutil.GetFlagString(cmd, "output"),
	})
}
