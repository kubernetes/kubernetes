/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package secret

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

const (
	makeLong = `
Create a secret based on a file, directory, or specified literal value

Key files can be specified using their file path, in which case a default name will be given to them, or optionally 
with a name and file path, in which case the given name will be used. Specifying a directory will create a secret 
using with all valid keys in that directory.
`

	makeExample = `  // Create a new secret named my-secret with keys for each file in folder bar
  $ kubectl secret make my-secret --from-file=path/to/bar

  // Create a new secret named my-secret with specified keys instead of names on disk
  $ kubectl secret make my-secret --from-file=ssh-privatekey=~/.ssh/id_rsa --from-file=ssh-publickey=~/.ssh/id_rsa.pub

  // Create a new secret named my-secret with key1=supersecret and key2=topsecret
  $ kubectl secret make my-secret --from-literal=key1=supersecret --from-literal=key2=topsecret`
)

// TODO: --quiet -q to suppress warnings on ignored files
func NewCmdMakeSecret(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "make NAME [--from-file=[key=]source] [--from-literal=key1=value1] [--dry-run=bool]",
		Short:   "Make a secret from a local file, directory or literal value.",
		Long:    makeLong,
		Example: makeExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := MakeSecret(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "", "The name of the API generator to use.  If not specified, default to secret/v1")
	cmd.Flags().StringSlice("from-file", []string{}, "Key files can be specified using their file path, in which case a default name will be given to them, or optionally with a name and file path, in which case the given name will be used.  Specifying a directory will iterate each named file in the directory that is a valid secret key.")
	cmd.Flags().StringSlice("from-literal", []string{}, "Specify a key and literal value to insert in secret (i.e. mykey=somevalue)")
	cmd.Flags().String("type", "", "The type of secret to make")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")
	return cmd
}

func MakeSecret(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmdutil.UsageError(cmd, "NAME is required for make")
	}

	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	generatorName := cmdutil.GetFlagString(cmd, "generator")
	if len(generatorName) == 0 {
		generatorName = "secret/v1"
	}
	generator, found := f.Generator(generatorName)
	if !found {
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not found.", generatorName))
	}
	names := generator.ParamNames()
	params := kubectl.MakeParams(cmd, names)
	params["name"] = args[0]
	if len(args) > 1 {
		params["args"] = args[1:]
	}

	params["from-file"] = cmdutil.GetFlagStringSlice(cmd, "from-file")
	params["from-literal"] = cmdutil.GetFlagStringSlice(cmd, "from-literal")

	err = kubectl.ValidateParams(names, params)
	if err != nil {
		return err
	}

	obj, err := generator.Generate(params)
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	version, kind, err := typer.ObjectVersionAndKind(obj)
	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(kind, version)
	if err != nil {
		return err
	}
	client, err := f.RESTClient(mapping)
	if err != nil {
		return err
	}

	// TODO: extract this flag to a central location, when such a location exists.
	if !cmdutil.GetFlagBool(cmd, "dry-run") {
		resourceMapper := &resource.Mapper{ObjectTyper: typer, RESTMapper: mapper, ClientMapper: f.ClientMapperForCommand()}
		info, err := resourceMapper.InfoForObject(obj)
		if err != nil {
			return err
		}

		// Serialize the configuration into an annotation.
		if err := kubectl.UpdateApplyAnnotation(info); err != nil {
			return err
		}

		// Serialize the object with the annotation applied.
		data, err := mapping.Codec.Encode(info.Object)
		if err != nil {
			return err
		}

		obj, err = resource.NewHelper(client, mapping).Create(namespace, false, data)
		if err != nil {
			return err
		}
	}
	outputFormat := cmdutil.GetFlagString(cmd, "output")
	if outputFormat != "" {
		return f.PrintObject(cmd, obj, cmdOut)
	}
	cmdutil.PrintSuccess(mapper, false, cmdOut, mapping.Resource, args[0], "created")
	return nil
}
