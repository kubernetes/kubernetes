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
	makeDockercfgLong = `
Create a new dockercfg secret

Dockercfg secrets are used to authenticate against Docker registries.

When using the Docker command line to push images, you can authenticate to a given registry by running
  'docker login DOCKER_REGISTRY_SERVER --username=DOCKER_USER --password=DOCKER_PASSWORD --email=DOCKER_EMAIL'.
That produces a ~/.dockercfg file that is used by subsequent 'docker push' and 'docker pull' commands to
authenticate to the registry.

When creating applications, you may have a Docker registry that requires authentication.  In order for the
nodes to pull images on your behalf, they have to have the credentials.  You can provide this information
by creating a dockercfg secret and attaching it to your service account.`

	makeDockercfgExample = `  // If you don't already have a .dockercfg file, you can create a dockercfg secret directly by using:
  $ kubectl secret make-dockercfg my-secret --docker-server=DOCKER_REGISTRY_SERVER --docker-username=DOCKER_USER --docker-password=DOCKER_PASSWORD --docker-email=DOCKER_EMAIL

  // If you do already have a .dockercfg file, you can create a dockercfg secret by using:
  $ kubectl secret make-dockercfg my-secret -f path/to/.dockercfg`
)

func NewCmdMakeDockercfgSecret(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "make-dockercfg NAME [-f path/to/.dockercfg] [--docker-server=DOCKER_REGISTRY_SERVER --docker-username=DOCKER_USER --docker-password=DOCKER_PASSWORD --docker-email=DOCKER_EMAIL] [--dry-run=bool]",
		Short:   "Make a secret for use against Docker registries.",
		Long:    makeDockercfgLong,
		Example: makeDockercfgExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := MakeDockercfgSecret(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "", "The name of the API generator to use.  If not specified, default to secret/v1")
	cmd.Flags().String("docker-username", "", "Username for Docker registry authentication")
	cmd.Flags().String("docker-password", "", "Password for Docker registry authentication")
	cmd.Flags().String("docker-email", "", "Email for Docker registry")
	cmd.Flags().String("docker-server", "https://index.docker.io/v1/", "Server location for Docker registry")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")
	return cmd
}

func MakeDockercfgSecret(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
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
