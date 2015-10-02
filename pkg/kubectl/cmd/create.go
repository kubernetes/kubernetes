/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

// CreateOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type CreateOptions struct {
	Filenames []string
}

const (
	create_long = `Create a resource by filename or stdin.

JSON and YAML formats are accepted.`
	create_example = `# Create a pod using the data in pod.json.
$ kubectl create -f ./pod.json

# Create a pod based on the JSON passed into stdin.
$ cat pod.json | kubectl create -f -`
)

func NewCmdCreate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &CreateOptions{}

	cmd := &cobra.Command{
		Use:     "create -f FILENAME",
		Short:   "Create a resource by filename or stdin",
		Long:    create_long,
		Example: create_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(ValidateArgs(cmd, args))
			// Limit output options for non dry-run execution
			if !cmdutil.GetFlagBool(cmd, "dry-run") {
				cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			}
			cmdutil.CheckErr(RunCreate(f, cmd, out, options))
		},
	}

	usage := "Filename, directory, or URL to file to use to create the resource"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.MarkFlagRequired("filename")
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("transforms", "", "Comma separated list of transforms to apply.  If non-empty, load the transforms and use them to transform input.  Transforms are applied in the order specified in the list.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")
	return cmd
}

func ValidateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageError(cmd, "Unexpected args: %v", args)
	}
	return nil
}

func RunCreate(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, options *CreateOptions) error {
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Filenames...).
		Flatten()
	transformArg := cmdutil.GetFlagString(cmd, "transforms")
	if len(transformArg) > 0 {
		transform, err := getTransform(transformArg, f)
		if err != nil {
			return err
		}
		b.StreamTransform(transform)
	}
	r := b.Do()
	err = r.Err()
	if err != nil {
		return err
	}

	dryRun := cmdutil.GetFlagBool(cmd, "dry-run")

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		// Update the annotation used by kubectl apply
		if err := kubectl.UpdateApplyAnnotation(info); err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}

		// Serialize the object with the annotation applied.
		data, err := info.Mapping.Codec.Encode(info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}
		count++
		if dryRun {
			printer, err := f.PrinterForMapping(cmd, info.ResourceMapping(), false)
			if err != nil {
				return err
			}
			return printer.PrintObj(info.Object, out)
		}
		obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, data)
		if err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}
		info.Refresh(obj, true)
		shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
		if !shortOutput {
			printObjectSpecificMessage(info.Object, out)
		}
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, "created")
		return nil
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to create")
	}
	return nil
}

func printObjectSpecificMessage(obj runtime.Object, out io.Writer) {
	switch obj := obj.(type) {
	case *api.Service:
		if obj.Spec.Type == api.ServiceTypeNodePort {
			msg := fmt.Sprintf(
				`You have exposed your service on an external port on all nodes in your
cluster.  If you want to expose this service to the external internet, you may
need to set up firewall rules for the service port(s) (%s) to serve traffic.

See http://releases.k8s.io/HEAD/docs/user-guide/services-firewalls.md for more details.
`,
				makePortsString(obj.Spec.Ports, true))
			out.Write([]byte(msg))
		}
	}
}

func makePortsString(ports []api.ServicePort, useNodePort bool) string {
	pieces := make([]string, len(ports))
	for ix := range ports {
		var port int
		if useNodePort {
			port = ports[ix].NodePort
		} else {
			port = ports[ix].Port
		}
		pieces[ix] = fmt.Sprintf("%s:%d", strings.ToLower(string(ports[ix].Protocol)), port)
	}
	return strings.Join(pieces, ",")
}

// given a list of transform specs, give back a single compound transform, or error if one occurs.
func getTransform(transformSpec string, f *cmdutil.Factory) (resource.StreamTransform, error) {
	parts := strings.Split(transformSpec, ",")
	transforms := []resource.StreamTransform{}
	for _, part := range parts {
		transform, err := getOneTransform(part, f)
		if err != nil {
			return nil, err
		}
		transforms = append(transforms, transform)
	}
	return resource.StreamTransformList(transforms).Transform, nil
}

// given one transform spec, give back a StreamTransform, or error if one occurs.
func getOneTransform(transformSpec string, f *cmdutil.Factory) (resource.StreamTransform, error) {
	parts := strings.Split(transformSpec, ":")
	if len(parts) != 2 {
		return nil, fmt.Errorf("expected <name>:<arg>, saw: %s", transformSpec)
	}
	name := parts[0]
	arg := parts[1]
	switch name {
	case "generator":
		cfg, err := f.ClientConfig()
		if err != nil {
			return nil, err
		}
		generator, ok := f.Generator(arg)
		if !ok {
			return nil, err
		}
		gen := kubectl.GeneratorTransformer{
			Generator: generator,
			Codec:     cfg.Codec,
		}
		return gen.Transform, nil
	}
	return nil, fmt.Errorf("unknown transform: %s", name)
}
