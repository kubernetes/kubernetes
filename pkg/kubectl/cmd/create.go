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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

// CreateOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type CreateOptions struct {
	Filenames []string
	Recursive bool
}

const (
	create_long = `Create a resource by filename or stdin.

JSON and YAML formats are accepted.`
	create_example = `# Create a pod using the data in pod.json.
kubectl create -f ./pod.json

# Create a pod based on the JSON passed into stdin.
cat pod.json | kubectl create -f -`
)

func NewCmdCreate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &CreateOptions{}

	cmd := &cobra.Command{
		Use:     "create -f FILENAME",
		Short:   "Create a resource by filename or stdin",
		Long:    create_long,
		Example: create_example,
		Run: func(cmd *cobra.Command, args []string) {
			if len(options.Filenames) == 0 {
				cmd.Help()
				return
			}
			cmdutil.CheckErr(ValidateArgs(cmd, args))
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			cmdutil.CheckErr(RunCreate(f, cmd, out, options))
		},
	}

	usage := "Filename, directory, or URL to file to use to create the resource"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.MarkFlagRequired("filename")
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddOutputFlagsForMutation(cmd)
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	// create subcommands
	cmd.AddCommand(NewCmdCreateNamespace(f, out))
	cmd.AddCommand(NewCmdCreateSecret(f, out))
	cmd.AddCommand(NewCmdCreateConfigMap(f, out))
	cmd.AddCommand(NewCmdCreateServiceAccount(f, out))
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

	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}

		if cmdutil.ShouldRecord(cmd, info) {
			if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
		}

		if err := createAndRefresh(info); err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}

		count++
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
		var port int32
		if useNodePort {
			port = ports[ix].NodePort
		} else {
			port = ports[ix].Port
		}
		pieces[ix] = fmt.Sprintf("%s:%d", strings.ToLower(string(ports[ix].Protocol)), port)
	}
	return strings.Join(pieces, ",")
}

// createAndRefresh creates an object from input info and refreshes info with that object
func createAndRefresh(info *resource.Info) error {
	obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, info.Object)
	if err != nil {
		return err
	}
	info.Refresh(obj, true)
	return nil
}

// NameFromCommandArgs is a utility function for commands that assume the first argument is a resource name
func NameFromCommandArgs(cmd *cobra.Command, args []string) (string, error) {
	if len(args) == 0 {
		return "", cmdutil.UsageError(cmd, "NAME is required")
	}
	return args[0], nil
}

// CreateSubcommandOptions is an options struct to support create subcommands
type CreateSubcommandOptions struct {
	// Name of resource being created
	Name string
	// StructuredGenerator is the resource generator for the object being created
	StructuredGenerator kubectl.StructuredGenerator
	// DryRun is true if the command should be simulated but not run against the server
	DryRun bool
	// OutputFormat
	OutputFormat string
}

// RunCreateSubcommand executes a create subcommand using the specified options
func RunCreateSubcommand(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, options *CreateSubcommandOptions) error {
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	obj, err := options.StructuredGenerator.StructuredGenerate()
	if err != nil {
		return err
	}
	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	gvk, err := typer.ObjectKind(obj)
	mapping, err := mapper.RESTMapping(unversioned.GroupKind{Group: gvk.Group, Kind: gvk.Kind}, gvk.Version)
	if err != nil {
		return err
	}
	client, err := f.ClientForMapping(mapping)
	if err != nil {
		return err
	}
	resourceMapper := &resource.Mapper{
		ObjectTyper:  typer,
		RESTMapper:   mapper,
		ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),
	}
	info, err := resourceMapper.InfoForObject(obj, nil)
	if err != nil {
		return err
	}
	if err := kubectl.UpdateApplyAnnotation(info, f.JSONEncoder()); err != nil {
		return err
	}
	if !options.DryRun {
		obj, err = resource.NewHelper(client, mapping).Create(namespace, false, info.Object)
		if err != nil {
			return err
		}
	}

	if useShortOutput := options.OutputFormat == "name"; useShortOutput || len(options.OutputFormat) == 0 {
		cmdutil.PrintSuccess(mapper, useShortOutput, out, mapping.Resource, options.Name, "created")
		return nil
	}

	return f.PrintObject(cmd, mapper, obj, out)
}
