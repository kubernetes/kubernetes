/*
Copyright 2014 The Kubernetes Authors.

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
	gruntime "runtime"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

var (
	create_long = templates.LongDesc(`
		Create a resource by filename or stdin.

		JSON and YAML formats are accepted.`)

	create_example = templates.Examples(`
		# Create a pod using the data in pod.json.
		kubectl create -f ./pod.json

		# Create a pod based on the JSON passed into stdin.
		cat pod.json | kubectl create -f -

		# Edit the data in docker-registry.yaml in JSON using the v1 API format then create the resource using the edited data.
		kubectl create -f docker-registry.yaml --edit --output-version=v1 -o json`)
)

func NewCmdCreate(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	cmd := &cobra.Command{
		Use:     "create -f FILENAME",
		Short:   "Create a resource by filename or stdin",
		Long:    create_long,
		Example: create_example,
		Run: func(cmd *cobra.Command, args []string) {
			if cmdutil.IsFilenameEmpty(options.Filenames) {
				defaultRunFunc := cmdutil.DefaultSubCommandRun(errOut)
				defaultRunFunc(cmd, args)
				return
			}
			cmdutil.CheckErr(ValidateArgs(cmd, args))
			cmdutil.CheckErr(RunCreate(f, cmd, out, errOut, options))
		},
	}

	usage := "to use to create the resource"
	cmdutil.AddFilenameOptionFlags(cmd, options, usage)
	cmd.MarkFlagRequired("filename")
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().Bool("edit", false, "Edit the API resource before creating")
	cmd.Flags().Bool("windows-line-endings", gruntime.GOOS == "windows", "Only relevant if --edit=true. Use Windows line-endings (default Unix line-endings)")
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	// create subcommands
	cmd.AddCommand(NewCmdCreateNamespace(f, out))
	cmd.AddCommand(NewCmdCreateQuota(f, out))
	cmd.AddCommand(NewCmdCreateSecret(f, out, errOut))
	cmd.AddCommand(NewCmdCreateConfigMap(f, out))
	cmd.AddCommand(NewCmdCreateServiceAccount(f, out))
	cmd.AddCommand(NewCmdCreateService(f, out, errOut))
	cmd.AddCommand(NewCmdCreateDeployment(f, out))
	cmd.AddCommand(NewCmdCreateClusterRoleBinding(f, out))
	cmd.AddCommand(NewCmdCreatePodDisruptionBudget(f, out))
	return cmd
}

func ValidateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageError(cmd, "Unexpected args: %v", args)
	}
	return nil
}

func RunCreate(f cmdutil.Factory, cmd *cobra.Command, out, errOut io.Writer, options *resource.FilenameOptions) error {
	if cmdutil.GetFlagBool(cmd, "edit") {
		return RunEditOnCreate(f, out, errOut, cmd, options)
	}
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), runtime.UnstructuredJSONScheme).
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		Flatten().
		Do()
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
		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}

		if cmdutil.ShouldRecord(cmd, info) {
			if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
		}

		if !dryRun {
			if err := createAndRefresh(info); err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
		}

		count++
		shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
		if !shortOutput {
			f.PrintObjectSpecificMessage(info.Object, out)
		}

		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, dryRun, "created")
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

func RunEditOnCreate(f cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, options *resource.FilenameOptions) error {
	return runEdit(f, out, errOut, cmd, []string{}, options, EditBeforeCreateMode)
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
func RunCreateSubcommand(f cmdutil.Factory, cmd *cobra.Command, out io.Writer, options *CreateSubcommandOptions) error {
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	obj, err := options.StructuredGenerator.StructuredGenerate()
	if err != nil {
		return err
	}
	mapper, typer := f.Object()
	gvks, _, err := typer.ObjectKinds(obj)
	if err != nil {
		return err
	}
	gvk := gvks[0]
	mapping, err := mapper.RESTMapping(schema.GroupKind{Group: gvk.Group, Kind: gvk.Kind}, gvk.Version)
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
		cmdutil.PrintSuccess(mapper, useShortOutput, out, mapping.Resource, options.Name, options.DryRun, "created")
		return nil
	}

	return f.PrintObject(cmd, mapper, obj, out)
}
