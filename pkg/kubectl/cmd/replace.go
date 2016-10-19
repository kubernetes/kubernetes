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
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

// ReplaceSubcommandOptions is an options struct to support replace subcommands
type ReplaceSubcommandOptions struct {
	// Name of the resource
	Name string
	// Resource type of the resource
	Resource string
	// Subtype of the resource (optional)
	Subtype string
	// OutputFormat
	OutputFormat string
}

var (
	replace_long = templates.LongDesc(`
		Replace a resource by filename or stdin.

		JSON and YAML formats are accepted. If replacing an existing resource, the
		complete resource spec must be provided. This can be obtained by

		    $ kubectl get TYPE NAME -o yaml

		Please refer to the models in https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/HEAD/docs/api-reference/v1/definitions.html to find if a field is mutable.`)

	replace_example = templates.Examples(`
		# Replace a pod using the data in pod.json.
		kubectl replace -f ./pod.json

		# Replace a pod based on the JSON passed into stdin.
		cat pod.json | kubectl replace -f -

		# Update a single-container pod's image version (tag) to v4
		kubectl get pod mypod -o yaml | sed 's/\(image: myimage\):.*$/\1:v4/' | kubectl replace -f -

		# Force replace, delete and then re-create the resource
		kubectl replace --force -f ./pod.json`)
)

func NewCmdReplace(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	cmd := &cobra.Command{
		Use: "replace -f FILENAME",
		// update is deprecated.
		Aliases: []string{"update"},
		Short:   "Replace a resource by filename or stdin",
		Long:    replace_long,
		Example: replace_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			err := RunReplace(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}
	usage := "to use to replace the resource."
	cmdutil.AddFilenameOptionFlags(cmd, options, usage)
	cmd.MarkFlagRequired("filename")
	cmdutil.AddForceReplaceFlags(cmd, true)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddOutputFlagsForMutation(cmd)
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	cmd.AddCommand(NewCmdReplaceConfigMap(f, out))
	return cmd
}

func RunReplace(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *resource.FilenameOptions) error {
	if len(os.Args) > 1 && os.Args[1] == "update" {
		printDeprecationWarning("replace", "update")
	}
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	force := cmdutil.GetFlagBool(cmd, "force")
	if cmdutil.IsFilenameEmpty(options.Filenames) {
		return cmdutil.UsageError(cmd, "Must specify --filename to replace")
	}

	shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
	if force {
		return forceReplace(f, out, cmd, args, shortOutput, options)
	}

	if cmdutil.GetFlagInt(cmd, "grace-period") >= 0 {
		return fmt.Errorf("--grace-period must have --force specified")
	}

	if cmdutil.GetFlagDuration(cmd, "timeout") != 0 {
		return fmt.Errorf("--timeout must have --force specified")
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
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

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return cmdutil.AddSourceToErr("replacing", info.Source, err)
		}

		if cmdutil.ShouldRecord(cmd, info) {
			if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
				return cmdutil.AddSourceToErr("replacing", info.Source, err)
			}
		}

		// Serialize the object with the annotation applied.
		obj, err := resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, true, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr("replacing", info.Source, err)
		}

		info.Refresh(obj, true)
		f.PrintObjectSpecificMessage(obj, out)
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, false, "replaced")
		return nil
	})
}

func forceReplace(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, shortOutput bool, options *resource.FilenameOptions) error {
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	for i, filename := range options.Filenames {
		if filename == "-" {
			tempDir, err := ioutil.TempDir("", "kubectl_replace_")
			if err != nil {
				return err
			}
			defer os.RemoveAll(tempDir)
			tempFilename := filepath.Join(tempDir, "resource.stdin")
			err = cmdutil.DumpReaderToFile(os.Stdin, tempFilename)
			if err != nil {
				return err
			}
			options.Filenames[i] = tempFilename
		}
	}

	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), runtime.UnstructuredJSONScheme).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		ResourceTypeOrNameArgs(false, args...).RequireObject(false).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	//Replace will create a resource if it doesn't exist already, so ignore not found error
	ignoreNotFound := true
	timeout := cmdutil.GetFlagDuration(cmd, "timeout")
	// By default use a reaper to delete all related resources.
	if cmdutil.GetFlagBool(cmd, "cascade") {
		glog.Warningf("\"cascade\" is set, kubectl will delete and re-create all resources managed by this resource (e.g. Pods created by a ReplicationController). Consider using \"kubectl rolling-update\" if you want to update a ReplicationController together with its Pods.")
		err = ReapResult(r, f, out, cmdutil.GetFlagBool(cmd, "cascade"), ignoreNotFound, timeout, cmdutil.GetFlagInt(cmd, "grace-period"), shortOutput, mapper, false)
	} else {
		err = DeleteResult(r, out, ignoreNotFound, shortOutput, mapper)
	}
	if err != nil {
		return err
	}

	if timeout == 0 {
		timeout = kubectl.Timeout
	}
	r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		return wait.PollImmediate(kubectl.Interval, timeout, func() (bool, error) {
			if err := info.Get(); !errors.IsNotFound(err) {
				return false, err
			}
			return true, nil
		})
	})

	r = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), runtime.UnstructuredJSONScheme).
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

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return err
		}

		if cmdutil.ShouldRecord(cmd, info) {
			if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
				return cmdutil.AddSourceToErr("replacing", info.Source, err)
			}
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, info.Object)
		if err != nil {
			return err
		}

		count++
		info.Refresh(obj, true)
		f.PrintObjectSpecificMessage(obj, out)
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, false, "replaced")
		return nil
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to replace")
	}
	return nil
}

// RunReplaceSubcommand executes a replace subcommand using the specified options
func RunReplaceSubcommand(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, args []string, options *ReplaceSubcommandOptions) error {
	if len(os.Args) > 1 && os.Args[1] == "update" {
		printDeprecationWarning("replace", "update")
	}

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	force := cmdutil.GetFlagBool(cmd, "force")
	shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
	if force {
		return forceReplaceSubcommand(f, out, cmd, args, shortOutput, options)
	}
	if cmdutil.GetFlagDuration(cmd, "timeout") != 0 {
		return fmt.Errorf("--timeout must have --force specified")
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceTypeOrNameArgs(true, options.Resource, options.Name).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return cmdutil.AddSourceToErr("replacing", info.Source, err)
		}

		if cmdutil.ShouldRecord(cmd, info) {
			if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
				return cmdutil.AddSourceToErr("replacing", info.Source, err)
			}
		}

		switch obj := info.Object.(type) {
		case *api.ConfigMap:
			err := kubectl.HandleConfigMapReplace(obj,
				cmdutil.GetFlagStringSlice(cmd, "from-file"),
				cmdutil.GetFlagStringSlice(cmd, "from-literal"),
			)
			if err != nil {
				return cmdutil.AddSourceToErr("replacing", info.Source, err)
			}
		default:
			return cmdutil.AddSourceToErr("replacing", info.Source, fmt.Errorf("Replacing object type \"%T\" is not supported", obj))
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, true, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr("replacing", info.Source, err)
		}
		err = info.Refresh(obj, true)
		if err != nil {
			return err
		}
		f.PrintObjectSpecificMessage(obj, out)
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, "replaced")
		return nil
	})
}

func forceReplaceSubcommand(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, shortOutput bool, options *ReplaceSubcommandOptions) error {
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), runtime.UnstructuredJSONScheme).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceTypeOrNameArgs(true, options.Resource, options.Name).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	//Replace will create a resource if it doesn't exist already, so ignore not found error
	ignoreNotFound := true
	timeout := cmdutil.GetFlagDuration(cmd, "timeout")
	err = DeleteResult(r, out, ignoreNotFound, shortOutput, mapper)
	if err != nil {
		return err
	}

	if timeout == 0 {
		timeout = kubectl.Timeout
	}
	r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		return wait.PollImmediate(kubectl.Interval, timeout, func() (bool, error) {
			if err := info.Get(); !errors.IsNotFound(err) {
				return false, err
			}
			return true, nil
		})
	})

	var generator kubectl.StructuredGenerator
	switch options.Resource {
	case "configmaps":
		generator = &kubectl.ConfigMapGeneratorV1{
			Name:           options.Name,
			FileSources:    cmdutil.GetFlagStringSlice(cmd, "from-file"),
			LiteralSources: cmdutil.GetFlagStringSlice(cmd, "from-literal"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Force replacing \"%s\" is not supported.", options.Resource))
	}

	obj, err := generator.StructuredGenerate()
	if err != nil {
		return err
	}
	mapper, typer = f.Object()
	gvks, _, err := typer.ObjectKinds(obj)
	if err != nil {
		return err
	}
	gvk := gvks[0]
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
	if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
		return err
	}
	obj, err = resource.NewHelper(client, mapping).Create(cmdNamespace, true, info.Object)
	if err != nil {
		return err
	}
	err = info.Refresh(obj, true)
	if err != nil {
		return err
	}

	if useShortOutput := options.OutputFormat == "name"; useShortOutput || len(options.OutputFormat) == 0 {
		cmdutil.PrintSuccess(mapper, useShortOutput, out, mapping.Resource, options.Name, "replaced")
	}
	return nil
}
