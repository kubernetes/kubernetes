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

package create

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"runtime"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/cmd/util/editor"
	"k8s.io/kubectl/pkg/generate"
	"k8s.io/kubectl/pkg/rawhttp"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// CreateOptions is the commandline options for 'create' sub command
type CreateOptions struct {
	PrintFlags  *genericclioptions.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	DryRunStrategy cmdutil.DryRunStrategy
	DryRunVerifier *resource.DryRunVerifier

	fieldManager string

	FilenameOptions  resource.FilenameOptions
	Selector         string
	EditBeforeCreate bool
	Raw              string

	Recorder genericclioptions.Recorder
	PrintObj func(obj kruntime.Object) error

	genericclioptions.IOStreams
}

var (
	createLong = templates.LongDesc(i18n.T(`
		Create a resource from a file or from stdin.

		JSON and YAML formats are accepted.`))

	createExample = templates.Examples(i18n.T(`
		# Create a pod using the data in pod.json
		kubectl create -f ./pod.json

		# Create a pod based on the JSON passed into stdin
		cat pod.json | kubectl create -f -

		# Edit the data in docker-registry.yaml in JSON then create the resource using the edited data
		kubectl create -f docker-registry.yaml --edit -o json`))
)

// NewCreateOptions returns an initialized CreateOptions instance
func NewCreateOptions(ioStreams genericclioptions.IOStreams) *CreateOptions {
	return &CreateOptions{
		PrintFlags:  genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},

		IOStreams: ioStreams,
	}
}

// NewCmdCreate returns new initialized instance of create sub command
func NewCmdCreate(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreateOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "create -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a resource from a file or from stdin"),
		Long:                  createLong,
		Example:               createExample,
		Run: func(cmd *cobra.Command, args []string) {
			if cmdutil.IsFilenameSliceEmpty(o.FilenameOptions.Filenames, o.FilenameOptions.Kustomize) {
				ioStreams.ErrOut.Write([]byte("Error: must specify one of -f and -k\n\n"))
				defaultRunFunc := cmdutil.DefaultSubCommandRun(ioStreams.ErrOut)
				defaultRunFunc(cmd, args)
				return
			}
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.ValidateArgs(cmd, args))
			cmdutil.CheckErr(o.RunCreate(f, cmd))
		},
	}

	// bind flag structs
	o.RecordFlags.AddFlags(cmd)

	usage := "to use to create the resource"
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().BoolVar(&o.EditBeforeCreate, "edit", o.EditBeforeCreate, "Edit the API resource before creating")
	cmd.Flags().Bool("windows-line-endings", runtime.GOOS == "windows",
		"Only relevant if --edit=true. Defaults to the line ending native to your platform.")
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().StringVar(&o.Raw, "raw", o.Raw, "Raw URI to POST to the server.  Uses the transport specified by the kubeconfig file.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.fieldManager, "kubectl-create")

	o.PrintFlags.AddFlags(cmd)

	// create subcommands
	cmd.AddCommand(NewCmdCreateNamespace(f, ioStreams))
	cmd.AddCommand(NewCmdCreateQuota(f, ioStreams))
	cmd.AddCommand(NewCmdCreateSecret(f, ioStreams))
	cmd.AddCommand(NewCmdCreateConfigMap(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceAccount(f, ioStreams))
	cmd.AddCommand(NewCmdCreateService(f, ioStreams))
	cmd.AddCommand(NewCmdCreateDeployment(f, ioStreams))
	cmd.AddCommand(NewCmdCreateClusterRole(f, ioStreams))
	cmd.AddCommand(NewCmdCreateClusterRoleBinding(f, ioStreams))
	cmd.AddCommand(NewCmdCreateRole(f, ioStreams))
	cmd.AddCommand(NewCmdCreateRoleBinding(f, ioStreams))
	cmd.AddCommand(NewCmdCreatePodDisruptionBudget(f, ioStreams))
	cmd.AddCommand(NewCmdCreatePriorityClass(f, ioStreams))
	cmd.AddCommand(NewCmdCreateJob(f, ioStreams))
	cmd.AddCommand(NewCmdCreateCronJob(f, ioStreams))
	cmd.AddCommand(NewCmdCreateIngress(f, ioStreams))
	return cmd
}

// ValidateArgs makes sure there is no discrepency in command options
func (o *CreateOptions) ValidateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}
	if len(o.Raw) > 0 {
		if o.EditBeforeCreate {
			return cmdutil.UsageErrorf(cmd, "--raw and --edit are mutually exclusive")
		}
		if len(o.FilenameOptions.Filenames) != 1 {
			return cmdutil.UsageErrorf(cmd, "--raw can only use a single local file or stdin")
		}
		if strings.Index(o.FilenameOptions.Filenames[0], "http://") == 0 || strings.Index(o.FilenameOptions.Filenames[0], "https://") == 0 {
			return cmdutil.UsageErrorf(cmd, "--raw cannot read from a url")
		}
		if o.FilenameOptions.Recursive {
			return cmdutil.UsageErrorf(cmd, "--raw and --recursive are mutually exclusive")
		}
		if len(o.Selector) > 0 {
			return cmdutil.UsageErrorf(cmd, "--raw and --selector (-l) are mutually exclusive")
		}
		if len(cmdutil.GetFlagString(cmd, "output")) > 0 {
			return cmdutil.UsageErrorf(cmd, "--raw and --output are mutually exclusive")
		}
		if _, err := url.ParseRequestURI(o.Raw); err != nil {
			return cmdutil.UsageErrorf(cmd, "--raw must be a valid URL path: %v", err)
		}
	}

	return nil
}

// Complete completes all the required options
func (o *CreateOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error
	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, f.OpenAPIGetter())

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj kruntime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

// RunCreate performs the creation
func (o *CreateOptions) RunCreate(f cmdutil.Factory, cmd *cobra.Command) error {
	// raw only makes sense for a single file resource multiple objects aren't likely to do what you want.
	// the validator enforces this, so
	if len(o.Raw) > 0 {
		restClient, err := f.RESTClient()
		if err != nil {
			return err
		}
		return rawhttp.RawPost(restClient, o.IOStreams, o.Raw, o.FilenameOptions.Filenames[0])
	}

	if o.EditBeforeCreate {
		return RunEditOnCreate(f, o.PrintFlags, o.RecordFlags, o.IOStreams, cmd, &o.FilenameOptions, o.fieldManager)
	}
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		Unstructured().
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		LabelSelectorParam(o.Selector).
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
		if err := util.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info.Object, scheme.DefaultJSONEncoder()); err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}

		if err := o.Recorder.Record(info.Object); err != nil {
			klog.V(4).Infof("error recording current command: %v", err)
		}

		if o.DryRunStrategy != cmdutil.DryRunClient {
			if o.DryRunStrategy == cmdutil.DryRunServer {
				if err := o.DryRunVerifier.HasSupport(info.Mapping.GroupVersionKind); err != nil {
					return cmdutil.AddSourceToErr("creating", info.Source, err)
				}
			}
			obj, err := resource.
				NewHelper(info.Client, info.Mapping).
				DryRun(o.DryRunStrategy == cmdutil.DryRunServer).
				WithFieldManager(o.fieldManager).
				Create(info.Namespace, true, info.Object)
			if err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
			info.Refresh(obj, true)
		}

		count++

		return o.PrintObj(info.Object)
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to create")
	}
	return nil
}

// RunEditOnCreate performs edit on creation
func RunEditOnCreate(f cmdutil.Factory, printFlags *genericclioptions.PrintFlags, recordFlags *genericclioptions.RecordFlags, ioStreams genericclioptions.IOStreams, cmd *cobra.Command, options *resource.FilenameOptions, fieldManager string) error {
	editOptions := editor.NewEditOptions(editor.EditBeforeCreateMode, ioStreams)
	editOptions.FilenameOptions = *options
	editOptions.ValidateOptions = cmdutil.ValidateOptions{
		EnableValidation: cmdutil.GetFlagBool(cmd, "validate"),
	}
	editOptions.PrintFlags = printFlags
	editOptions.ApplyAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)
	editOptions.RecordFlags = recordFlags
	editOptions.FieldManager = "kubectl-create"

	err := editOptions.Complete(f, []string{}, cmd)
	if err != nil {
		return err
	}
	return editOptions.Run()
}

// NameFromCommandArgs is a utility function for commands that assume the first argument is a resource name
func NameFromCommandArgs(cmd *cobra.Command, args []string) (string, error) {
	argsLen := cmd.ArgsLenAtDash()
	// ArgsLenAtDash returns -1 when -- was not specified
	if argsLen == -1 {
		argsLen = len(args)
	}
	if argsLen != 1 {
		return "", cmdutil.UsageErrorf(cmd, "exactly one NAME is required, got %d", argsLen)
	}
	return args[0], nil
}

// CreateSubcommandOptions is an options struct to support create subcommands
type CreateSubcommandOptions struct {
	// PrintFlags holds options necessary for obtaining a printer
	PrintFlags *genericclioptions.PrintFlags
	// Name of resource being created
	Name string
	// StructuredGenerator is the resource generator for the object being created
	StructuredGenerator generate.StructuredGenerator
	DryRunStrategy      cmdutil.DryRunStrategy
	DryRunVerifier      *resource.DryRunVerifier
	CreateAnnotation    bool
	FieldManager        string

	Namespace        string
	EnforceNamespace bool

	Mapper        meta.RESTMapper
	DynamicClient dynamic.Interface

	PrintObj printers.ResourcePrinterFunc

	genericclioptions.IOStreams
}

// NewCreateSubcommandOptions returns initialized CreateSubcommandOptions
func NewCreateSubcommandOptions(ioStreams genericclioptions.IOStreams) *CreateSubcommandOptions {
	return &CreateSubcommandOptions{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// Complete completes all the required options
func (o *CreateSubcommandOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string, generator generate.StructuredGenerator) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	o.Name = name
	o.StructuredGenerator = generator
	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, f.OpenAPIGetter())
	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj kruntime.Object, out io.Writer) error {
		return printer.PrintObj(obj, out)
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.DynamicClient, err = f.DynamicClient()
	if err != nil {
		return err
	}

	o.Mapper, err = f.ToRESTMapper()
	if err != nil {
		return err
	}

	return nil
}

// Run executes a create subcommand using the specified options
func (o *CreateSubcommandOptions) Run() error {
	obj, err := o.StructuredGenerator.StructuredGenerate()
	if err != nil {
		return err
	}
	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, obj, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}
	if o.DryRunStrategy != cmdutil.DryRunClient {
		// create subcommands have compiled knowledge of things they create, so type them directly
		gvks, _, err := scheme.Scheme.ObjectKinds(obj)
		if err != nil {
			return err
		}
		gvk := gvks[0]
		mapping, err := o.Mapper.RESTMapping(schema.GroupKind{Group: gvk.Group, Kind: gvk.Kind}, gvk.Version)
		if err != nil {
			return err
		}

		asUnstructured := &unstructured.Unstructured{}
		if err := scheme.Scheme.Convert(obj, asUnstructured, nil); err != nil {
			return err
		}
		if mapping.Scope.Name() == meta.RESTScopeNameRoot {
			o.Namespace = ""
		}
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(mapping.GroupVersionKind); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		actualObject, err := o.DynamicClient.Resource(mapping.Resource).Namespace(o.Namespace).Create(context.TODO(), asUnstructured, createOptions)
		if err != nil {
			return err
		}

		// ensure we pass a versioned object to the printer
		obj = actualObject
	} else {
		if meta, err := meta.Accessor(obj); err == nil && o.EnforceNamespace {
			meta.SetNamespace(o.Namespace)
		}
	}

	return o.PrintObj(obj, o.Out)
}
