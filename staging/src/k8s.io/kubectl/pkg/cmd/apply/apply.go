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

package apply

import (
	"fmt"
	"io"
	"net/http"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/cmd/delete"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/util/prune"
	"k8s.io/kubectl/pkg/util/templates"
	"k8s.io/kubectl/pkg/validation"
)

// ApplyFlags directly reflect the information that CLI is gathering via flags.  They will be converted to Options, which
// reflect the runtime requirements for the command.  This structure reduces the transformation to wiring and makes
// the logic itself easy to unit test
type ApplyFlags struct {
	Factory cmdutil.Factory

	RecordFlags *genericclioptions.RecordFlags
	PrintFlags  *genericclioptions.PrintFlags

	DeleteFlags *delete.DeleteFlags

	FieldManager   string
	Selector       string
	Prune          bool
	PruneResources []prune.Resource
	All            bool
	Overwrite      bool
	OpenAPIPatch   bool
	PruneWhitelist []string

	genericclioptions.IOStreams
}

// ApplyOptions defines flags and other configuration parameters for the `apply` command
type ApplyOptions struct {
	Recorder genericclioptions.Recorder

	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	DeleteOptions *delete.DeleteOptions

	ServerSideApply         bool
	ForceConflicts          bool
	FieldManager            string
	Selector                string
	DryRunStrategy          cmdutil.DryRunStrategy
	DryRunVerifier          *resource.QueryParamVerifier
	FieldValidationVerifier *resource.QueryParamVerifier
	Prune                   bool
	PruneResources          []prune.Resource
	cmdBaseName             string
	All                     bool
	Overwrite               bool
	OpenAPIPatch            bool
	PruneWhitelist          []string

	ValidationDirective string
	Validator           validation.Schema
	Builder             *resource.Builder
	Mapper              meta.RESTMapper
	DynamicClient       dynamic.Interface
	OpenAPISchema       openapi.Resources

	Namespace        string
	EnforceNamespace bool

	genericclioptions.IOStreams

	// Objects (and some denormalized data) which are to be
	// applied. The standard way to fill in this structure
	// is by calling "GetObjects()", which will use the
	// resource builder if "objectsCached" is false. The other
	// way to set this field is to use "SetObjects()".
	// Subsequent calls to "GetObjects()" after setting would
	// not call the resource builder; only return the set objects.
	objects       []*resource.Info
	objectsCached bool

	// Stores visited objects/namespaces for later use
	// calculating the set of objects to prune.
	VisitedUids       sets.String
	VisitedNamespaces sets.String

	// Function run after the objects are generated and
	// stored in the "objects" field, but before the
	// apply is run on these objects.
	PreProcessorFn func() error
	// Function run after all objects have been applied.
	// The standard PostProcessorFn is "PrintAndPrunePostProcessor()".
	PostProcessorFn func() error
}

var (
	applyLong = templates.LongDesc(i18n.T(`
		Apply a configuration to a resource by file name or stdin.
		The resource name must be specified. This resource will be created if it doesn't exist yet.
		To use 'apply', always create the resource initially with either 'apply' or 'create --save-config'.

		JSON and YAML formats are accepted.

		Alpha Disclaimer: the --prune functionality is not yet complete. Do not use unless you are aware of what the current state is. See https://issues.k8s.io/34274.`))

	applyExample = templates.Examples(i18n.T(`
		# Apply the configuration in pod.json to a pod
		kubectl apply -f ./pod.json

		# Apply resources from a directory containing kustomization.yaml - e.g. dir/kustomization.yaml
		kubectl apply -k dir/

		# Apply the JSON passed into stdin to a pod
		cat pod.json | kubectl apply -f -

		# Apply the configuration from all files that end with '.json' - i.e. expand wildcard characters in file names
		kubectl apply -f '*.json'

		# Note: --prune is still in Alpha
		# Apply the configuration in manifest.yaml that matches label app=nginx and delete all other resources that are not in the file and match label app=nginx
		kubectl apply --prune -f manifest.yaml -l app=nginx

		# Apply the configuration in manifest.yaml and delete all the other config maps that are not in the file
		kubectl apply --prune -f manifest.yaml --all --prune-whitelist=core/v1/ConfigMap`))

	warningNoLastAppliedConfigAnnotation = "Warning: resource %[1]s is missing the %[2]s annotation which is required by %[3]s apply. %[3]s apply should only be used on resources created declaratively by either %[3]s create --save-config or %[3]s apply. The missing annotation will be patched automatically.\n"
	warningChangesOnDeletingResource     = "Warning: Detected changes to resource %[1]s which is currently being deleted.\n"
)

// NewApplyFlags returns a default ApplyFlags
func NewApplyFlags(f cmdutil.Factory, streams genericclioptions.IOStreams) *ApplyFlags {
	return &ApplyFlags{
		Factory:     f,
		RecordFlags: genericclioptions.NewRecordFlags(),
		DeleteFlags: delete.NewDeleteFlags("The files that contain the configurations to apply."),
		PrintFlags:  genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),

		Overwrite:    true,
		OpenAPIPatch: true,

		IOStreams: streams,
	}
}

// NewCmdApply creates the `apply` command
func NewCmdApply(baseName string, f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	flags := NewApplyFlags(f, ioStreams)

	cmd := &cobra.Command{
		Use:                   "apply (-f FILENAME | -k DIRECTORY)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Apply a configuration to a resource by file name or stdin"),
		Long:                  applyLong,
		Example:               applyExample,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(cmd, baseName, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	flags.AddFlags(cmd)

	// apply subcommands
	cmd.AddCommand(NewCmdApplyViewLastApplied(flags.Factory, flags.IOStreams))
	cmd.AddCommand(NewCmdApplySetLastApplied(flags.Factory, flags.IOStreams))
	cmd.AddCommand(NewCmdApplyEditLastApplied(flags.Factory, flags.IOStreams))

	return cmd
}

// AddFlags registers flags for a cli
func (flags *ApplyFlags) AddFlags(cmd *cobra.Command) {
	// bind flag structs
	flags.DeleteFlags.AddFlags(cmd)
	flags.RecordFlags.AddFlags(cmd)
	flags.PrintFlags.AddFlags(cmd)

	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddServerSideApplyFlags(cmd)
	cmdutil.AddFieldManagerFlagVar(cmd, &flags.FieldManager, FieldManagerClientSideApply)
	cmdutil.AddLabelSelectorFlagVar(cmd, &flags.Selector)

	cmd.Flags().BoolVar(&flags.Overwrite, "overwrite", flags.Overwrite, "Automatically resolve conflicts between the modified and live configuration by using values from the modified configuration")
	cmd.Flags().BoolVar(&flags.Prune, "prune", flags.Prune, "Automatically delete resource objects, that do not appear in the configs and are created by either apply or create --save-config. Should be used with either -l or --all.")
	cmd.Flags().BoolVar(&flags.All, "all", flags.All, "Select all resources in the namespace of the specified resource types.")
	cmd.Flags().StringArrayVar(&flags.PruneWhitelist, "prune-whitelist", flags.PruneWhitelist, "Overwrite the default whitelist with <group/version/kind> for --prune")
	cmd.Flags().BoolVar(&flags.OpenAPIPatch, "openapi-patch", flags.OpenAPIPatch, "If true, use openapi to calculate diff when the openapi presents and the resource can be found in the openapi spec. Otherwise, fall back to use baked-in types.")
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *ApplyFlags) ToOptions(cmd *cobra.Command, baseName string, args []string) (*ApplyOptions, error) {
	if len(args) != 0 {
		return nil, cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}

	serverSideApply := cmdutil.GetServerSideApplyFlag(cmd)
	forceConflicts := cmdutil.GetForceConflictsFlag(cmd)
	dryRunStrategy, err := cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return nil, err
	}

	dynamicClient, err := flags.Factory.DynamicClient()
	if err != nil {
		return nil, err
	}

	dryRunVerifier := resource.NewQueryParamVerifier(dynamicClient, flags.Factory.OpenAPIGetter(), resource.QueryParamDryRun)
	fieldValidationVerifier := resource.NewQueryParamVerifier(dynamicClient, flags.Factory.OpenAPIGetter(), resource.QueryParamFieldValidation)
	fieldManager := GetApplyFieldManagerFlag(cmd, serverSideApply)

	// allow for a success message operation to be specified at print time
	toPrinter := func(operation string) (printers.ResourcePrinter, error) {
		flags.PrintFlags.NamePrintFlags.Operation = operation
		cmdutil.PrintFlagsWithDryRunStrategy(flags.PrintFlags, dryRunStrategy)
		return flags.PrintFlags.ToPrinter()
	}

	flags.RecordFlags.Complete(cmd)
	recorder, err := flags.RecordFlags.ToRecorder()
	if err != nil {
		return nil, err
	}

	deleteOptions, err := flags.DeleteFlags.ToOptions(dynamicClient, flags.IOStreams)
	if err != nil {
		return nil, err
	}

	err = deleteOptions.FilenameOptions.RequireFilenameOrKustomize()
	if err != nil {
		return nil, err
	}

	openAPISchema, _ := flags.Factory.OpenAPISchema()

	validationDirective, err := cmdutil.GetValidationDirective(cmd)
	if err != nil {
		return nil, err
	}
	validator, err := flags.Factory.Validator(validationDirective, fieldValidationVerifier)
	if err != nil {
		return nil, err
	}
	builder := flags.Factory.NewBuilder()
	mapper, err := flags.Factory.ToRESTMapper()
	if err != nil {
		return nil, err
	}

	namespace, enforceNamespace, err := flags.Factory.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return nil, err
	}

	if flags.Prune {
		flags.PruneResources, err = prune.ParseResources(mapper, flags.PruneWhitelist)
		if err != nil {
			return nil, err
		}
	}

	o := &ApplyOptions{
		// 	Store baseName for use in printing warnings / messages involving the base command name.
		// 	This is useful for downstream command that wrap this one.
		cmdBaseName: baseName,

		PrintFlags: flags.PrintFlags,

		DeleteOptions:   deleteOptions,
		ToPrinter:       toPrinter,
		ServerSideApply: serverSideApply,
		ForceConflicts:  forceConflicts,
		FieldManager:    fieldManager,
		Selector:        flags.Selector,
		DryRunStrategy:  dryRunStrategy,
		DryRunVerifier:  dryRunVerifier,
		Prune:           flags.Prune,
		PruneResources:  flags.PruneResources,
		All:             flags.All,
		Overwrite:       flags.Overwrite,
		OpenAPIPatch:    flags.OpenAPIPatch,
		PruneWhitelist:  flags.PruneWhitelist,

		Recorder:            recorder,
		Namespace:           namespace,
		EnforceNamespace:    enforceNamespace,
		Validator:           validator,
		ValidationDirective: validationDirective,
		Builder:             builder,
		Mapper:              mapper,
		DynamicClient:       dynamicClient,
		OpenAPISchema:       openAPISchema,

		IOStreams: flags.IOStreams,

		objects:       []*resource.Info{},
		objectsCached: false,

		VisitedUids:       sets.NewString(),
		VisitedNamespaces: sets.NewString(),
	}

	o.PostProcessorFn = o.PrintAndPrunePostProcessor()

	return o, nil
}

// Validate verifies if ApplyOptions are valid and without conflicts.
func (o *ApplyOptions) Validate() error {
	if o.ForceConflicts && !o.ServerSideApply {
		return fmt.Errorf("--force-conflicts only works with --server-side")
	}

	if o.DryRunStrategy == cmdutil.DryRunClient && o.ServerSideApply {
		return fmt.Errorf("--dry-run=client doesn't work with --server-side (did you mean --dry-run=server instead?)")
	}

	if o.ServerSideApply && o.DeleteOptions.ForceDeletion {
		return fmt.Errorf("--force cannot be used with --server-side")
	}

	if o.DryRunStrategy == cmdutil.DryRunServer && o.DeleteOptions.ForceDeletion {
		return fmt.Errorf("--dry-run=server cannot be used with --force")
	}

	if o.All && len(o.Selector) > 0 {
		return fmt.Errorf("cannot set --all and --selector at the same time")
	}

	if o.Prune && !o.All && o.Selector == "" {
		return fmt.Errorf("all resources selected for prune without explicitly passing --all. To prune all resources, pass the --all flag. If you did not mean to prune all resources, specify a label selector")
	}

	// Do not force the recreation of an object(s) if we're pruning; this can cause
	// undefined behavior since object UID's change.
	if o.Prune && o.DeleteOptions.ForceDeletion {
		return fmt.Errorf("--force cannot be used with --prune")
	}

	return nil
}

func isIncompatibleServerError(err error) bool {
	// 415: Unsupported media type means we're talking to a server which doesn't
	// support server-side apply.
	if _, ok := err.(*errors.StatusError); !ok {
		// Non-StatusError means the error isn't because the server is incompatible.
		return false
	}
	return err.(*errors.StatusError).Status().Code == http.StatusUnsupportedMediaType
}

// GetObjects returns a (possibly cached) version of all the valid objects to apply
// as a slice of pointer to resource.Info and an error if one or more occurred.
// IMPORTANT: This function can return both valid objects AND an error, since
// "ContinueOnError" is set on the builder. This function should not be called
// until AFTER the "complete" and "validate" methods have been called to ensure that
// the ApplyOptions is filled in and valid.
func (o *ApplyOptions) GetObjects() ([]*resource.Info, error) {
	var err error = nil
	if !o.objectsCached {
		r := o.Builder.
			Unstructured().
			Schema(o.Validator).
			ContinueOnError().
			NamespaceParam(o.Namespace).DefaultNamespace().
			FilenameParam(o.EnforceNamespace, &o.DeleteOptions.FilenameOptions).
			LabelSelectorParam(o.Selector).
			Flatten().
			Do()
		o.objects, err = r.Infos()
		o.objectsCached = true
	}
	return o.objects, err
}

// SetObjects stores the set of objects (as resource.Info) to be
// subsequently applied.
func (o *ApplyOptions) SetObjects(infos []*resource.Info) {
	o.objects = infos
	o.objectsCached = true
}

// Run executes the `apply` command.
func (o *ApplyOptions) Run() error {
	if o.PreProcessorFn != nil {
		klog.V(4).Infof("Running apply pre-processor function")
		if err := o.PreProcessorFn(); err != nil {
			return err
		}
	}

	// Enforce CLI specified namespace on server request.
	if o.EnforceNamespace {
		o.VisitedNamespaces.Insert(o.Namespace)
	}

	// Generates the objects using the resource builder if they have not
	// already been stored by calling "SetObjects()" in the pre-processor.
	errs := []error{}
	infos, err := o.GetObjects()
	if err != nil {
		errs = append(errs, err)
	}
	if len(infos) == 0 && len(errs) == 0 {
		return fmt.Errorf("no objects passed to apply")
	}
	// Iterate through all objects, applying each one.
	for _, info := range infos {
		if err := o.applyOneObject(info); err != nil {
			errs = append(errs, err)
		}
	}
	// If any errors occurred during apply, then return error (or
	// aggregate of errors).
	if len(errs) == 1 {
		return errs[0]
	}
	if len(errs) > 1 {
		return utilerrors.NewAggregate(errs)
	}

	if o.PostProcessorFn != nil {
		klog.V(4).Infof("Running apply post-processor function")
		if err := o.PostProcessorFn(); err != nil {
			return err
		}
	}

	return nil
}

func (o *ApplyOptions) applyOneObject(info *resource.Info) error {
	o.MarkNamespaceVisited(info)

	if err := o.Recorder.Record(info.Object); err != nil {
		klog.V(4).Infof("error recording current command: %v", err)
	}

	if len(info.Name) == 0 {
		metadata, _ := meta.Accessor(info.Object)
		generatedName := metadata.GetGenerateName()
		if len(generatedName) > 0 {
			return fmt.Errorf("from %s: cannot use generate name with apply", generatedName)
		}
	}

	helper := resource.NewHelper(info.Client, info.Mapping).
		DryRun(o.DryRunStrategy == cmdutil.DryRunServer).
		WithFieldManager(o.FieldManager).
		WithFieldValidation(o.ValidationDirective)

	if o.DryRunStrategy == cmdutil.DryRunServer {
		// Ensure the APIServer supports server-side dry-run for the resource,
		// otherwise fail early.
		// For APIServers that don't support server-side dry-run will persist
		// changes.
		if err := o.DryRunVerifier.HasSupport(info.Mapping.GroupVersionKind); err != nil {
			return err
		}
	}

	if o.ServerSideApply {
		// Send the full object to be applied on the server side.
		data, err := runtime.Encode(unstructured.UnstructuredJSONScheme, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr("serverside-apply", info.Source, err)
		}

		options := metav1.PatchOptions{
			Force: &o.ForceConflicts,
		}
		obj, err := helper.Patch(
			info.Namespace,
			info.Name,
			types.ApplyPatchType,
			data,
			&options,
		)
		if err != nil {
			if isIncompatibleServerError(err) {
				err = fmt.Errorf("Server-side apply not available on the server: (%v)", err)
			}
			if errors.IsConflict(err) {
				err = fmt.Errorf(`%v
Please review the fields above--they currently have other managers. Here
are the ways you can resolve this warning:
* If you intend to manage all of these fields, please re-run the apply
  command with the `+"`--force-conflicts`"+` flag.
* If you do not intend to manage all of the fields, please edit your
  manifest to remove references to the fields that should keep their
  current managers.
* You may co-own fields by updating your manifest to match the existing
  value; in this case, you'll become the manager if the other manager(s)
  stop managing the field (remove it from their configuration).
See https://kubernetes.io/docs/reference/using-api/server-side-apply/#conflicts`, err)
			}
			return err
		}

		info.Refresh(obj, true)

		WarnIfDeleting(info.Object, o.ErrOut)

		if err := o.MarkObjectVisited(info); err != nil {
			return err
		}

		if o.shouldPrintObject() {
			return nil
		}

		printer, err := o.ToPrinter("serverside-applied")
		if err != nil {
			return err
		}

		if err = printer.PrintObj(info.Object, o.Out); err != nil {
			return err
		}
		return nil
	}

	// Get the modified configuration of the object. Embed the result
	// as an annotation in the modified configuration, so that it will appear
	// in the patch sent to the server.
	modified, err := util.GetModifiedConfiguration(info.Object, true, unstructured.UnstructuredJSONScheme)
	if err != nil {
		return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving modified configuration from:\n%s\nfor:", info.String()), info.Source, err)
	}

	if err := info.Get(); err != nil {
		if !errors.IsNotFound(err) {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%s\nfrom server for:", info.String()), info.Source, err)
		}

		// Create the resource if it doesn't exist
		// First, update the annotation used by kubectl apply
		if err := util.CreateApplyAnnotation(info.Object, unstructured.UnstructuredJSONScheme); err != nil {
			return cmdutil.AddSourceToErr("creating", info.Source, err)
		}

		if o.DryRunStrategy != cmdutil.DryRunClient {
			// Then create the resource and skip the three-way merge
			obj, err := helper.Create(info.Namespace, true, info.Object)
			if err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
			info.Refresh(obj, true)
		}

		if err := o.MarkObjectVisited(info); err != nil {
			return err
		}

		if o.shouldPrintObject() {
			return nil
		}

		printer, err := o.ToPrinter("created")
		if err != nil {
			return err
		}
		if err = printer.PrintObj(info.Object, o.Out); err != nil {
			return err
		}
		return nil
	}

	if err := o.MarkObjectVisited(info); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		metadata, _ := meta.Accessor(info.Object)
		annotationMap := metadata.GetAnnotations()
		if _, ok := annotationMap[corev1.LastAppliedConfigAnnotation]; !ok {
			fmt.Fprintf(o.ErrOut, warningNoLastAppliedConfigAnnotation, info.ObjectName(), corev1.LastAppliedConfigAnnotation, o.cmdBaseName)
		}

		patcher, err := newPatcher(o, info, helper)
		if err != nil {
			return err
		}
		patchBytes, patchedObject, err := patcher.Patch(info.Object, modified, info.Source, info.Namespace, info.Name, o.ErrOut)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("applying patch:\n%s\nto:\n%v\nfor:", patchBytes, info), info.Source, err)
		}

		info.Refresh(patchedObject, true)

		WarnIfDeleting(info.Object, o.ErrOut)

		if string(patchBytes) == "{}" && !o.shouldPrintObject() {
			printer, err := o.ToPrinter("unchanged")
			if err != nil {
				return err
			}
			if err = printer.PrintObj(info.Object, o.Out); err != nil {
				return err
			}
			return nil
		}
	}

	if o.shouldPrintObject() {
		return nil
	}

	printer, err := o.ToPrinter("configured")
	if err != nil {
		return err
	}
	if err = printer.PrintObj(info.Object, o.Out); err != nil {
		return err
	}

	return nil
}

func (o *ApplyOptions) shouldPrintObject() bool {
	// Print object only if output format other than "name" is specified
	shouldPrint := false
	output := *o.PrintFlags.OutputFormat
	shortOutput := output == "name"
	if len(output) > 0 && !shortOutput {
		shouldPrint = true
	}
	return shouldPrint
}

func (o *ApplyOptions) printObjects() error {

	if !o.shouldPrintObject() {
		return nil
	}

	infos, err := o.GetObjects()
	if err != nil {
		return err
	}

	if len(infos) > 0 {
		printer, err := o.ToPrinter("")
		if err != nil {
			return err
		}

		objToPrint := infos[0].Object
		if len(infos) > 1 {
			objs := []runtime.Object{}
			for _, info := range infos {
				objs = append(objs, info.Object)
			}
			list := &corev1.List{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{},
			}
			if err := meta.SetList(list, objs); err != nil {
				return err
			}

			objToPrint = list
		}
		if err := printer.PrintObj(objToPrint, o.Out); err != nil {
			return err
		}
	}

	return nil
}

// MarkNamespaceVisited keeps track of which namespaces the applied
// objects belong to. Used for pruning.
func (o *ApplyOptions) MarkNamespaceVisited(info *resource.Info) {
	if info.Namespaced() {
		o.VisitedNamespaces.Insert(info.Namespace)
	}
}

// MarkObjectVisited keeps track of UIDs of the applied
// objects. Used for pruning.
func (o *ApplyOptions) MarkObjectVisited(info *resource.Info) error {
	metadata, err := meta.Accessor(info.Object)
	if err != nil {
		return err
	}
	o.VisitedUids.Insert(string(metadata.GetUID()))
	return nil
}

// PrintAndPrunePostProcessor returns a function which meets the PostProcessorFn
// function signature. This returned function prints all the
// objects as a list (if configured for that), and prunes the
// objects not applied. The returned function is the standard
// apply post processor.
func (o *ApplyOptions) PrintAndPrunePostProcessor() func() error {

	return func() error {
		if err := o.printObjects(); err != nil {
			return err
		}

		if o.Prune {
			p := newPruner(o)
			return p.pruneAll(o)
		}

		return nil
	}
}

const (
	// FieldManagerClientSideApply is the default client-side apply field manager.
	//
	// The default field manager is not `kubectl-apply` to distinguish from
	// server-side apply.
	FieldManagerClientSideApply = "kubectl-client-side-apply"
	// The default server-side apply field manager is `kubectl`
	// instead of a field manager like `kubectl-server-side-apply`
	// for backward compatibility to not conflict with old versions
	// of kubectl server-side apply where `kubectl` has already been the field manager.
	fieldManagerServerSideApply = "kubectl"
)

// GetApplyFieldManagerFlag gets the field manager for kubectl apply
// if it is not set.
//
// The default field manager is not `kubectl-apply` to distinguish between
// client-side and server-side apply.
func GetApplyFieldManagerFlag(cmd *cobra.Command, serverSide bool) string {
	// The field manager flag was set
	if cmd.Flag("field-manager").Changed {
		return cmdutil.GetFlagString(cmd, "field-manager")
	}

	if serverSide {
		return fieldManagerServerSideApply
	}

	return FieldManagerClientSideApply
}

// WarnIfDeleting prints a warning if a resource is being deleted
func WarnIfDeleting(obj runtime.Object, stderr io.Writer) {
	metadata, _ := meta.Accessor(obj)
	if metadata != nil && metadata.GetDeletionTimestamp() != nil {
		// just warn the user about the conflict
		fmt.Fprintf(stderr, warningChangesOnDeletingResource, metadata.GetName())
	}
}
