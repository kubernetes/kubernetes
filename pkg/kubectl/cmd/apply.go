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
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/jsonmergepatch"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/printers"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	"k8s.io/client-go/dynamic"
	oapi "k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/validation"
)

type ApplyOptions struct {
	RecordFlags *genericclioptions.RecordFlags
	Recorder    genericclioptions.Recorder

	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	DeleteFlags   *DeleteFlags
	DeleteOptions *DeleteOptions

	Selector                   string
	DryRun                     bool
	ServerDryRun               bool
	Prune                      bool
	PruneResources             []pruneResource
	cmdBaseName                string
	All                        bool
	Overwrite                  bool
	OpenApiPatch               bool
	PruneWhitelist             []string
	ShouldIncludeUninitialized bool

	Validator     validation.Schema
	Builder       *resource.Builder
	Mapper        meta.RESTMapper
	DynamicClient dynamic.Interface
	OpenAPISchema openapi.Resources

	Namespace        string
	EnforceNamespace bool

	genericclioptions.IOStreams
}

const (
	// maxPatchRetry is the maximum number of conflicts retry for during a patch operation before returning failure
	maxPatchRetry = 5
	// backOffPeriod is the period to back off when apply patch resutls in error.
	backOffPeriod = 1 * time.Second
	// how many times we can retry before back off
	triesBeforeBackOff = 1
)

var (
	applyLong = templates.LongDesc(i18n.T(`
		Apply a configuration to a resource by filename or stdin.
		The resource name must be specified. This resource will be created if it doesn't exist yet.
		To use 'apply', always create the resource initially with either 'apply' or 'create --save-config'.

		JSON and YAML formats are accepted.

		Alpha Disclaimer: the --prune functionality is not yet complete. Do not use unless you are aware of what the current state is. See https://issues.k8s.io/34274.`))

	applyExample = templates.Examples(i18n.T(`
		# Apply the configuration in pod.json to a pod.
		kubectl apply -f ./pod.json

		# Apply the JSON passed into stdin to a pod.
		cat pod.json | kubectl apply -f -

		# Note: --prune is still in Alpha
		# Apply the configuration in manifest.yaml that matches label app=nginx and delete all the other resources that are not in the file and match label app=nginx.
		kubectl apply --prune -f manifest.yaml -l app=nginx

		# Apply the configuration in manifest.yaml and delete all the other configmaps that are not in the file.
		kubectl apply --prune -f manifest.yaml --all --prune-whitelist=core/v1/ConfigMap`))

	warningNoLastAppliedConfigAnnotation = "Warning: %[1]s apply should be used on resource created by either %[1]s create --save-config or %[1]s apply\n"
)

func NewApplyOptions(ioStreams genericclioptions.IOStreams) *ApplyOptions {
	return &ApplyOptions{
		RecordFlags: genericclioptions.NewRecordFlags(),
		DeleteFlags: NewDeleteFlags("that contains the configuration to apply"),
		PrintFlags:  genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),

		Overwrite:    true,
		OpenApiPatch: true,

		Recorder: genericclioptions.NoopRecorder{},

		IOStreams: ioStreams,
	}
}

func NewCmdApply(baseName string, f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewApplyOptions(ioStreams)

	// Store baseName for use in printing warnings / messages involving the base command name.
	// This is useful for downstream command that wrap this one.
	o.cmdBaseName = baseName

	cmd := &cobra.Command{
		Use: "apply -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Apply a configuration to a resource by filename or stdin"),
		Long:    applyLong,
		Example: applyExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(validateArgs(cmd, args))
			cmdutil.CheckErr(validatePruneAll(o.Prune, o.All, o.Selector))
			cmdutil.CheckErr(o.Run())
		},
	}

	// bind flag structs
	o.DeleteFlags.AddFlags(cmd)
	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.MarkFlagRequired("filename")
	cmd.Flags().BoolVar(&o.Overwrite, "overwrite", o.Overwrite, "Automatically resolve conflicts between the modified and live configuration by using values from the modified configuration")
	cmd.Flags().BoolVar(&o.Prune, "prune", o.Prune, "Automatically delete resource objects, including the uninitialized ones, that do not appear in the configs and are created by either apply or create --save-config. Should be used with either -l or --all.")
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.All, "all", o.All, "Select all resources in the namespace of the specified resource types.")
	cmd.Flags().StringArrayVar(&o.PruneWhitelist, "prune-whitelist", o.PruneWhitelist, "Overwrite the default whitelist with <group/version/kind> for --prune")
	cmd.Flags().BoolVar(&o.OpenApiPatch, "openapi-patch", o.OpenApiPatch, "If true, use openapi to calculate diff when the openapi presents and the resource can be found in the openapi spec. Otherwise, fall back to use baked-in types.")
	cmd.Flags().BoolVar(&o.ServerDryRun, "server-dry-run", o.ServerDryRun, "If true, request will be sent to server with dry-run flag, which means the modifications won't be persisted. This is an alpha feature and flag.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)

	// apply subcommands
	cmd.AddCommand(NewCmdApplyViewLastApplied(f, ioStreams))
	cmd.AddCommand(NewCmdApplySetLastApplied(f, ioStreams))
	cmd.AddCommand(NewCmdApplyEditLastApplied(f, ioStreams))

	return cmd
}

func (o *ApplyOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	o.DryRun = cmdutil.GetDryRunFlag(cmd)

	if o.DryRun && o.ServerDryRun {
		return fmt.Errorf("--dry-run and --server-dry-run can't be used together")
	}

	// allow for a success message operation to be specified at print time
	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		if o.DryRun {
			o.PrintFlags.Complete("%s (dry run)")
		}
		if o.ServerDryRun {
			o.PrintFlags.Complete("%s (server dry run)")
		}
		return o.PrintFlags.ToPrinter()
	}

	var err error
	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DeleteOptions = o.DeleteFlags.ToOptions(dynamicClient, o.IOStreams)
	o.ShouldIncludeUninitialized = cmdutil.ShouldIncludeUninitialized(cmd, o.Prune)

	o.OpenAPISchema, _ = f.OpenAPISchema()
	o.Validator, err = f.Validator(cmdutil.GetFlagBool(cmd, "validate"))
	o.Builder = f.NewBuilder()
	o.Mapper, err = f.ToRESTMapper()
	if err != nil {
		return err
	}

	o.DynamicClient, err = f.DynamicClient()
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	return nil
}

func validateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}
	return nil
}

func validatePruneAll(prune, all bool, selector string) error {
	if all && len(selector) > 0 {
		return fmt.Errorf("cannot set --all and --selector at the same time")
	}
	if prune && !all && selector == "" {
		return fmt.Errorf("all resources selected for prune without explicitly passing --all. To prune all resources, pass the --all flag. If you did not mean to prune all resources, specify a label selector.")
	}
	return nil
}

func parsePruneResources(mapper meta.RESTMapper, gvks []string) ([]pruneResource, error) {
	pruneResources := []pruneResource{}
	for _, groupVersionKind := range gvks {
		gvk := strings.Split(groupVersionKind, "/")
		if len(gvk) != 3 {
			return nil, fmt.Errorf("invalid GroupVersionKind format: %v, please follow <group/version/kind>", groupVersionKind)
		}

		if gvk[0] == "core" {
			gvk[0] = ""
		}
		mapping, err := mapper.RESTMapping(schema.GroupKind{Group: gvk[0], Kind: gvk[2]}, gvk[1])
		if err != nil {
			return pruneResources, err
		}
		var namespaced bool
		namespaceScope := mapping.Scope.Name()
		switch namespaceScope {
		case meta.RESTScopeNameNamespace:
			namespaced = true
		case meta.RESTScopeNameRoot:
			namespaced = false
		default:
			return pruneResources, fmt.Errorf("Unknown namespace scope: %q", namespaceScope)
		}

		pruneResources = append(pruneResources, pruneResource{gvk[0], gvk[1], gvk[2], namespaced})
	}
	return pruneResources, nil
}

func (o *ApplyOptions) Run() error {
	var openapiSchema openapi.Resources
	if o.OpenApiPatch {
		openapiSchema = o.OpenAPISchema
	}

	// include the uninitialized objects by default if --prune is true
	// unless explicitly set --include-uninitialized=false
	r := o.Builder.
		Unstructured().
		Schema(o.Validator).
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.DeleteOptions.FilenameOptions).
		LabelSelectorParam(o.Selector).
		IncludeUninitialized(o.ShouldIncludeUninitialized).
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	var err error
	if o.Prune {
		o.PruneResources, err = parsePruneResources(o.Mapper, o.PruneWhitelist)
		if err != nil {
			return err
		}
	}

	output := *o.PrintFlags.OutputFormat
	shortOutput := output == "name"

	visitedUids := sets.NewString()
	visitedNamespaces := sets.NewString()

	var objs []runtime.Object

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if info.Namespaced() {
			visitedNamespaces.Insert(info.Namespace)
		}

		if err := o.Recorder.Record(info.Object); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		}

		// Get the modified configuration of the object. Embed the result
		// as an annotation in the modified configuration, so that it will appear
		// in the patch sent to the server.
		modified, err := kubectl.GetModifiedConfiguration(info.Object, true, unstructured.UnstructuredJSONScheme)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving modified configuration from:\n%s\nfor:", info.String()), info.Source, err)
		}

		// Print object only if output format other than "name" is specified
		printObject := len(output) > 0 && !shortOutput

		if err := info.Get(); err != nil {
			if !errors.IsNotFound(err) {
				return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%s\nfrom server for:", info.String()), info.Source, err)
			}
			// Create the resource if it doesn't exist
			// First, update the annotation used by kubectl apply
			if err := kubectl.CreateApplyAnnotation(info.Object, unstructured.UnstructuredJSONScheme); err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}

			if !o.DryRun {
				// Then create the resource and skip the three-way merge
				options := metav1.CreateOptions{}
				if o.ServerDryRun {
					options.DryRun = []string{metav1.DryRunAll}
				}
				obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, info.Object, &options)
				if err != nil {
					return cmdutil.AddSourceToErr("creating", info.Source, err)
				}
				info.Refresh(obj, true)
				metadata, err := meta.Accessor(info.Object)
				if err != nil {
					return err
				}
				visitedUids.Insert(string(metadata.GetUID()))
			}

			count++

			if printObject {
				objs = append(objs, info.Object)
				return nil
			}

			printer, err := o.ToPrinter("created")
			if err != nil {
				return err
			}
			return printer.PrintObj(info.Object, o.Out)
		}

		if !o.DryRun {
			metadata, err := meta.Accessor(info.Object)
			if err != nil {
				return err
			}

			annotationMap := metadata.GetAnnotations()
			if _, ok := annotationMap[corev1.LastAppliedConfigAnnotation]; !ok {
				fmt.Fprintf(o.ErrOut, warningNoLastAppliedConfigAnnotation, o.cmdBaseName)
			}

			helper := resource.NewHelper(info.Client, info.Mapping)
			patcher := &patcher{
				mapping:       info.Mapping,
				helper:        helper,
				dynamicClient: o.DynamicClient,
				overwrite:     o.Overwrite,
				backOff:       clockwork.NewRealClock(),
				force:         o.DeleteOptions.ForceDeletion,
				cascade:       o.DeleteOptions.Cascade,
				timeout:       o.DeleteOptions.Timeout,
				gracePeriod:   o.DeleteOptions.GracePeriod,
				serverDryRun:  o.ServerDryRun,
				openapiSchema: openapiSchema,
			}

			patchBytes, patchedObject, err := patcher.patch(info.Object, modified, info.Source, info.Namespace, info.Name, o.ErrOut)
			if err != nil {
				return cmdutil.AddSourceToErr(fmt.Sprintf("applying patch:\n%s\nto:\n%v\nfor:", patchBytes, info), info.Source, err)
			}

			info.Refresh(patchedObject, true)

			visitedUids.Insert(string(metadata.GetUID()))

			if string(patchBytes) == "{}" && !printObject {
				count++

				printer, err := o.ToPrinter("unchanged")
				if err != nil {
					return err
				}
				return printer.PrintObj(info.Object, o.Out)
			}
		}
		count++

		if printObject {
			objs = append(objs, info.Object)
			return nil
		}

		printer, err := o.ToPrinter("configured")
		if err != nil {
			return err
		}
		return printer.PrintObj(info.Object, o.Out)
	})
	if err != nil {
		return err
	}

	if count == 0 {
		return fmt.Errorf("no objects passed to apply")
	}

	// print objects
	if len(objs) > 0 {
		printer, err := o.ToPrinter("")
		if err != nil {
			return err
		}

		objToPrint := objs[0]
		if len(objs) > 1 {
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

	if !o.Prune {
		return nil
	}

	p := pruner{
		mapper:        o.Mapper,
		dynamicClient: o.DynamicClient,

		labelSelector: o.Selector,
		visitedUids:   visitedUids,

		cascade:      o.DeleteOptions.Cascade,
		dryRun:       o.DryRun,
		serverDryRun: o.ServerDryRun,
		gracePeriod:  o.DeleteOptions.GracePeriod,

		toPrinter: o.ToPrinter,

		out: o.Out,
	}

	namespacedRESTMappings, nonNamespacedRESTMappings, err := getRESTMappings(o.Mapper, &(o.PruneResources))
	if err != nil {
		return fmt.Errorf("error retrieving RESTMappings to prune: %v", err)
	}

	for n := range visitedNamespaces {
		for _, m := range namespacedRESTMappings {
			if err := p.prune(n, m, o.ShouldIncludeUninitialized); err != nil {
				return fmt.Errorf("error pruning namespaced object %v: %v", m.GroupVersionKind, err)
			}
		}
	}
	for _, m := range nonNamespacedRESTMappings {
		if err := p.prune(metav1.NamespaceNone, m, o.ShouldIncludeUninitialized); err != nil {
			return fmt.Errorf("error pruning nonNamespaced object %v: %v", m.GroupVersionKind, err)
		}
	}

	return nil
}

type pruneResource struct {
	group      string
	version    string
	kind       string
	namespaced bool
}

func (pr pruneResource) String() string {
	return fmt.Sprintf("%v/%v, Kind=%v, Namespaced=%v", pr.group, pr.version, pr.kind, pr.namespaced)
}

func getRESTMappings(mapper meta.RESTMapper, pruneResources *[]pruneResource) (namespaced, nonNamespaced []*meta.RESTMapping, err error) {
	if len(*pruneResources) == 0 {
		// default whitelist
		// TODO: need to handle the older api versions - e.g. v1beta1 jobs. Github issue: #35991
		*pruneResources = []pruneResource{
			{"", "v1", "ConfigMap", true},
			{"", "v1", "Endpoints", true},
			{"", "v1", "Namespace", false},
			{"", "v1", "PersistentVolumeClaim", true},
			{"", "v1", "PersistentVolume", false},
			{"", "v1", "Pod", true},
			{"", "v1", "ReplicationController", true},
			{"", "v1", "Secret", true},
			{"", "v1", "Service", true},
			{"batch", "v1", "Job", true},
			{"batch", "v1beta1", "CronJob", true},
			{"extensions", "v1beta1", "DaemonSet", true},
			{"extensions", "v1beta1", "Deployment", true},
			{"extensions", "v1beta1", "Ingress", true},
			{"extensions", "v1beta1", "ReplicaSet", true},
			{"apps", "v1beta1", "StatefulSet", true},
			{"apps", "v1beta1", "Deployment", true},
		}
	}

	for _, resource := range *pruneResources {
		addedMapping, err := mapper.RESTMapping(schema.GroupKind{Group: resource.group, Kind: resource.kind}, resource.version)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid resource %v: %v", resource, err)
		}
		if resource.namespaced {
			namespaced = append(namespaced, addedMapping)
		} else {
			nonNamespaced = append(nonNamespaced, addedMapping)
		}
	}

	return namespaced, nonNamespaced, nil
}

type pruner struct {
	mapper        meta.RESTMapper
	dynamicClient dynamic.Interface

	visitedUids   sets.String
	labelSelector string
	fieldSelector string

	cascade      bool
	serverDryRun bool
	dryRun       bool
	gracePeriod  int

	toPrinter func(string) (printers.ResourcePrinter, error)

	out io.Writer
}

func (p *pruner) prune(namespace string, mapping *meta.RESTMapping, includeUninitialized bool) error {
	objList, err := p.dynamicClient.Resource(mapping.Resource).
		Namespace(namespace).
		List(metav1.ListOptions{
			LabelSelector:        p.labelSelector,
			FieldSelector:        p.fieldSelector,
			IncludeUninitialized: includeUninitialized,
		})
	if err != nil {
		return err
	}

	objs, err := meta.ExtractList(objList)
	if err != nil {
		return err
	}

	for _, obj := range objs {
		metadata, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		annots := metadata.GetAnnotations()
		if _, ok := annots[corev1.LastAppliedConfigAnnotation]; !ok {
			// don't prune resources not created with apply
			continue
		}
		uid := metadata.GetUID()
		if p.visitedUids.Has(string(uid)) {
			continue
		}
		name := metadata.GetName()
		if !p.dryRun {
			if err := p.delete(namespace, name, mapping); err != nil {
				return err
			}
		}

		printer, err := p.toPrinter("pruned")
		if err != nil {
			return err
		}
		printer.PrintObj(obj, p.out)
	}
	return nil
}

func (p *pruner) delete(namespace, name string, mapping *meta.RESTMapping) error {
	return runDelete(namespace, name, mapping, p.dynamicClient, p.cascade, p.gracePeriod, p.serverDryRun)
}

func runDelete(namespace, name string, mapping *meta.RESTMapping, c dynamic.Interface, cascade bool, gracePeriod int, serverDryRun bool) error {
	options := &metav1.DeleteOptions{}
	if gracePeriod >= 0 {
		options = metav1.NewDeleteOptions(int64(gracePeriod))
	}
	if serverDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	policy := metav1.DeletePropagationForeground
	if !cascade {
		policy = metav1.DeletePropagationOrphan
	}
	options.PropagationPolicy = &policy
	return c.Resource(mapping.Resource).Namespace(namespace).Delete(name, options)
}

func (p *patcher) delete(namespace, name string) error {
	return runDelete(namespace, name, p.mapping, p.dynamicClient, p.cascade, p.gracePeriod, p.serverDryRun)
}

type patcher struct {
	mapping       *meta.RESTMapping
	helper        *resource.Helper
	dynamicClient dynamic.Interface

	overwrite bool
	backOff   clockwork.Clock

	force        bool
	cascade      bool
	timeout      time.Duration
	gracePeriod  int
	serverDryRun bool

	openapiSchema openapi.Resources
}

func (p *patcher) patchSimple(obj runtime.Object, modified []byte, source, namespace, name string, errOut io.Writer) ([]byte, runtime.Object, error) {
	// Serialize the current configuration of the object from the server.
	current, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf("serializing current configuration from:\n%v\nfor:", obj), source, err)
	}

	// Retrieve the original configuration of the object from the annotation.
	original, err := kubectl.GetOriginalConfiguration(obj)
	if err != nil {
		return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf("retrieving original configuration from:\n%v\nfor:", obj), source, err)
	}

	var patchType types.PatchType
	var patch []byte
	var lookupPatchMeta strategicpatch.LookupPatchMeta
	var schema oapi.Schema
	createPatchErrFormat := "creating patch with:\noriginal:\n%s\nmodified:\n%s\ncurrent:\n%s\nfor:"

	// Create the versioned struct from the type defined in the restmapping
	// (which is the API version we'll be submitting the patch to)
	versionedObject, err := scheme.Scheme.New(p.mapping.GroupVersionKind)
	switch {
	case runtime.IsNotRegisteredError(err):
		// fall back to generic JSON merge patch
		patchType = types.MergePatchType
		preconditions := []mergepatch.PreconditionFunc{mergepatch.RequireKeyUnchanged("apiVersion"),
			mergepatch.RequireKeyUnchanged("kind"), mergepatch.RequireMetadataKeyUnchanged("name")}
		patch, err = jsonmergepatch.CreateThreeWayJSONMergePatch(original, modified, current, preconditions...)
		if err != nil {
			if mergepatch.IsPreconditionFailed(err) {
				return nil, nil, fmt.Errorf("%s", "At least one of apiVersion, kind and name was changed")
			}
			return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf(createPatchErrFormat, original, modified, current), source, err)
		}
	case err != nil:
		return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf("getting instance of versioned object for %v:", p.mapping.GroupVersionKind), source, err)
	case err == nil:
		// Compute a three way strategic merge patch to send to server.
		patchType = types.StrategicMergePatchType

		// Try to use openapi first if the openapi spec is available and can successfully calculate the patch.
		// Otherwise, fall back to baked-in types.
		if p.openapiSchema != nil {
			if schema = p.openapiSchema.LookupResource(p.mapping.GroupVersionKind); schema != nil {
				lookupPatchMeta = strategicpatch.PatchMetaFromOpenAPI{Schema: schema}
				if openapiPatch, err := strategicpatch.CreateThreeWayMergePatch(original, modified, current, lookupPatchMeta, p.overwrite); err != nil {
					fmt.Fprintf(errOut, "warning: error calculating patch from openapi spec: %v\n", err)
				} else {
					patchType = types.StrategicMergePatchType
					patch = openapiPatch
				}
			}
		}

		if patch == nil {
			lookupPatchMeta, err = strategicpatch.NewPatchMetaFromStruct(versionedObject)
			if err != nil {
				return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf(createPatchErrFormat, original, modified, current), source, err)
			}
			patch, err = strategicpatch.CreateThreeWayMergePatch(original, modified, current, lookupPatchMeta, p.overwrite)
			if err != nil {
				return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf(createPatchErrFormat, original, modified, current), source, err)
			}
		}
	}

	if string(patch) == "{}" {
		return patch, obj, nil
	}

	options := metav1.UpdateOptions{}
	if p.serverDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}

	patchedObj, err := p.helper.Patch(namespace, name, patchType, patch, &options)
	return patch, patchedObj, err
}

func (p *patcher) patch(current runtime.Object, modified []byte, source, namespace, name string, errOut io.Writer) ([]byte, runtime.Object, error) {
	var getErr error
	patchBytes, patchObject, err := p.patchSimple(current, modified, source, namespace, name, errOut)
	for i := 1; i <= maxPatchRetry && errors.IsConflict(err); i++ {
		if i > triesBeforeBackOff {
			p.backOff.Sleep(backOffPeriod)
		}
		current, getErr = p.helper.Get(namespace, name, false)
		if getErr != nil {
			return nil, nil, getErr
		}
		patchBytes, patchObject, err = p.patchSimple(current, modified, source, namespace, name, errOut)
	}
	if err != nil && (errors.IsConflict(err) || errors.IsInvalid(err)) && p.force {
		patchBytes, patchObject, err = p.deleteAndCreate(current, modified, namespace, name)
	}
	return patchBytes, patchObject, err
}

func (p *patcher) deleteAndCreate(original runtime.Object, modified []byte, namespace, name string) ([]byte, runtime.Object, error) {
	if err := p.delete(namespace, name); err != nil {
		return modified, nil, err
	}
	// TODO: use wait
	if err := wait.PollImmediate(1*time.Second, p.timeout, func() (bool, error) {
		if _, err := p.helper.Get(namespace, name, false); !errors.IsNotFound(err) {
			return false, err
		}
		return true, nil
	}); err != nil {
		return modified, nil, err
	}
	versionedObject, _, err := unstructured.UnstructuredJSONScheme.Decode(modified, nil, nil)
	if err != nil {
		return modified, nil, err
	}
	options := metav1.CreateOptions{}
	if p.serverDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	createdObject, err := p.helper.Create(namespace, true, versionedObject, &options)
	if err != nil {
		// restore the original object if we fail to create the new one
		// but still propagate and advertise error to user
		recreated, recreateErr := p.helper.Create(namespace, true, original, &options)
		if recreateErr != nil {
			err = fmt.Errorf("An error occurred force-replacing the existing object with the newly provided one:\n\n%v.\n\nAdditionally, an error occurred attempting to restore the original object:\n\n%v\n", err, recreateErr)
		} else {
			createdObject = recreated
		}
	}
	return modified, createdObject, err
}
