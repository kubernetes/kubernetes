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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/jsonmergepatch"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

type ApplyOptions struct {
	FilenameOptions resource.FilenameOptions
	Selector        string
	Force           bool
	Prune           bool
	Cascade         bool
	GracePeriod     int
	PruneResources  []pruneResource
	Timeout         time.Duration
	cmdBaseName     string
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

func NewCmdApply(baseName string, f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	var options ApplyOptions

	// Store baseName for use in printing warnings / messages involving the base command name.
	// This is useful for downstream command that wrap this one.
	options.cmdBaseName = baseName

	cmd := &cobra.Command{
		Use:     "apply -f FILENAME",
		Short:   i18n.T("Apply a configuration to a resource by filename or stdin"),
		Long:    applyLong,
		Example: applyExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(validateArgs(cmd, args))
			cmdutil.CheckErr(validatePruneAll(options.Prune, cmdutil.GetFlagBool(cmd, "all"), options.Selector))
			cmdutil.CheckErr(RunApply(f, cmd, out, errOut, &options))
		},
	}

	usage := "that contains the configuration to apply"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.MarkFlagRequired("filename")
	cmd.Flags().Bool("overwrite", true, "Automatically resolve conflicts between the modified and live configuration by using values from the modified configuration")
	cmd.Flags().BoolVar(&options.Prune, "prune", false, "Automatically delete resource objects, including the uninitialized ones, that do not appear in the configs and are created by either apply or create --save-config. Should be used with either -l or --all.")
	cmd.Flags().BoolVar(&options.Cascade, "cascade", true, "Only relevant during a prune or a force apply. If true, cascade the deletion of the resources managed by pruned or deleted resources (e.g. Pods created by a ReplicationController).")
	cmd.Flags().IntVar(&options.GracePeriod, "grace-period", -1, "Only relevant during a prune or a force apply. Period of time in seconds given to pruned or deleted resources to terminate gracefully. Ignored if negative.")
	cmd.Flags().BoolVar(&options.Force, "force", false, fmt.Sprintf("Delete and re-create the specified resource, when PATCH encounters conflict and has retried for %d times.", maxPatchRetry))
	cmd.Flags().DurationVar(&options.Timeout, "timeout", 0, "Only relevant during a force apply. The length of time to wait before giving up on a delete of the old resource, zero means determine a timeout from the size of the object. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().Bool("all", false, "Select all resources in the namespace of the specified resource types.")
	cmd.Flags().StringArray("prune-whitelist", []string{}, "Overwrite the default whitelist with <group/version/kind> for --prune")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)

	// apply subcommands
	cmd.AddCommand(NewCmdApplyViewLastApplied(f, out, errOut))
	cmd.AddCommand(NewCmdApplySetLastApplied(f, out, errOut))
	cmd.AddCommand(NewCmdApplyEditLastApplied(f, out, errOut))

	return cmd
}

func validateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}

	return nil
}

func validatePruneAll(prune, all bool, selector string) error {
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

func RunApply(f cmdutil.Factory, cmd *cobra.Command, out, errOut io.Writer, options *ApplyOptions) error {
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"))
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

	if options.Prune {
		options.PruneResources, err = parsePruneResources(mapper, cmdutil.GetFlagStringArray(cmd, "prune-whitelist"))
		if err != nil {
			return err
		}
	}

	// include the uninitialized objects by default if --prune is true
	// unless explicitly set --include-uninitialized=false
	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, options.Prune)
	r := f.NewBuilder().
		Unstructured(f.UnstructuredClientForMapping, mapper, typer).
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		SelectorParam(options.Selector).
		IncludeUninitialized(includeUninitialized).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	dryRun := cmdutil.GetFlagBool(cmd, "dry-run")
	output := cmdutil.GetFlagString(cmd, "output")
	shortOutput := output == "name"

	encoder := f.JSONEncoder()
	decoder := f.Decoder(false)

	visitedUids := sets.NewString()
	visitedNamespaces := sets.NewString()

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		// In this method, info.Object contains the object retrieved from the server
		// and info.VersionedObject contains the object decoded from the input source.
		if err != nil {
			return err
		}

		if info.Namespaced() {
			visitedNamespaces.Insert(info.Namespace)
		}

		// Add change-cause annotation to resource info if it should be recorded
		if cmdutil.ShouldRecord(cmd, info) {
			recordInObj := info.VersionedObject
			if info.VersionedObject == nil {
				recordInObj = info.Object
			}
			if err := cmdutil.RecordChangeCause(recordInObj, f.Command(cmd, false)); err != nil {
				glog.V(4).Infof("error recording current command: %v", err)
			}
		}

		// Get the modified configuration of the object. Embed the result
		// as an annotation in the modified configuration, so that it will appear
		// in the patch sent to the server.
		modified, err := kubectl.GetModifiedConfiguration(info, true, encoder)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving modified configuration from:\n%v\nfor:", info), info.Source, err)
		}

		if err := info.Get(); err != nil {
			if !errors.IsNotFound(err) {
				return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
			}
			// Create the resource if it doesn't exist
			// First, update the annotation used by kubectl apply
			if err := kubectl.CreateApplyAnnotation(info, encoder); err != nil {
				return cmdutil.AddSourceToErr("creating", info.Source, err)
			}

			if !dryRun {
				// Then create the resource and skip the three-way merge
				if err := createAndRefresh(info); err != nil {
					return cmdutil.AddSourceToErr("creating", info.Source, err)
				}
				if uid, err := info.Mapping.UID(info.Object); err != nil {
					return err
				} else {
					visitedUids.Insert(string(uid))
				}
			}

			count++
			if len(output) > 0 && !shortOutput {
				return cmdutil.PrintResourceInfoForCommand(cmd, info, f, out)
			}
			cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, dryRun, "created")
			return nil
		}

		if !dryRun {
			annotationMap, err := info.Mapping.MetadataAccessor.Annotations(info.Object)
			if err != nil {
				return err
			}
			if _, ok := annotationMap[api.LastAppliedConfigAnnotation]; !ok {
				fmt.Fprintf(errOut, warningNoLastAppliedConfigAnnotation, options.cmdBaseName)
			}
			overwrite := cmdutil.GetFlagBool(cmd, "overwrite")
			helper := resource.NewHelper(info.Client, info.Mapping)
			patcher := &patcher{
				encoder:       encoder,
				decoder:       decoder,
				mapping:       info.Mapping,
				helper:        helper,
				clientFunc:    f.UnstructuredClientForMapping,
				clientsetFunc: f.ClientSet,
				overwrite:     overwrite,
				backOff:       clockwork.NewRealClock(),
				force:         options.Force,
				cascade:       options.Cascade,
				timeout:       options.Timeout,
				gracePeriod:   options.GracePeriod,
			}

			patchBytes, patchedObject, err := patcher.patch(info.Object, modified, info.Source, info.Namespace, info.Name)
			if err != nil {
				return cmdutil.AddSourceToErr(fmt.Sprintf("applying patch:\n%s\nto:\n%v\nfor:", patchBytes, info), info.Source, err)
			}

			info.Refresh(patchedObject, true)

			if uid, err := info.Mapping.UID(info.Object); err != nil {
				return err
			} else {
				visitedUids.Insert(string(uid))
			}
		}
		count++
		if len(output) > 0 && !shortOutput {
			return cmdutil.PrintResourceInfoForCommand(cmd, info, f, out)
		}
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, dryRun, "configured")
		return nil
	})

	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to apply")
	}

	if !options.Prune {
		return nil
	}

	p := pruner{
		mapper:        mapper,
		clientFunc:    f.UnstructuredClientForMapping,
		clientsetFunc: f.ClientSet,

		selector:    options.Selector,
		visitedUids: visitedUids,

		cascade:     options.Cascade,
		dryRun:      dryRun,
		gracePeriod: options.GracePeriod,

		out: out,
	}

	namespacedRESTMappings, nonNamespacedRESTMappings, err := getRESTMappings(mapper, &(options.PruneResources))
	if err != nil {
		return fmt.Errorf("error retrieving RESTMappings to prune: %v", err)
	}

	for n := range visitedNamespaces {
		for _, m := range namespacedRESTMappings {
			if err := p.prune(n, m, shortOutput, includeUninitialized); err != nil {
				return fmt.Errorf("error pruning namespaced object %v: %v", m.GroupVersionKind, err)
			}
		}
	}
	for _, m := range nonNamespacedRESTMappings {
		if err := p.prune(metav1.NamespaceNone, m, shortOutput, includeUninitialized); err != nil {
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
	clientFunc    resource.ClientMapperFunc
	clientsetFunc func() (internalclientset.Interface, error)

	visitedUids sets.String
	selector    string

	cascade     bool
	dryRun      bool
	gracePeriod int

	out io.Writer
}

func (p *pruner) prune(namespace string, mapping *meta.RESTMapping, shortOutput, includeUninitialized bool) error {
	c, err := p.clientFunc(mapping)
	if err != nil {
		return err
	}

	objList, err := resource.NewHelper(c, mapping).List(namespace, mapping.GroupVersionKind.Version, p.selector, false, includeUninitialized)
	if err != nil {
		return err
	}
	objs, err := meta.ExtractList(objList)
	if err != nil {
		return err
	}

	for _, obj := range objs {
		annots, err := mapping.MetadataAccessor.Annotations(obj)
		if err != nil {
			return err
		}
		if _, ok := annots[api.LastAppliedConfigAnnotation]; !ok {
			// don't prune resources not created with apply
			continue
		}
		uid, err := mapping.UID(obj)
		if err != nil {
			return err
		}
		if p.visitedUids.Has(string(uid)) {
			continue
		}

		name, err := mapping.Name(obj)
		if err != nil {
			return err
		}
		if !p.dryRun {
			if err := p.delete(namespace, name, mapping); err != nil {
				return err
			}
		}
		cmdutil.PrintSuccess(p.mapper, shortOutput, p.out, mapping.Resource, name, p.dryRun, "pruned")
	}
	return nil
}

func (p *pruner) delete(namespace, name string, mapping *meta.RESTMapping) error {
	c, err := p.clientFunc(mapping)
	if err != nil {
		return err
	}

	return runDelete(namespace, name, mapping, c, nil, p.cascade, p.gracePeriod, p.clientsetFunc)
}

func runDelete(namespace, name string, mapping *meta.RESTMapping, c resource.RESTClient, helper *resource.Helper, cascade bool, gracePeriod int, clientsetFunc func() (internalclientset.Interface, error)) error {
	if !cascade {
		if helper == nil {
			helper = resource.NewHelper(c, mapping)
		}
		return helper.Delete(namespace, name)
	}
	cs, err := clientsetFunc()
	if err != nil {
		return err
	}
	r, err := kubectl.ReaperFor(mapping.GroupVersionKind.GroupKind(), cs)
	if err != nil {
		if _, ok := err.(*kubectl.NoSuchReaperError); !ok {
			return err
		}
		return resource.NewHelper(c, mapping).Delete(namespace, name)
	}
	var options *metav1.DeleteOptions
	if gracePeriod >= 0 {
		options = metav1.NewDeleteOptions(int64(gracePeriod))
	}
	if err := r.Stop(namespace, name, 2*time.Minute, options); err != nil {
		return err
	}
	return nil
}

func (p *patcher) delete(namespace, name string) error {
	c, err := p.clientFunc(p.mapping)
	if err != nil {
		return err
	}
	return runDelete(namespace, name, p.mapping, c, p.helper, p.cascade, p.gracePeriod, p.clientsetFunc)
}

type patcher struct {
	encoder runtime.Encoder
	decoder runtime.Decoder

	mapping       *meta.RESTMapping
	helper        *resource.Helper
	clientFunc    resource.ClientMapperFunc
	clientsetFunc func() (internalclientset.Interface, error)

	overwrite bool
	backOff   clockwork.Clock

	force       bool
	cascade     bool
	timeout     time.Duration
	gracePeriod int
}

func (p *patcher) patchSimple(obj runtime.Object, modified []byte, source, namespace, name string) ([]byte, runtime.Object, error) {
	// Serialize the current configuration of the object from the server.
	current, err := runtime.Encode(p.encoder, obj)
	if err != nil {
		return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf("serializing current configuration from:\n%v\nfor:", obj), source, err)
	}

	// Retrieve the original configuration of the object from the annotation.
	original, err := kubectl.GetOriginalConfiguration(p.mapping, obj)
	if err != nil {
		return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf("retrieving original configuration from:\n%v\nfor:", obj), source, err)
	}

	// Create the versioned struct from the type defined in the restmapping
	// (which is the API version we'll be submitting the patch to)
	versionedObject, err := legacyscheme.Scheme.New(p.mapping.GroupVersionKind)
	var patchType types.PatchType
	var patch []byte
	createPatchErrFormat := "creating patch with:\noriginal:\n%s\nmodified:\n%s\ncurrent:\n%s\nfor:"
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
		patch, err = strategicpatch.CreateThreeWayMergePatch(original, modified, current, versionedObject, p.overwrite)
		if err != nil {
			return nil, nil, cmdutil.AddSourceToErr(fmt.Sprintf(createPatchErrFormat, original, modified, current), source, err)
		}
	}

	patchedObj, err := p.helper.Patch(namespace, name, patchType, patch)
	return patch, patchedObj, err
}

func (p *patcher) patch(current runtime.Object, modified []byte, source, namespace, name string) ([]byte, runtime.Object, error) {
	var getErr error
	patchBytes, patchObject, err := p.patchSimple(current, modified, source, namespace, name)
	for i := 1; i <= maxPatchRetry && errors.IsConflict(err); i++ {
		if i > triesBeforeBackOff {
			p.backOff.Sleep(backOffPeriod)
		}
		current, getErr = p.helper.Get(namespace, name, false)
		if getErr != nil {
			return nil, nil, getErr
		}
		patchBytes, patchObject, err = p.patchSimple(current, modified, source, namespace, name)
	}
	if err != nil && p.force {
		patchBytes, patchObject, err = p.deleteAndCreate(modified, namespace, name)
	}
	return patchBytes, patchObject, err
}

func (p *patcher) deleteAndCreate(modified []byte, namespace, name string) ([]byte, runtime.Object, error) {
	err := p.delete(namespace, name)
	if err != nil {
		return modified, nil, err
	}
	err = wait.PollImmediate(kubectl.Interval, p.timeout, func() (bool, error) {
		if _, err := p.helper.Get(namespace, name, false); !errors.IsNotFound(err) {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		return modified, nil, err
	}
	versionedObject, _, err := p.decoder.Decode(modified, nil, nil)
	if err != nil {
		return modified, nil, err
	}
	createdObject, err := p.helper.Create(namespace, true, versionedObject)
	return modified, createdObject, err
}
