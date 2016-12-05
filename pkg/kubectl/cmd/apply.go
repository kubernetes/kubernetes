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

	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
	"k8s.io/kubernetes/pkg/util/wait"
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
	apply_long = templates.LongDesc(`
		Apply a configuration to a resource by filename or stdin.
		This resource will be created if it doesn't exist yet.
		To use 'apply', always create the resource initially with either 'apply' or 'create --save-config'.

		JSON and YAML formats are accepted.

		Alpha Disclaimer: the --prune functionality is not yet complete. Do not use unless you are aware of what the current state is. See https://issues.k8s.io/34274.`)

	apply_example = templates.Examples(`
		# Apply the configuration in pod.json to a pod.
		kubectl apply -f ./pod.json

		# Apply the JSON passed into stdin to a pod.
		cat pod.json | kubectl apply -f -

		# Note: --prune is still in Alpha
		# Apply the configuration in manifest.yaml that matches label app=nginx and delete all the other resources that are not in the file and match label app=nginx.
		kubectl apply --prune -f manifest.yaml -l app=nginx

		# Apply the configuration in manifest.yaml and delete all the other configmaps that are not in the file.
		kubectl apply --prune -f manifest.yaml --all --prune-whitelist=core/v1/ConfigMap`)
)

func NewCmdApply(f cmdutil.Factory, out io.Writer) *cobra.Command {
	var options ApplyOptions

	cmd := &cobra.Command{
		Use:     "apply -f FILENAME",
		Short:   "Apply a configuration to a resource by filename or stdin",
		Long:    apply_long,
		Example: apply_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(validateArgs(cmd, args))
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			cmdutil.CheckErr(validatePruneAll(options.Prune, cmdutil.GetFlagBool(cmd, "all"), options.Selector))
			cmdutil.CheckErr(RunApply(f, cmd, out, &options))
		},
	}

	usage := "that contains the configuration to apply"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.MarkFlagRequired("filename")
	cmd.Flags().Bool("overwrite", true, "Automatically resolve conflicts between the modified and live configuration by using values from the modified configuration")
	cmd.Flags().BoolVar(&options.Prune, "prune", false, "Automatically delete resource objects that do not appear in the configs and are created by either apply or create --save-config. Should be used with either -l or --all.")
	cmd.Flags().BoolVar(&options.Cascade, "cascade", true, "Only relevant during a prune or a force apply. If true, cascade the deletion of the resources managed by pruned or deleted resources (e.g. Pods created by a ReplicationController).")
	cmd.Flags().IntVar(&options.GracePeriod, "grace-period", -1, "Only relevant during a prune or a force apply. Period of time in seconds given to pruned or deleted resources to terminate gracefully. Ignored if negative.")
	cmd.Flags().BoolVar(&options.Force, "force", false, fmt.Sprintf("Delete and re-create the specified resource, when PATCH encounters conflict and has retried for %d times.", maxPatchRetry))
	cmd.Flags().DurationVar(&options.Timeout, "timeout", 0, "Only relevant during a force apply. The length of time to wait before giving up on a delete of the old resource, zero means determine a timeout from the size of the object. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().Bool("all", false, "[-all] to select all the specified resources.")
	cmd.Flags().StringArray("prune-whitelist", []string{}, "Overwrite the default whitelist with <group/version/kind> for --prune")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func validateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageError(cmd, "Unexpected args: %v", args)
	}

	return nil
}

func validatePruneAll(prune, all bool, selector string) error {
	if prune && !all && selector == "" {
		return fmt.Errorf("all resources selected for prune without explicitly passing --all. To prune all resources, pass the --all flag. If you did not mean to prune all resources, specify a label selector.")
	}
	return nil
}

func parsePruneResources(gvks []string) ([]pruneResource, error) {
	pruneResources := []pruneResource{}
	for _, groupVersionKind := range gvks {
		gvk := strings.Split(groupVersionKind, "/")
		if len(gvk) != 3 {
			return nil, fmt.Errorf("invalid GroupVersionKind format: %v, please follow <group/version/kind>", groupVersionKind)
		}

		namespaced := true
		if gvk[2] == "Namespace" ||
			gvk[2] == "Node" ||
			gvk[2] == "PersistentVolume" {
			namespaced = false
		}
		if gvk[0] == "core" {
			gvk[0] = ""
		}
		pruneResources = append(pruneResources, pruneResource{gvk[0], gvk[1], gvk[2], namespaced})
	}
	return pruneResources, nil
}

func RunApply(f cmdutil.Factory, cmd *cobra.Command, out io.Writer, options *ApplyOptions) error {
	shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if options.Prune {
		options.PruneResources, err = parsePruneResources(cmdutil.GetFlagStringArray(cmd, "prune-whitelist"))
		if err != nil {
			return err
		}
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		SelectorParam(options.Selector).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	dryRun := cmdutil.GetFlagBool(cmd, "dry-run")

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

			if cmdutil.ShouldRecord(cmd, info) {
				if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
					return cmdutil.AddSourceToErr("creating", info.Source, err)
				}
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
			cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, dryRun, "created")
			return nil
		}

		if !dryRun {
			overwrite := cmdutil.GetFlagBool(cmd, "overwrite")
			helper := resource.NewHelper(info.Client, info.Mapping)
			patcher := &patcher{
				encoder:       encoder,
				decoder:       decoder,
				mapping:       info.Mapping,
				helper:        helper,
				clientsetFunc: f.ClientSet,
				overwrite:     overwrite,
				backOff:       clockwork.NewRealClock(),
				force:         options.Force,
				cascade:       options.Cascade,
				timeout:       options.Timeout,
				gracePeriod:   options.GracePeriod,
			}

			patchBytes, err := patcher.patch(info.Object, modified, info.Source, info.Namespace, info.Name)
			if err != nil {
				return cmdutil.AddSourceToErr(fmt.Sprintf("applying patch:\n%s\nto:\n%v\nfor:", patchBytes, info), info.Source, err)
			}

			if cmdutil.ShouldRecord(cmd, info) {
				patch, err := cmdutil.ChangeResourcePatch(info, f.Command())
				if err != nil {
					return err
				}
				_, err = helper.Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch)
				if err != nil {
					return cmdutil.AddSourceToErr(fmt.Sprintf("applying patch:\n%s\nto:\n%v\nfor:", patch, info), info.Source, err)
				}
			}

			if uid, err := info.Mapping.UID(info.Object); err != nil {
				return err
			} else {
				visitedUids.Insert(string(uid))
			}
		}
		count++
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

	selector, err := labels.Parse(options.Selector)
	if err != nil {
		return err
	}
	p := pruner{
		mapper:        mapper,
		clientFunc:    f.ClientForMapping,
		clientsetFunc: f.ClientSet,

		selector:    selector,
		visitedUids: visitedUids,

		cascade:     options.Cascade,
		dryRun:      dryRun,
		gracePeriod: options.GracePeriod,

		out: out,
	}

	namespacedRESTMappings, nonNamespacedRESTMappings, err := getRESTMappings(&(options.PruneResources))
	if err != nil {
		return fmt.Errorf("error retrieving RESTMappings to prune: %v", err)
	}

	for n := range visitedNamespaces {
		for _, m := range namespacedRESTMappings {
			if err := p.prune(n, m, shortOutput); err != nil {
				return fmt.Errorf("error pruning namespaced object %v: %v", m.GroupVersionKind, err)
			}
		}
	}
	for _, m := range nonNamespacedRESTMappings {
		if err := p.prune(api.NamespaceNone, m, shortOutput); err != nil {
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

func getRESTMappings(pruneResources *[]pruneResource) (namespaced, nonNamespaced []*meta.RESTMapping, err error) {
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
			{"extensions", "v1beta1", "HorizontalPodAutoscaler", true},
			{"extensions", "v1beta1", "Ingress", true},
			{"extensions", "v1beta1", "ReplicaSet", true},
			{"apps", "v1beta1", "StatefulSet", true},
		}
	}
	registeredMapper := registered.RESTMapper()
	for _, resource := range *pruneResources {
		addedMapping, err := registeredMapper.RESTMapping(schema.GroupKind{Group: resource.group, Kind: resource.kind}, resource.version)
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
	clientsetFunc func() (*internalclientset.Clientset, error)

	visitedUids sets.String
	selector    labels.Selector

	cascade     bool
	dryRun      bool
	gracePeriod int

	out io.Writer
}

func (p *pruner) prune(namespace string, mapping *meta.RESTMapping, shortOutput bool) error {
	c, err := p.clientFunc(mapping)
	if err != nil {
		return err
	}

	objList, err := resource.NewHelper(c, mapping).List(namespace, mapping.GroupVersionKind.Version, p.selector, false)
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
		if _, ok := annots[annotations.LastAppliedConfigAnnotation]; !ok {
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
			if err := p.delete(namespace, name, mapping, c); err != nil {
				return err
			}
		}
		cmdutil.PrintSuccess(p.mapper, shortOutput, p.out, mapping.Resource, name, p.dryRun, "pruned")
	}
	return nil
}

func (p *pruner) delete(namespace, name string, mapping *meta.RESTMapping, c resource.RESTClient) error {
	return runDelete(namespace, name, mapping, c, nil, p.cascade, p.gracePeriod, p.clientsetFunc)
}

func runDelete(namespace, name string, mapping *meta.RESTMapping, c resource.RESTClient, helper *resource.Helper, cascade bool, gracePeriod int, clientsetFunc func() (*internalclientset.Clientset, error)) error {
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
	var options *api.DeleteOptions
	if gracePeriod >= 0 {
		options = api.NewDeleteOptions(int64(gracePeriod))
	}
	if err := r.Stop(namespace, name, 2*time.Minute, options); err != nil {
		return err
	}
	return nil
}

func (p *patcher) delete(namespace, name string) error {
	return runDelete(namespace, name, p.mapping, nil, p.helper, p.cascade, p.gracePeriod, p.clientsetFunc)
}

type patcher struct {
	encoder runtime.Encoder
	decoder runtime.Decoder

	mapping       *meta.RESTMapping
	helper        *resource.Helper
	clientsetFunc func() (*internalclientset.Clientset, error)

	overwrite bool
	backOff   clockwork.Clock

	force       bool
	cascade     bool
	timeout     time.Duration
	gracePeriod int
}

func (p *patcher) patchSimple(obj runtime.Object, modified []byte, source, namespace, name string) ([]byte, error) {
	// Serialize the current configuration of the object from the server.
	current, err := runtime.Encode(p.encoder, obj)
	if err != nil {
		return nil, cmdutil.AddSourceToErr(fmt.Sprintf("serializing current configuration from:\n%v\nfor:", obj), source, err)
	}

	// Retrieve the original configuration of the object from the annotation.
	original, err := kubectl.GetOriginalConfiguration(p.mapping, obj)
	if err != nil {
		return nil, cmdutil.AddSourceToErr(fmt.Sprintf("retrieving original configuration from:\n%v\nfor:", obj), source, err)
	}

	// Create the versioned struct from the original from the server for
	// strategic patch.
	// TODO: Move all structs in apply to use raw data. Can be done once
	// builder has a RawResult method which delivers raw data instead of
	// internal objects.
	versionedObject, _, err := p.decoder.Decode(current, nil, nil)
	if err != nil {
		return nil, cmdutil.AddSourceToErr(fmt.Sprintf("converting encoded server-side object back to versioned struct:\n%v\nfor:", obj), source, err)
	}

	// Compute a three way strategic merge patch to send to server.
	patch, err := strategicpatch.CreateThreeWayMergePatch(original, modified, current, versionedObject, p.overwrite)
	if err != nil {
		format := "creating patch with:\noriginal:\n%s\nmodified:\n%s\ncurrent:\n%s\nfor:"
		return nil, cmdutil.AddSourceToErr(fmt.Sprintf(format, original, modified, current), source, err)
	}

	_, err = p.helper.Patch(namespace, name, api.StrategicMergePatchType, patch)
	return patch, err
}

func (p *patcher) patch(current runtime.Object, modified []byte, source, namespace, name string) ([]byte, error) {
	var getErr error
	patchBytes, err := p.patchSimple(current, modified, source, namespace, name)
	for i := 1; i <= maxPatchRetry && errors.IsConflict(err); i++ {
		if i > triesBeforeBackOff {
			p.backOff.Sleep(backOffPeriod)
		}
		current, getErr = p.helper.Get(namespace, name, false)
		if getErr != nil {
			return nil, getErr
		}
		patchBytes, err = p.patchSimple(current, modified, source, namespace, name)
	}
	if err != nil && p.force {
		patchBytes, err = p.deleteAndCreate(modified, namespace, name)
	}
	return patchBytes, err
}

func (p *patcher) deleteAndCreate(modified []byte, namespace, name string) ([]byte, error) {
	err := p.delete(namespace, name)
	if err != nil {
		return modified, err
	}
	err = wait.PollImmediate(kubectl.Interval, p.timeout, func() (bool, error) {
		if _, err := p.helper.Get(namespace, name, false); !errors.IsNotFound(err) {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		return modified, err
	}
	versionedObject, _, err := p.decoder.Decode(modified, nil, nil)
	if err != nil {
		return modified, err
	}
	_, err = p.helper.Create(namespace, true, versionedObject)
	return modified, err
}
