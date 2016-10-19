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
	"time"

	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
)

type ApplyOptions struct {
	FilenameOptions resource.FilenameOptions
	Selector        string
	Prune           bool
	Cascade         bool
	GracePeriod     int
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
		cat pod.json | kubectl apply -f -`)
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
	cmd.Flags().BoolVar(&options.Prune, "prune", false, "Automatically delete resource objects that do not appear in the configs")
	cmd.Flags().BoolVar(&options.Cascade, "cascade", true, "Only relevant during a prune. If true, cascade the deletion of the resources managed by pruned resources (e.g. Pods created by a ReplicationController).")
	cmd.Flags().IntVar(&options.GracePeriod, "grace-period", -1, "Period of time in seconds given to pruned resources to terminate gracefully. Ignored if negative.")
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().Bool("all", false, "[-all] to select all the specified resources.")
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
	visitedNamespacedRESTMappings := map[unversioned.GroupVersionKind]*meta.RESTMapping{}
	visitedNonNamespacedRESTMappings := map[unversioned.GroupVersionKind]*meta.RESTMapping{}

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		// In this method, info.Object contains the object retrieved from the server
		// and info.VersionedObject contains the object decoded from the input source.
		if err != nil {
			return err
		}

		if info.Namespaced() {
			visitedNamespaces.Insert(info.Namespace)
			visitedNamespacedRESTMappings[info.Mapping.GroupVersionKind] = info.Mapping
		} else {
			visitedNonNamespacedRESTMappings[info.Mapping.GroupVersionKind] = info.Mapping
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
			patcher := NewPatcher(encoder, decoder, info.Mapping, helper, overwrite)

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
		gracePeriod: options.GracePeriod,

		out: out,
	}
	for n := range visitedNamespaces {
		for _, m := range visitedNamespacedRESTMappings {
			if err := p.prune(n, m, shortOutput); err != nil {
				return fmt.Errorf("error pruning objects: %v", err)
			}
		}
	}
	for _, m := range visitedNonNamespacedRESTMappings {
		if err := p.prune(api.NamespaceNone, m, shortOutput); err != nil {
			return fmt.Errorf("error pruning objects: %v", err)
		}
	}

	return nil
}

type pruner struct {
	mapper        meta.RESTMapper
	clientFunc    resource.ClientMapperFunc
	clientsetFunc func() (*internalclientset.Clientset, error)

	visitedUids sets.String
	selector    labels.Selector

	cascade     bool
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
		if err := p.delete(namespace, name, mapping, c); err != nil {
			return err
		}
		cmdutil.PrintSuccess(p.mapper, shortOutput, p.out, mapping.Resource, name, false, "pruned")
	}
	return nil
}

func (p *pruner) delete(namespace, name string, mapping *meta.RESTMapping, c resource.RESTClient) error {
	if !p.cascade {
		return resource.NewHelper(c, mapping).Delete(namespace, name)
	}
	cs, err := p.clientsetFunc()
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
	if err := r.Stop(namespace, name, 2*time.Minute, api.NewDeleteOptions(int64(p.gracePeriod))); err != nil {
		return err
	}
	return nil
}

type patcher struct {
	encoder runtime.Encoder
	decoder runtime.Decoder

	mapping *meta.RESTMapping
	helper  *resource.Helper

	overwrite bool
	backOff   clockwork.Clock
}

func NewPatcher(encoder runtime.Encoder, decoder runtime.Decoder, mapping *meta.RESTMapping, helper *resource.Helper, overwrite bool) *patcher {
	return &patcher{
		encoder:   encoder,
		decoder:   decoder,
		mapping:   mapping,
		helper:    helper,
		overwrite: overwrite,
		backOff:   clockwork.NewRealClock(),
	}
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

	return patchBytes, err
}
