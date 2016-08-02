/*
Copyright 2016 The Kubernetes Authors.

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

	"encoding/json"

	"github.com/golang/glog"
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
	"k8s.io/kubernetes/pkg/util/validation"
)

// TaintOptions have the data required to perform the taint operation
type TaintOptions struct {
	resources       []string
	taintsToAdd     []api.Taint
	removeTaintKeys []string
	builder         *resource.Builder
	selector        string
	overwrite       bool
	all             bool
	f               *cmdutil.Factory
	out             io.Writer
	cmd             *cobra.Command
}

var (
	taint_long = dedent.Dedent(`
		Update the taints on one or more nodes.

		A taint consists of a key, value, and effect. As an argument here, it is expressed as key=value:effect.
		The key must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
		The value must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
		The effect must be NoSchedule or PreferNoSchedule.
		Currently taint can only apply to node.`)
	taint_example = dedent.Dedent(`
		# Update node 'foo' with a taint with key 'dedicated' and value 'special-user' and effect 'NoSchedule'.
		# If a taint with that key already exists, its value and effect are replaced as specified.
		kubectl taint nodes foo dedicated=special-user:NoSchedule
		# Remove from node 'foo' the taint with key 'dedicated' if one exists.
		kubectl taint nodes foo dedicated-`)
)

func NewCmdTaint(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TaintOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs := []string{}
	p, err := f.Printer(nil, kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
	}

	cmd := &cobra.Command{
		Use:     "taint NODE NAME KEY_1=VAL_1:TAINT_EFFECT_1 ... KEY_N=VAL_N:TAINT_EFFECT_N",
		Short:   "Update the taints on one or more nodes",
		Long:    fmt.Sprintf(taint_long, validation.DNS1123SubdomainMaxLength, validation.LabelValueMaxLength),
		Example: taint_example,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, out, cmd, args); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(args); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			if err := options.RunTaint(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		ValidArgs: validArgs,
	}
	cmdutil.AddValidateFlags(cmd)

	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmd.Flags().StringVarP(&options.selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.overwrite, "overwrite", false, "If true, allow taints to be overwritten, otherwise reject taint updates that overwrite existing taints.")
	cmd.Flags().BoolVar(&options.all, "all", false, "select all nodes in the cluster")
	return cmd
}

func deleteTaintByKey(taints []api.Taint, key string) ([]api.Taint, error) {
	newTaints := []api.Taint{}
	found := false
	for _, taint := range taints {
		if taint.Key == key {
			found = true
			continue
		}
		newTaints = append(newTaints, taint)
	}

	if !found {
		return nil, fmt.Errorf("taint key=\"%s\" not found.", key)
	}
	return newTaints, nil
}

// reorganizeTaints returns the updated set of taints, taking into account old taints that were not updated,
// old taints that were updated, old taints that were deleted, and new taints.
func reorganizeTaints(accessor meta.Object, overwrite bool, taintsToAdd []api.Taint, removeKeys []string) ([]api.Taint, error) {
	newTaints := append([]api.Taint{}, taintsToAdd...)

	var oldTaints []api.Taint
	var err error
	annotations := accessor.GetAnnotations()
	if annotations != nil {
		if oldTaints, err = api.GetTaintsFromNodeAnnotations(annotations); err != nil {
			return nil, err
		}
	}

	// add taints that already existing but not updated to newTaints
	for _, oldTaint := range oldTaints {
		existsInNew := false
		for _, taint := range newTaints {
			if taint.Key == oldTaint.Key {
				existsInNew = true
				break
			}
		}
		if !existsInNew {
			newTaints = append(newTaints, oldTaint)
		}
	}

	allErrs := []error{}
	for _, taintToRemove := range removeKeys {
		newTaints, err = deleteTaintByKey(newTaints, taintToRemove)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}
	return newTaints, utilerrors.NewAggregate(allErrs)
}

func parseTaints(spec []string) ([]api.Taint, []string, error) {
	var taints []api.Taint
	var remove []string
	for _, taintSpec := range spec {
		if strings.Index(taintSpec, "=") != -1 && strings.Index(taintSpec, ":") != -1 {
			parts := strings.Split(taintSpec, "=")
			if len(parts) != 2 || len(parts[1]) == 0 || len(validation.IsQualifiedName(parts[0])) > 0 {
				return nil, nil, fmt.Errorf("invalid taint spec: %v", taintSpec)
			}

			parts2 := strings.Split(parts[1], ":")
			errs := validation.IsValidLabelValue(parts2[0])
			if len(parts2) != 2 || len(errs) != 0 {
				return nil, nil, fmt.Errorf("invalid taint spec: %v, %s", taintSpec, strings.Join(errs, "; "))
			}

			if parts2[1] != string(api.TaintEffectNoSchedule) && parts2[1] != string(api.TaintEffectPreferNoSchedule) {
				return nil, nil, fmt.Errorf("invalid taint spec: %v, unsupported taint effect", taintSpec)
			}

			effect := api.TaintEffect(parts2[1])
			newTaint := api.Taint{
				Key:    parts[0],
				Value:  parts2[0],
				Effect: effect,
			}

			taints = append(taints, newTaint)
		} else if strings.HasSuffix(taintSpec, "-") {
			remove = append(remove, taintSpec[:len(taintSpec)-1])
		} else {
			return nil, nil, fmt.Errorf("unknown taint spec: %v", taintSpec)
		}
	}
	return taints, remove, nil
}

// Complete adapts from the command line args and factory to the data required.
func (o *TaintOptions) Complete(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) (err error) {
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	// retrieves resource and taint args from args
	// also checks args to verify that all resources are specified before taints
	taintArgs := []string{}
	metTaintArg := false
	for _, s := range args {
		isTaint := strings.Contains(s, "=") || strings.HasSuffix(s, "-")
		switch {
		case !metTaintArg && isTaint:
			metTaintArg = true
			fallthrough
		case metTaintArg && isTaint:
			taintArgs = append(taintArgs, s)
		case !metTaintArg && !isTaint:
			o.resources = append(o.resources, s)
		case metTaintArg && !isTaint:
			return fmt.Errorf("all resources must be specified before taint changes: %s", s)
		}
	}

	if len(o.resources) < 1 {
		return fmt.Errorf("one or more resources must be specified as <resource> <name>")
	}
	if len(taintArgs) < 1 {
		return fmt.Errorf("at least one taint update is required")
	}

	if o.taintsToAdd, o.removeTaintKeys, err = parseTaints(taintArgs); err != nil {
		return cmdutil.UsageError(cmd, err.Error())
	}

	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	o.builder = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace()
	if o.all {
		o.builder = o.builder.SelectAllParam(o.all).ResourceTypes("node")
	} else {
		if len(o.resources) < 2 {
			return fmt.Errorf("at least one resource name must be specified since 'all' parameter is not set")
		}
		o.builder = o.builder.ResourceNames("node", o.resources[1:]...)
	}
	o.builder = o.builder.SelectorParam(o.selector).
		Flatten().
		Latest()

	o.f = f
	o.out = out
	o.cmd = cmd

	return nil
}

// Validate checks to the TaintOptions to see if there is sufficient information run the command.
func (o TaintOptions) Validate(args []string) error {
	resourceType := strings.ToLower(o.resources[0])
	if resourceType != "node" && resourceType != "nodes" {
		return fmt.Errorf("invalid resource type %s, only node(s) is supported", o.resources[0])
	}

	// check the format of taint args and checks removed taints aren't in the new taints list
	conflictKeys := []string{}
	removeTaintKeysSet := sets.NewString(o.removeTaintKeys...)
	for _, taint := range o.taintsToAdd {
		if removeTaintKeysSet.Has(taint.Key) {
			conflictKeys = append(conflictKeys, taint.Key)
		}
	}
	if len(conflictKeys) > 0 {
		return fmt.Errorf("can not both modify and remove the following taint(s) in the same command: %s", strings.Join(conflictKeys, ", "))
	}

	return nil
}

// RunTaint does the work
func (o TaintOptions) RunTaint() error {
	r := o.builder.Do()
	if err := r.Err(); err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		obj, err := info.Mapping.ConvertToVersion(info.Object, info.Mapping.GroupVersionKind.GroupVersion())
		if err != nil {
			return err
		}
		name, namespace := info.Name, info.Namespace
		oldData, err := json.Marshal(obj)
		if err != nil {
			return err
		}

		if err := o.updateTaints(obj); err != nil {
			return err
		}
		newData, err := json.Marshal(obj)
		if err != nil {
			return err
		}
		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, obj)
		createdPatch := err == nil
		if err != nil {
			glog.V(2).Infof("couldn't compute patch: %v", err)
		}

		mapping := info.ResourceMapping()
		client, err := o.f.ClientForMapping(mapping)
		if err != nil {
			return err
		}
		helper := resource.NewHelper(client, mapping)

		var outputObj runtime.Object
		if createdPatch {
			outputObj, err = helper.Patch(namespace, name, api.StrategicMergePatchType, patchBytes)
		} else {
			outputObj, err = helper.Replace(namespace, name, false, obj)
		}
		if err != nil {
			return err
		}

		mapper, _ := o.f.Object(cmdutil.GetIncludeThirdPartyAPIs(o.cmd))
		outputFormat := cmdutil.GetFlagString(o.cmd, "output")
		if outputFormat != "" {
			return o.f.PrintObject(o.cmd, mapper, outputObj, o.out)
		}

		cmdutil.PrintSuccess(mapper, false, o.out, info.Mapping.Resource, info.Name, "tainted")
		return nil
	})
}

// validateNoTaintOverwrites validates that when overwrite is false, to-be-updated taints don't exist in the node taint list (yet)
func validateNoTaintOverwrites(accessor meta.Object, taints []api.Taint) error {
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		return nil
	}

	allErrs := []error{}
	oldTaints, err := api.GetTaintsFromNodeAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, err)
		return utilerrors.NewAggregate(allErrs)
	}

	for _, taint := range taints {
		for _, oldTaint := range oldTaints {
			if taint.Key == oldTaint.Key {
				allErrs = append(allErrs, fmt.Errorf("Node '%s' already has a taint (%+v), and --overwrite is false", accessor.GetName(), taint))
				break
			}
		}
	}
	return utilerrors.NewAggregate(allErrs)
}

// updateTaints updates taints of obj
func (o TaintOptions) updateTaints(obj runtime.Object) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	if !o.overwrite {
		if err := validateNoTaintOverwrites(accessor, o.taintsToAdd); err != nil {
			return err
		}
	}

	annotations := accessor.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}

	newTaints, err := reorganizeTaints(accessor, o.overwrite, o.taintsToAdd, o.removeTaintKeys)
	if err != nil {
		return err
	}
	taintsData, err := json.Marshal(newTaints)
	if err != nil {
		return err
	}
	annotations[api.TaintsAnnotationKey] = string(taintsData)
	accessor.SetAnnotations(annotations)

	return nil
}
