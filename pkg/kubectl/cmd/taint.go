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
	resources      []string
	taintsToAdd    []api.Taint
	taintsToRemove []api.Taint
	builder        *resource.Builder
	selector       string
	overwrite      bool
	all            bool
	f              *cmdutil.Factory
	out            io.Writer
	cmd            *cobra.Command
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
		# If a taint with that key and effect already exists, its value is replaced as specified.
		kubectl taint nodes foo dedicated=special-user:NoSchedule

		# Remove from node 'foo' the taint with key 'dedicated' and effect 'NoSchedule' if one exists.
		kubectl taint nodes foo dedicated:NoSchedule-

		# Remove from node 'foo' all the taints with key 'dedicated'
		kubectl taint nodes foo dedicated-`)
)

func NewCmdTaint(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TaintOptions{}

	validArgs := []string{"node"}
	argAliases := kubectl.ResourceAliases(validArgs)

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
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmdutil.AddValidateFlags(cmd)

	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmd.Flags().StringVarP(&options.selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.overwrite, "overwrite", false, "If true, allow taints to be overwritten, otherwise reject taint updates that overwrite existing taints.")
	cmd.Flags().BoolVar(&options.all, "all", false, "select all nodes in the cluster")
	return cmd
}

func deleteTaint(taints []api.Taint, taintToDelete api.Taint) ([]api.Taint, error) {
	newTaints := []api.Taint{}
	found := false
	for _, taint := range taints {
		if taint.Key == taintToDelete.Key &&
			(len(taintToDelete.Effect) == 0 || taint.Effect == taintToDelete.Effect) {
			found = true
			continue
		}
		newTaints = append(newTaints, taint)
	}

	if !found {
		return nil, fmt.Errorf("taint key=\"%s\" and effect=\"%s\" not found.", taintToDelete.Key, taintToDelete.Effect)
	}
	return newTaints, nil
}

// reorganizeTaints returns the updated set of taints, taking into account old taints that were not updated,
// old taints that were updated, old taints that were deleted, and new taints.
func reorganizeTaints(accessor meta.Object, overwrite bool, taintsToAdd []api.Taint, taintsToRemove []api.Taint) ([]api.Taint, error) {
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
			if taint.MatchTaint(oldTaint) {
				existsInNew = true
				break
			}
		}
		if !existsInNew {
			newTaints = append(newTaints, oldTaint)
		}
	}

	allErrs := []error{}
	for _, taintToRemove := range taintsToRemove {
		newTaints, err = deleteTaint(newTaints, taintToRemove)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}
	return newTaints, utilerrors.NewAggregate(allErrs)
}

func parseTaints(spec []string) ([]api.Taint, []api.Taint, error) {
	var taints, taintsToRemove []api.Taint
	uniqueTaints := map[api.TaintEffect]sets.String{}

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

			// validate if taint is unique by <key, effect>
			if len(uniqueTaints[newTaint.Effect]) > 0 && uniqueTaints[newTaint.Effect].Has(newTaint.Key) {
				return nil, nil, fmt.Errorf("duplicated taints with the same key and effect: %v", newTaint)
			}
			// add taint to existingTaints for uniqueness check
			if len(uniqueTaints[newTaint.Effect]) == 0 {
				uniqueTaints[newTaint.Effect] = sets.String{}
			}
			uniqueTaints[newTaint.Effect].Insert(newTaint.Key)

			taints = append(taints, newTaint)
		} else if strings.HasSuffix(taintSpec, "-") {
			taintKey := taintSpec[:len(taintSpec)-1]
			var effect api.TaintEffect
			if strings.Index(taintKey, ":") != -1 {
				parts := strings.Split(taintKey, ":")
				taintKey = parts[0]
				effect = api.TaintEffect(parts[1])
			}
			taintsToRemove = append(taintsToRemove, api.Taint{Key: taintKey, Effect: effect})
		} else {
			return nil, nil, fmt.Errorf("unknown taint spec: %v", taintSpec)
		}
	}
	return taints, taintsToRemove, nil
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

	if o.taintsToAdd, o.taintsToRemove, err = parseTaints(taintArgs); err != nil {
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
	validResources, isValidResource := append(kubectl.ResourceAliases([]string{"node"}), "node"), false
	for _, validResource := range validResources {
		if resourceType == validResource {
			isValidResource = true
			break
		}
	}
	if !isValidResource {
		return fmt.Errorf("invalid resource type %s, only %q are supported", o.resources[0], validResources)
	}

	// check the format of taint args and checks removed taints aren't in the new taints list
	var conflictTaints []string
	for _, taintAdd := range o.taintsToAdd {
		for _, taintRemove := range o.taintsToRemove {
			if taintAdd.Key != taintRemove.Key {
				continue
			}
			if len(taintRemove.Effect) == 0 || taintAdd.Effect == taintRemove.Effect {
				conflictTaint := fmt.Sprintf("{\"%s\":\"%s\"}", taintRemove.Key, taintRemove.Effect)
				conflictTaints = append(conflictTaints, conflictTaint)
			}
		}
	}
	if len(conflictTaints) > 0 {
		return fmt.Errorf("can not both modify and remove the following taint(s) in the same command: %s", strings.Join(conflictTaints, ", "))
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

		cmdutil.PrintSuccess(mapper, false, o.out, info.Mapping.Resource, info.Name, false, "tainted")
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
			if taint.Key == oldTaint.Key && taint.Effect == oldTaint.Effect {
				allErrs = append(allErrs, fmt.Errorf("Node '%s' already has a taint with key (%s) and effect (%v), and --overwrite is false", accessor.GetName(), taint.Key, taint.Effect))
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

	newTaints, err := reorganizeTaints(accessor, o.overwrite, o.taintsToAdd, o.taintsToRemove)
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
