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
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/api/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/i18n"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"
)

const (
	MODIFIED  = "modified"
	TAINTED   = "tainted"
	UNTAINTED = "untainted"
)

// TaintOptions have the data required to perform the taint operation
type TaintOptions struct {
	resources      []string
	taintsToAdd    []v1.Taint
	taintsToRemove []v1.Taint
	builder        *resource.Builder
	selector       string
	overwrite      bool
	all            bool
	f              cmdutil.Factory
	out            io.Writer
	cmd            *cobra.Command
}

var (
	taintLong = templates.LongDesc(i18n.T(`
		Update the taints on one or more nodes.

		* A taint consists of a key, value, and effect. As an argument here, it is expressed as key=value:effect.
		* The key must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[1]d characters.
		* The value must begin with a letter or number, and may contain letters, numbers, hyphens, dots, and underscores, up to %[2]d characters.
		* The effect must be NoSchedule, PreferNoSchedule or NoExecute.
		* Currently taint can only apply to node.`))

	taintExample = templates.Examples(i18n.T(`
		# Update node 'foo' with a taint with key 'dedicated' and value 'special-user' and effect 'NoSchedule'.
		# If a taint with that key and effect already exists, its value is replaced as specified.
		kubectl taint nodes foo dedicated=special-user:NoSchedule

		# Remove from node 'foo' the taint with key 'dedicated' and effect 'NoSchedule' if one exists.
		kubectl taint nodes foo dedicated:NoSchedule-

		# Remove from node 'foo' all the taints with key 'dedicated'
		kubectl taint nodes foo dedicated-

		# Add a taint with key 'dedicated' on nodes having label mylabel=X
		kubectl taint node -l myLabel=X  dedicated=foo:PreferNoSchedule`))
)

func NewCmdTaint(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TaintOptions{}

	validArgs := []string{"node"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "taint NODE NAME KEY_1=VAL_1:TAINT_EFFECT_1 ... KEY_N=VAL_N:TAINT_EFFECT_N",
		Short:   i18n.T("Update the taints on one or more nodes"),
		Long:    fmt.Sprintf(taintLong, validation.DNS1123SubdomainMaxLength, validation.LabelValueMaxLength),
		Example: taintExample,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, out, cmd, args); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(); err != nil {
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
	cmd.Flags().StringVarP(&options.selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().BoolVar(&options.overwrite, "overwrite", false, "If true, allow taints to be overwritten, otherwise reject taint updates that overwrite existing taints.")
	cmd.Flags().BoolVar(&options.all, "all", false, "select all nodes in the cluster")
	return cmd
}

// reorganizeTaints returns the updated set of taints, taking into account old taints that were not updated,
// old taints that were updated, old taints that were deleted, and new taints.
func reorganizeTaints(node *v1.Node, overwrite bool, taintsToAdd []v1.Taint, taintsToRemove []v1.Taint) (string, []v1.Taint, error) {
	newTaints := append([]v1.Taint{}, taintsToAdd...)
	oldTaints := node.Spec.Taints
	// add taints that already existing but not updated to newTaints
	added := addTaints(oldTaints, &newTaints)
	allErrs, deleted := deleteTaints(taintsToRemove, &newTaints)
	if (added && deleted) || overwrite {
		return MODIFIED, newTaints, utilerrors.NewAggregate(allErrs)
	} else if added {
		return TAINTED, newTaints, utilerrors.NewAggregate(allErrs)
	}
	return UNTAINTED, newTaints, utilerrors.NewAggregate(allErrs)
}

// deleteTaints deletes the given taints from the node's taintlist.
func deleteTaints(taintsToRemove []v1.Taint, newTaints *[]v1.Taint) ([]error, bool) {
	allErrs := []error{}
	var removed bool
	for _, taintToRemove := range taintsToRemove {
		removed = false
		if len(taintToRemove.Effect) > 0 {
			*newTaints, removed = v1helper.DeleteTaint(*newTaints, &taintToRemove)
		} else {
			*newTaints, removed = v1helper.DeleteTaintsByKey(*newTaints, taintToRemove.Key)
		}
		if !removed {
			allErrs = append(allErrs, fmt.Errorf("taint %q not found", taintToRemove.ToString()))
		}
	}
	return allErrs, removed
}

// addTaints adds the newTaints list to existing ones and updates the newTaints List.
// TODO: This needs a rewrite to take only the new values instead of appended newTaints list to be consistent.
func addTaints(oldTaints []v1.Taint, newTaints *[]v1.Taint) bool {
	for _, oldTaint := range oldTaints {
		existsInNew := false
		for _, taint := range *newTaints {
			if taint.MatchTaint(&oldTaint) {
				existsInNew = true
				break
			}
		}
		if !existsInNew {
			*newTaints = append(*newTaints, oldTaint)
		}
	}
	return len(oldTaints) != len(*newTaints)
}

func parseTaints(spec []string) ([]v1.Taint, []v1.Taint, error) {
	var taints, taintsToRemove []v1.Taint
	uniqueTaints := map[v1.TaintEffect]sets.String{}

	for _, taintSpec := range spec {
		if strings.Index(taintSpec, "=") != -1 && strings.Index(taintSpec, ":") != -1 {
			newTaint, err := utiltaints.ParseTaint(taintSpec)
			if err != nil {
				return nil, nil, err
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
			var effect v1.TaintEffect
			if strings.Index(taintKey, ":") != -1 {
				parts := strings.Split(taintKey, ":")
				taintKey = parts[0]
				effect = v1.TaintEffect(parts[1])
			}
			taintsToRemove = append(taintsToRemove, v1.Taint{Key: taintKey, Effect: effect})
		} else {
			return nil, nil, fmt.Errorf("unknown taint spec: %v", taintSpec)
		}
	}
	return taints, taintsToRemove, nil
}

// Complete adapts from the command line args and factory to the data required.
func (o *TaintOptions) Complete(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) (err error) {
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
	o.builder = f.NewBuilder(true).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace()
	if o.selector != "" {
		o.builder = o.builder.SelectorParam(o.selector).ResourceTypes("node")
	}
	if o.all {
		o.builder = o.builder.SelectAllParam(o.all).ResourceTypes("node").Flatten().Latest()
	}
	if !o.all && o.selector == "" && len(o.resources) >= 2 {
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

// validateFlags checks for the validation of flags for kubectl taints.
func (o TaintOptions) validateFlags() error {
	// Cannot have a non-empty selector and all flag set. They are mutually exclusive.
	if o.all && o.selector != "" {
		return fmt.Errorf("setting 'all' parameter with a non empty selector is prohibited.")
	}
	// If both selector and all are not set.
	if !o.all && o.selector == "" {
		if len(o.resources) < 2 {
			return fmt.Errorf("at least one resource name must be specified since 'all' parameter is not set")
		} else {
			return nil
		}
	}
	return nil
}

// Validate checks to the TaintOptions to see if there is sufficient information run the command.
func (o TaintOptions) Validate() error {
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
	return o.validateFlags()
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
		operation, err := o.updateTaints(obj)
		if err != nil {
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
			outputObj, err = helper.Patch(namespace, name, types.StrategicMergePatchType, patchBytes)
		} else {
			outputObj, err = helper.Replace(namespace, name, false, obj)
		}
		if err != nil {
			return err
		}

		mapper, _ := o.f.Object()
		outputFormat := cmdutil.GetFlagString(o.cmd, "output")
		if outputFormat != "" {
			return o.f.PrintObject(o.cmd, false, mapper, outputObj, o.out)
		}

		cmdutil.PrintSuccess(mapper, false, o.out, info.Mapping.Resource, info.Name, false, operation)
		return nil
	})
}

// validateNoTaintOverwrites validates that when overwrite is false, to-be-updated taints don't exist in the node taint list (yet)
func validateNoTaintOverwrites(node *v1.Node, taints []v1.Taint) error {
	allErrs := []error{}
	oldTaints := node.Spec.Taints
	for _, taint := range taints {
		for _, oldTaint := range oldTaints {
			if taint.Key == oldTaint.Key && taint.Effect == oldTaint.Effect {
				allErrs = append(allErrs, fmt.Errorf("Node '%s' already has a taint with key (%s) and effect (%v), and --overwrite is false", node.Name, taint.Key, taint.Effect))
				break
			}
		}
	}
	return utilerrors.NewAggregate(allErrs)
}

// updateTaints applies a taint option(o) to a node in cluster after computing the net effect of operation(i.e. does it result in an overwrite?), it reports back the end result in a way that user can easily interpret.
func (o TaintOptions) updateTaints(obj runtime.Object) (string, error) {
	node, ok := obj.(*v1.Node)
	if !ok {
		return "", fmt.Errorf("unexpected type %T, expected Node", obj)
	}
	if !o.overwrite {
		if err := validateNoTaintOverwrites(node, o.taintsToAdd); err != nil {
			return "", err
		}
	}
	operation, newTaints, err := reorganizeTaints(node, o.overwrite, o.taintsToAdd, o.taintsToRemove)
	if err != nil {
		return "", err
	}
	node.Spec.Taints = newTaints
	return operation, nil
}
