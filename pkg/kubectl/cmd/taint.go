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
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
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
				cmdutil.CheckErr(cmdutil.UsageErrorf(cmd, err.Error()))
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
	cmd.Flags().StringVarP(&options.selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.overwrite, "overwrite", false, "If true, allow taints to be overwritten, otherwise reject taint updates that overwrite existing taints.")
	cmd.Flags().BoolVar(&options.all, "all", false, "Select all nodes in the cluster")
	return cmd
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

	if o.taintsToAdd, o.taintsToRemove, err = taintutils.ParseTaints(taintArgs); err != nil {
		return cmdutil.UsageErrorf(cmd, err.Error())
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

// updateTaints applies a taint option(o) to a node in cluster after computing the net effect of operation(i.e. does it result in an overwrite?), it reports back the end result in a way that user can easily interpret.
func (o TaintOptions) updateTaints(obj runtime.Object) (string, error) {
	node, ok := obj.(*v1.Node)
	if !ok {
		return "", fmt.Errorf("unexpected type %T, expected Node", obj)
	}
	if !o.overwrite {
		if exists := taintutils.CheckIfTaintsAlreadyExists(node.Spec.Taints, o.taintsToAdd); len(exists) != 0 {
			return "", fmt.Errorf("Node %s already has %v taint(s) with same effect(s) and --overwrite is false", node.Name, exists)
		}
	}
	operation, newTaints, err := taintutils.ReorganizeTaints(node, o.overwrite, o.taintsToAdd, o.taintsToRemove)
	if err != nil {
		return "", err
	}
	node.Spec.Taints = newTaints
	return operation, nil
}
