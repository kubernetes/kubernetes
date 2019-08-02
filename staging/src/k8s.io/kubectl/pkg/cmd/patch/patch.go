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

package patch

import (
	"fmt"
	"reflect"
	"strings"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/spf13/cobra"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var patchTypes = map[string]types.PatchType{"json": types.JSONPatchType, "merge": types.MergePatchType, "strategic": types.StrategicMergePatchType}

// PatchOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type PatchOptions struct {
	resource.FilenameOptions

	RecordFlags *genericclioptions.RecordFlags
	PrintFlags  *genericclioptions.PrintFlags
	ToPrinter   func(string) (printers.ResourcePrinter, error)
	Recorder    genericclioptions.Recorder

	Local     bool
	PatchType string
	Patch     string

	namespace                    string
	enforceNamespace             bool
	dryRun                       bool
	outputFormat                 string
	args                         []string
	builder                      *resource.Builder
	unstructuredClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)

	genericclioptions.IOStreams
}

var (
	patchLong = templates.LongDesc(i18n.T(`
		Update field(s) of a resource using strategic merge patch, a JSON merge patch, or a JSON patch.

		JSON and YAML formats are accepted.`))

	patchExample = templates.Examples(i18n.T(`
		# Partially update a node using a strategic merge patch. Specify the patch as JSON.
		kubectl patch node k8s-node-1 -p '{"spec":{"unschedulable":true}}'

		# Partially update a node using a strategic merge patch. Specify the patch as YAML.
		kubectl patch node k8s-node-1 -p $'spec:\n unschedulable: true'

		# Partially update a node identified by the type and name specified in "node.json" using strategic merge patch.
		kubectl patch -f node.json -p '{"spec":{"unschedulable":true}}'

		# Update a container's image; spec.containers[*].name is required because it's a merge key.
		kubectl patch pod valid-pod -p '{"spec":{"containers":[{"name":"kubernetes-serve-hostname","image":"new image"}]}}'

		# Update a container's image using a json patch with positional arrays.
		kubectl patch pod valid-pod --type='json' -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"new image"}]'`))
)

func NewPatchOptions(ioStreams genericclioptions.IOStreams) *PatchOptions {
	return &PatchOptions{
		RecordFlags: genericclioptions.NewRecordFlags(),
		Recorder:    genericclioptions.NoopRecorder{},
		PrintFlags:  genericclioptions.NewPrintFlags("patched").WithTypeSetter(scheme.Scheme),
		IOStreams:   ioStreams,
	}
}

func NewCmdPatch(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewPatchOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "patch (-f FILENAME | TYPE NAME) -p PATCH",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Update field(s) of a resource using strategic merge patch"),
		Long:                  patchLong,
		Example:               patchExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunPatch())
		},
	}

	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().StringVarP(&o.Patch, "patch", "p", "", "The patch to be applied to the resource JSON file.")
	cmd.MarkFlagRequired("patch")
	cmd.Flags().StringVar(&o.PatchType, "type", "strategic", fmt.Sprintf("The type of patch being provided; one of %v", sets.StringKeySet(patchTypes).List()))
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, "identifying the resource to update")
	cmd.Flags().BoolVar(&o.Local, "local", o.Local, "If true, patch will operate on the content of the file, not the server-side resource.")

	return cmd
}

func (o *PatchOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.outputFormat = cmdutil.GetFlagString(cmd, "output")
	o.dryRun = cmdutil.GetFlagBool(cmd, "dry-run")

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		if o.dryRun {
			o.PrintFlags.Complete("%s (dry run)")
		}

		return o.PrintFlags.ToPrinter()
	}

	o.namespace, o.enforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	o.args = args
	o.builder = f.NewBuilder()
	o.unstructuredClientForMapping = f.UnstructuredClientForMapping

	return nil
}

func (o *PatchOptions) Validate() error {
	if o.Local && len(o.args) != 0 {
		return fmt.Errorf("cannot specify --local and server resources")
	}
	if len(o.Patch) == 0 {
		return fmt.Errorf("must specify -p to patch")
	}
	if len(o.PatchType) != 0 {
		if _, ok := patchTypes[strings.ToLower(o.PatchType)]; !ok {
			return fmt.Errorf("--type must be one of %v, not %q", sets.StringKeySet(patchTypes).List(), o.PatchType)
		}
	}

	return nil
}

func (o *PatchOptions) RunPatch() error {
	patchType := types.StrategicMergePatchType
	if len(o.PatchType) != 0 {
		patchType = patchTypes[strings.ToLower(o.PatchType)]
	}

	patchBytes, err := yaml.ToJSON([]byte(o.Patch))
	if err != nil {
		return fmt.Errorf("unable to parse %q: %v", o.Patch, err)
	}

	r := o.builder.
		Unstructured().
		ContinueOnError().
		LocalParam(o.Local).
		NamespaceParam(o.namespace).DefaultNamespace().
		FilenameParam(o.enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(false, o.args...).
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
		count++
		name, namespace := info.Name, info.Namespace

		if !o.Local && !o.dryRun {
			mapping := info.ResourceMapping()
			client, err := o.unstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}

			helper := resource.NewHelper(client, mapping)
			patchedObj, err := helper.Patch(namespace, name, patchType, patchBytes, nil)
			if err != nil {
				return err
			}

			didPatch := !reflect.DeepEqual(info.Object, patchedObj)

			// if the recorder makes a change, compute and create another patch
			if mergePatch, err := o.Recorder.MakeRecordMergePatch(patchedObj); err != nil {
				klog.V(4).Infof("error recording current command: %v", err)
			} else if len(mergePatch) > 0 {
				if recordedObj, err := helper.Patch(info.Namespace, info.Name, types.MergePatchType, mergePatch, nil); err != nil {
					klog.V(4).Infof("error recording reason: %v", err)
				} else {
					patchedObj = recordedObj
				}
			}

			printer, err := o.ToPrinter(patchOperation(didPatch))
			if err != nil {
				return err
			}
			return printer.PrintObj(patchedObj, o.Out)
		}

		originalObjJS, err := runtime.Encode(unstructured.UnstructuredJSONScheme, info.Object)
		if err != nil {
			return err
		}

		originalPatchedObjJS, err := getPatchedJSON(patchType, originalObjJS, patchBytes, info.Object.GetObjectKind().GroupVersionKind(), scheme.Scheme)
		if err != nil {
			return err
		}

		targetObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, originalPatchedObjJS)
		if err != nil {
			return err
		}

		didPatch := !reflect.DeepEqual(info.Object, targetObj)
		printer, err := o.ToPrinter(patchOperation(didPatch))
		if err != nil {
			return err
		}
		return printer.PrintObj(targetObj, o.Out)
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to patch")
	}
	return nil
}

func getPatchedJSON(patchType types.PatchType, originalJS, patchJS []byte, gvk schema.GroupVersionKind, creater runtime.ObjectCreater) ([]byte, error) {
	switch patchType {
	case types.JSONPatchType:
		patchObj, err := jsonpatch.DecodePatch(patchJS)
		if err != nil {
			return nil, err
		}
		bytes, err := patchObj.Apply(originalJS)
		// TODO: This is pretty hacky, we need a better structured error from the json-patch
		if err != nil && strings.Contains(err.Error(), "doc is missing key") {
			msg := err.Error()
			ix := strings.Index(msg, "key:")
			key := msg[ix+5:]
			return bytes, fmt.Errorf("Object to be patched is missing field (%s)", key)
		}
		return bytes, err

	case types.MergePatchType:
		return jsonpatch.MergePatch(originalJS, patchJS)

	case types.StrategicMergePatchType:
		// get a typed object for this GVK if we need to apply a strategic merge patch
		obj, err := creater.New(gvk)
		if err != nil {
			return nil, fmt.Errorf("cannot apply strategic merge patch for %s locally, try --type merge", gvk.String())
		}
		return strategicpatch.StrategicMergePatch(originalJS, patchJS, obj)

	default:
		// only here as a safety net - go-restful filters content-type
		return nil, fmt.Errorf("unknown Content-Type header for patch: %v", patchType)
	}
}

func patchOperation(didPatch bool) string {
	if didPatch {
		return "patched"
	}
	return "patched (no change)"
}
