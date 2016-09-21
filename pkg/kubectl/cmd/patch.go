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

	"github.com/evanphx/json-patch"
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
	"k8s.io/kubernetes/pkg/util/yaml"
)

var patchTypes = map[string]api.PatchType{"json": api.JSONPatchType, "merge": api.MergePatchType, "strategic": api.StrategicMergePatchType}

// PatchOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type PatchOptions struct {
	Filenames []string
	Recursive bool
	Local     bool

	OutputFormat string
}

var (
	patch_long = dedent.Dedent(`
		Update field(s) of a resource using strategic merge patch

		JSON and YAML formats are accepted.

		Please refer to the models in https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/release-1.4/docs/api-reference/v1/definitions.html to find if a field is mutable.`)
	patch_example = dedent.Dedent(`

		# Partially update a node using strategic merge patch
		kubectl patch node k8s-node-1 -p '{"spec":{"unschedulable":true}}'

		# Partially update a node identified by the type and name specified in "node.json" using strategic merge patch
		kubectl patch -f node.json -p '{"spec":{"unschedulable":true}}'

		# Update a container's image; spec.containers[*].name is required because it's a merge key
		kubectl patch pod valid-pod -p '{"spec":{"containers":[{"name":"kubernetes-serve-hostname","image":"new image"}]}}'

		# Update a container's image using a json patch with positional arrays
		kubectl patch pod valid-pod --type='json' -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"new image"}]'`)
)

func NewCmdPatch(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &PatchOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "patch (-f FILENAME | TYPE NAME) -p PATCH",
		Short:   "Update field(s) of a resource using strategic merge patch",
		Long:    patch_long,
		Example: patch_example,
		Run: func(cmd *cobra.Command, args []string) {
			options.OutputFormat = cmdutil.GetFlagString(cmd, "output")
			err := RunPatch(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmd.Flags().StringP("patch", "p", "", "The patch to be applied to the resource JSON file.")
	cmd.MarkFlagRequired("patch")
	cmd.Flags().String("type", "strategic", fmt.Sprintf("The type of patch being provided; one of %v", sets.StringKeySet(patchTypes).List()))
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	usage := "Filename, directory, or URL to a file identifying the resource to update"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)

	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, patch will operate on the content of the file, not the server-side resource.")

	return cmd
}

func RunPatch(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *PatchOptions) error {
	switch {
	case options.Local && len(args) != 0:
		return fmt.Errorf("cannot specify --local and server resources")
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	patchType := api.StrategicMergePatchType
	patchTypeString := strings.ToLower(cmdutil.GetFlagString(cmd, "type"))
	if len(patchTypeString) != 0 {
		ok := false
		patchType, ok = patchTypes[patchTypeString]
		if !ok {
			return cmdutil.UsageError(cmd, fmt.Sprintf("--type must be one of %v, not %q", sets.StringKeySet(patchTypes).List(), patchTypeString))
		}
	}

	patch := cmdutil.GetFlagString(cmd, "patch")
	if len(patch) == 0 {
		return cmdutil.UsageError(cmd, "Must specify -p to patch")
	}
	patchBytes, err := yaml.ToJSON([]byte(patch))
	if err != nil {
		return fmt.Errorf("unable to parse %q: %v", patch, err)
	}

	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
		ResourceTypeOrNameArgs(false, args...).
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
		name, namespace := info.Name, info.Namespace
		mapping := info.ResourceMapping()
		client, err := f.ClientForMapping(mapping)
		if err != nil {
			return err
		}

		if !options.Local {
			helper := resource.NewHelper(client, mapping)
			_, err := helper.Patch(namespace, name, patchType, patchBytes)
			if err != nil {
				return err
			}
			if cmdutil.ShouldRecord(cmd, info) {
				// don't return an error on failure.  The patch itself succeeded, its only the hint for that change that failed
				// don't bother checking for failures of this replace, because a failure to indicate the hint doesn't fail the command
				// also, don't force the replacement.  If the replacement fails on a resourceVersion conflict, then it means this
				// record hint is likely to be invalid anyway, so avoid the bad hint
				patch, err := cmdutil.ChangeResourcePatch(info, f.Command())
				if err == nil {
					helper.Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch)
				}
			}
			count++

			if options.OutputFormat == "name" || len(options.OutputFormat) == 0 {
				cmdutil.PrintSuccess(mapper, options.OutputFormat == "name", out, "", name, false, "patched")
			}
			return nil
		}

		count++

		patchedObj, err := api.Scheme.DeepCopy(info.VersionedObject)
		if err != nil {
			return err
		}
		originalObjJS, err := runtime.Encode(api.Codecs.LegacyCodec(mapping.GroupVersionKind.GroupVersion()), info.VersionedObject.(runtime.Object))
		if err != nil {
			return err
		}
		originalPatchedObjJS, err := getPatchedJSON(patchType, originalObjJS, patchBytes, patchedObj.(runtime.Object))
		if err != nil {
			return err
		}
		targetObj, err := runtime.Decode(api.Codecs.UniversalDecoder(), originalPatchedObjJS)
		if err != nil {
			return err
		}
		// TODO: if we ever want to go generic, this allows a clean -o yaml without trying to print columns or anything
		// rawExtension := &runtime.Unknown{
		//	Raw: originalPatchedObjJS,
		// }

		printer, err := f.PrinterForMapping(cmd, mapping, false)
		if err != nil {
			return err
		}
		if err := printer.PrintObj(targetObj, out); err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to patch")
	}
	return nil
}

func getPatchedJSON(patchType api.PatchType, originalJS, patchJS []byte, obj runtime.Object) ([]byte, error) {
	switch patchType {
	case api.JSONPatchType:
		patchObj, err := jsonpatch.DecodePatch(patchJS)
		if err != nil {
			return nil, err
		}
		return patchObj.Apply(originalJS)

	case api.MergePatchType:
		return jsonpatch.MergePatch(originalJS, patchJS)

	case api.StrategicMergePatchType:
		return strategicpatch.StrategicMergePatchData(originalJS, patchJS, obj)

	default:
		// only here as a safety net - go-restful filters content-type
		return nil, fmt.Errorf("unknown Content-Type header for patch: %v", patchType)
	}
}
