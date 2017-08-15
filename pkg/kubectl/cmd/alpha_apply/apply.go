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

package alpha_apply

import (
	"fmt"
	"io"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/kubernetes/bazel-kubernetes/external/go1.8.3.darwin-amd64/src/io/ioutil"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/apply/merge"
	"k8s.io/kubernetes/pkg/kubectl/apply/parse"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"os"
)

type ApplyOptions struct {
	FilenameOptions resource.FilenameOptions
	Selector        string
	Force           bool
	Prune           bool
	Cascade         bool
	GracePeriod     int
	Timeout         time.Duration
	cmdBaseName     string
}

const (
	// maxPatchRetry is the maximum number of conflicts retry for during a patch operation before returning failure
	maxPatchRetry = 5
)

var (
	applyLong = templates.LongDesc(i18n.T(`
		Apply a configuration to a resource by filename or stdin.
		The resource name must be specified. This resource will be created if it doesn't exist yet.
		To use 'apply', always create the resource initially with either 'apply' or 'create --save-config'.

		JSON and YAML formats are accepted.`))

	applyExample = templates.Examples(i18n.T(`
		# Apply the configuration in pod.json to a pod.
		kubectl apply -f ./pod.json`))
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
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().Bool("all", false, "Select all resources in the namespace of the specified resource types.")
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)

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

func RunApply(f cmdutil.Factory, cmd *cobra.Command, out, errOut io.Writer, options *ApplyOptions) error {
	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagBool(cmd, "openapi-validation"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder, err := f.NewUnstructuredBuilder(true)
	if err != nil {
		return err
	}

	// include the uninitialized objects by default if --prune is true
	// unless explicitly set --include-uninitialized=false
	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, options.Prune)

	r := builder.
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

	encoder := f.JSONEncoder()
	decoder := f.Decoder(false)

	oapi, err := f.OpenAPISchema()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {

		if err := info.Get(); err != nil {
			// Create
			return fmt.Errorf("Create not supported in alpha")
		}

		// Serialize the current configuration of the object from the server.
		remote, err := runtime.Encode(encoder, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("serializing current configuration from:\n%v\nfor:", info.Object), info.Source, err)
		}
		remoteMap := map[string]interface{}{}
		if len(remote) > 0 {
			if err := json.Unmarshal(remote, &remoteMap); err != nil {
				return mergepatch.ErrBadJSONDoc
			}
		}
		delete(remoteMap, "status")
		m := remoteMap["metadata"].(map[string]interface{})
		a := m["annotations"].(map[string]interface{})
		delete(a, "kubectl.kubernetes.io/last-applied-configuration")

		remoteBytes, err := json.Marshal(remoteMap)
		if err != nil {
			return err
		}
		remoteFile, err := ioutil.TempFile(os.TempDir(), "remote")
		if err != nil {
			return err
		}
		obj, _, err := decoder.Decode(remoteBytes, nil, nil)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "Remote: %s\n", remoteFile.Name())
		err = cmdutil.PrintResourceInfoForCommand(cmd, &resource.Info{Object: obj}, f, remoteFile)
		if err != nil {
			return err
		}

		// Retrieve the original configuration of the object from the annotation.
		recorded, err := kubectl.GetOriginalConfiguration(info.Mapping, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving original configuration from:\n%v\nfor:", info.Object), info.Source, err)
		}
		recordedMap := map[string]interface{}{}
		if len(remote) > 0 {
			if err := json.Unmarshal(recorded, &recordedMap); err != nil {
				return mergepatch.ErrBadJSONDoc
			}
		}
		delete(recordedMap, "status")

		// Get the modified configuration of the object. Embed the result
		// as an annotation in the modified configuration, so that it will appear
		// in the patch sent to the server.
		local, err := kubectl.GetModifiedConfiguration(info, true, encoder)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving modified configuration from:\n%v\nfor:", info), info.Source, err)
		}
		localMap := map[string]interface{}{}
		if len(remote) > 0 {
			if err := json.Unmarshal(local, &localMap); err != nil {
				return mergepatch.ErrBadJSONDoc
			}
		}
		delete(localMap, "status")

		mergeFactory := parse.Factory{oapi}

		element, err := mergeFactory.CreateElement(recordedMap, localMap, remoteMap)
		if err != nil {
			return err
		}

		result, err := element.Accept(merge.Create(merge.Options{}))
		if err != nil {
			return err
		}

		resultBytes, err := json.Marshal(result.MergedResult)
		if err != nil {
			return err
		}

		obj, _, err = decoder.Decode(resultBytes, nil, nil)
		if err != nil {
			return err
		}

		mergedFile, err := ioutil.TempFile(os.TempDir(), "merged")
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "Merged: %s\n", mergedFile.Name())
		err = cmdutil.PrintResourceInfoForCommand(cmd, &resource.Info{Object: obj}, f, mergedFile)
		if err != nil {
			return err
		}

		output := cmdutil.GetFlagString(cmd, "output")
		if len(output) > 0 {
			info.Object = obj
			return cmdutil.PrintResourceInfoForCommand(cmd, &resource.Info{Object: obj}, f, out)
		}

		// Print the result
		//fmt.Fprintf(out, "Result:\n%+v\n", result)
		return nil
	})
	return err
}
