/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	//"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdtools"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type DiffLastAppliedOptions struct {
	FilenameOptions  resource.FilenameOptions
	Mapper           meta.RESTMapper
	Typer            runtime.ObjectTyper
	Namespace        string
	EnforceNamespace bool
	DiffBuffer1      []byte
	DiffBuffer2      []byte
	Output           string
	Codec            runtime.Encoder
	Factory          cmdutil.Factory
	Out              io.Writer
	ErrOut           io.Writer
	ext              string
}

var (
	applyDiffLastAppliedLong = templates.LongDesc(`
		Opens up a 2-way diff in the default diff viewer. This should follow the same semantics as git diff.
		It should accept either a flag --diff-viewer=meld or check the environment variable KUBECTL_EXTERNAL_DIFF=meld.
		If neither is specified, the diff command should be used.
		`)

	applyDiffLastAppliedExample = templates.Examples(`
		#
		kubectl apply diff-last-applied -f deploy.yaml
		`)
)

func NewCmdApplyDiffLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &DiffLastAppliedOptions{Out: out, ErrOut: err}
	cmd := &cobra.Command{
		Use:     "diff-last-applied -f FILENAME",
		Short:   " the last-applied-configuration annotation on a live object to match the contents of a file.",
		Long:    applyDiffLastAppliedLong,
		Example: applyDiffLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.ValidateOutputArgs(cmd))
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate(f, cmd))
			cmdutil.CheckErr(options.RunDiffLastApplied(f, cmd))
		},
	}

	cmd.Flags().StringP("output", "o", "", "Output format. Must be one of yaml|json")
	usage := "that contains the last-applied-configuration annotations"
	kubectl.AddJsonFilenameFlag(cmd, &options.FilenameOptions.Filenames, "Filename, directory, or URL to files "+usage)

	return cmd
}

func (o *DiffLastAppliedOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.ext = "." + o.Output
	r := resource.NewBuilder(o.Mapper, f.CategoryExpander(), o.Typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme).
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		Latest().
		Flatten().
		Do()
	err := r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		/*
			o.DiffBuffer1, err = runtime.Encode(o.Codec, info.VersionedObject)
			if err != nil {
				return err
			}

			// Verify the object exists in the cluster
			if err := info.Get(); err != nil {
				if errors.IsNotFound(err) {
					return err
				} else {
					return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
				}
			}
			o.DiffBuffer2, err = kubectl.GetOriginalConfiguration(info.Mapping, info.Object)
			if err != nil {
				return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
			}
			if o.DiffBuffer2 == nil {
				return cmdutil.UsageError(cmd, "no last-applied-configuration annotation found on resource: %s, to create the annotation, run the command with --create-annotation", info.Name)
			}
		*/

		return nil
	})
	if err != nil {
		return err
	}
	return nil
}

func (o *DiffLastAppliedOptions) Validate(f cmdutil.Factory, cmd *cobra.Command) error {
	return nil
}

func (o *DiffLastAppliedOptions) RunDiffLastApplied(f cmdutil.Factory, cmd *cobra.Command) error {
	buf1, err := formatConvert(o.DiffBuffer1, o.Output)
	if err != nil {
		return err
	}
	r1 := bytes.NewReader(buf1)
	buf2, err := formatConvert(o.DiffBuffer2, o.Output)
	if err != nil {
		return err
	}
	r2 := bytes.NewReader(buf2)

	diff := cmdtools.NewDefaultCmdTool("diff", f.DiffEnvs())

	_, _, err = diff.LaunchTempFile(fmt.Sprintf("%s-diff-", filepath.Base(os.Args[0])), o.ext, r1, r2)
	if err != nil {
		return err
	}

	return nil
}

//TODO: try to remove duplicate code when move apply to it's own package
func formatConvert(input []byte, o string) ([]byte, error) {
	switch o {
	case "json":
		jsonBuffer := &bytes.Buffer{}
		err := json.Indent(jsonBuffer, []byte(input), "", "  ")
		if err != nil {
			return nil, err
		}
		return jsonBuffer.Bytes(), nil
	case "yaml":
		yamlOutput, err := yaml.JSONToYAML([]byte(input))
		if err != nil {
			return nil, err
		}
		return yamlOutput, nil
	}
	return nil, fmt.Errorf("Unexpected -o output mode")
}

//TODO: reduce duplicate code here
//ValidateOutputArgs is same as `func (o *ViewLastAppliedOptions) ValidateOutputArgs`
//will remove it when move apply to it's own package
func (o *DiffLastAppliedOptions) ValidateOutputArgs(cmd *cobra.Command) error {
	format := cmdutil.GetFlagString(cmd, "output")
	switch format {
	case "json":
		o.Output = "json"
		return nil
		// If flag -o is not specified, use yaml as default
	case "yaml", "":
		o.Output = "yaml"
		return nil
	default:
		return cmdutil.UsageError(cmd, "Unexpected -o output mode: %s, the flag 'output' must be one of yaml|json", format)
	}
}
