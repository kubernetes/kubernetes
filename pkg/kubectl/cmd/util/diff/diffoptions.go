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

package diff

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// Options contains all the options for running the apply diff-last-applied cli command.
type Options struct {
	FilenameOptions    resource.FilenameOptions
	Namespace          string
	EnforceNamespace   bool
	DiffBufferModified []byte
	DiffBufferOriginal []byte
	Output             string
	Factory            cmdutil.Factory
	Out                io.Writer
	ErrOut             io.Writer
}

// Complete completes all the required options for apply diff-last-applied
func (o *Options) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	if o.Output != "yaml" && o.Output != "json" {
		return fmt.Errorf("invalid output format %q, only yaml|json supported", o.Output)
	}

	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.DefaultNamespace()
	if err != nil {
		return err
	}
	codec := runtime.NewCodec(f.JSONEncoder(), f.Decoder(true))

	r := resource.NewBuilder(mapper, f.CategoryExpander(), typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme).
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		o.DiffBufferModified, err = runtime.Encode(codec, info.VersionedObject)
		if err != nil {
			return err
		}

		// Verify the object exists in the cluster
		if err := info.Get(); err != nil {
			if apierrors.IsNotFound(err) {
				return err
			}
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
		}

		o.DiffBufferOriginal, err = kubectl.GetOriginalConfiguration(info.Mapping, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
		}
		if o.DiffBufferOriginal == nil {
			return cmdutil.UsageErrorf(cmd, "no last-applied-configuration annotation found on resource: %s, to create the annotation, run the command with --create-annotation", info.Name)
		}

		//decode and delete the annotations fields
		targetObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, o.DiffBufferOriginal)
		if err != nil {
			return err
		}

		annotationMap, err := info.Mapping.MetadataAccessor.Annotations(targetObj)
		if err != nil {
			return err
		}

		//If there are no annotations in the object, meaning the `last-applied-config` annotation
		// is the only one annotation in the original object, we can remove the `annotations`
		// field before calculating the diff.
		if len(annotationMap) == 0 {
			err := deleteEmptyAnnotations(targetObj)
			if err != nil {
				return err
			}
		}

		o.DiffBufferOriginal, err = runtime.Encode(codec, targetObj)
		return err
	})

	return err
}

func deleteEmptyAnnotations(obj runtime.Object) error {
	if unstructured, ok := obj.(runtime.Unstructured); ok {
		if tarObj, ok := unstructured.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
			delete(tarObj, "annotations")
			return nil
		}
	}
	return fmt.Errorf("delete empty annotations failed")
}

// Validate validates all the required options for apply diff-last-applied.
func (o *Options) Validate(f cmdutil.Factory, cmd *cobra.Command) error {
	return nil
}

// RunDiffLastApplied implements all the necessary functionality for apply diff-last-applied.
func (o *Options) RunDiffLastApplied(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error
	o.DiffBufferModified, err = FormatConvert(o.DiffBufferModified, o.Output)
	if err != nil {
		return err
	}
	o.DiffBufferOriginal, err = FormatConvert(o.DiffBufferOriginal, o.Output)
	if err != nil {
		return err
	}
	diff := cmdutil.NewDefaultCmdTool(cmdutil.DiffCmd, f.DiffEnvs())

	tmpFileSlice := []cmdutil.TempFile{
		//be careful with the order of DiffBufferModified and DiffBufferOriginal
		{Prefix: "diff-last-applied-original-", Suffix: "." + o.Output, Buffer: bytes.NewReader(o.DiffBufferOriginal)},
		{Prefix: "diff-last-applied-modified-", Suffix: "." + o.Output, Buffer: bytes.NewReader(o.DiffBufferModified)},
	}
	_, err = diff.LaunchTempFiles(tmpFileSlice...)

	return err
}

// FormatConvert accept a JSON-encoded byte array and a convert type
// returned a JSON or YAML formatted byte array and an error
func FormatConvert(input []byte, o string) ([]byte, error) {
	switch o {
	case "json":
		var prettyJSON bytes.Buffer
		err := json.Indent(&prettyJSON, input, "", "  ")
		if err != nil {
			return nil, err
		}
		return prettyJSON.Bytes(), nil
	case "yaml":
		return yaml.JSONToYAML(input)
	}
	return nil, fmt.Errorf("unexpected -o output mode")
}
