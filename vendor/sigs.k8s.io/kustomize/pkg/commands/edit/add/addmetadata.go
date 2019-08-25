/*
Copyright 2018 The Kubernetes Authors.

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

package add

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/commands/kustfile"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/pgmconfig"
	"sigs.k8s.io/kustomize/pkg/types"
)

// kindOfAdd is the kind of metadata being added: label or annotation
type kindOfAdd int

const (
	annotation kindOfAdd = iota
	label
)

func (k kindOfAdd) String() string {
	kinds := [...]string{
		"annotation",
		"label",
	}
	if k < 0 || k > 1 {
		return "Unknown metadatakind"
	}
	return kinds[k]
}

type addMetadataOptions struct {
	force        bool
	metadata     map[string]string
	mapValidator func(map[string]string) error
	kind         kindOfAdd
}

// newCmdAddAnnotation adds one or more commonAnnotations to the kustomization file.
func newCmdAddAnnotation(fSys fs.FileSystem, v func(map[string]string) error) *cobra.Command {
	var o addMetadataOptions
	o.kind = annotation
	o.mapValidator = v
	cmd := &cobra.Command{
		Use:   "annotation",
		Short: "Adds one or more commonAnnotations to " + pgmconfig.KustomizationFileNames[0],
		Example: `
		add annotation {annotationKey1:annotationValue1},{annotationKey2:annotationValue2}`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.runE(args, fSys, o.addAnnotations)
		},
	}
	cmd.Flags().BoolVarP(&o.force, "force", "f", false,
		"overwrite commonAnnotation if it already exists",
	)
	return cmd
}

// newCmdAddLabel adds one or more commonLabels to the kustomization file.
func newCmdAddLabel(fSys fs.FileSystem, v func(map[string]string) error) *cobra.Command {
	var o addMetadataOptions
	o.kind = label
	o.mapValidator = v
	cmd := &cobra.Command{
		Use:   "label",
		Short: "Adds one or more commonLabels to " + pgmconfig.KustomizationFileNames[0],
		Example: `
		add label {labelKey1:labelValue1},{labelKey2:labelValue2}`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.runE(args, fSys, o.addLabels)
		},
	}
	cmd.Flags().BoolVarP(&o.force, "force", "f", false,
		"overwrite commonLabel if it already exists",
	)
	return cmd
}

func (o *addMetadataOptions) runE(
	args []string, fSys fs.FileSystem, adder func(*types.Kustomization) error) error {
	err := o.validateAndParse(args)
	if err != nil {
		return err
	}
	kf, err := kustfile.NewKustomizationFile(fSys)
	if err != nil {
		return err
	}
	m, err := kf.Read()
	if err != nil {
		return err
	}
	err = adder(m)
	if err != nil {
		return err
	}
	return kf.Write(m)
}

// validateAndParse validates `add` commands and parses them into o.metadata
func (o *addMetadataOptions) validateAndParse(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("must specify %s", o.kind)
	}
	if len(args) > 1 {
		return fmt.Errorf("%ss must be comma-separated, with no spaces", o.kind)
	}
	m, err := o.convertToMap(args[0])
	if err != nil {
		return err
	}
	if err = o.mapValidator(m); err != nil {
		return err
	}
	o.metadata = m
	return nil
}

func (o *addMetadataOptions) convertToMap(arg string) (map[string]string, error) {
	result := make(map[string]string)
	inputs := strings.Split(arg, ",")
	for _, input := range inputs {
		c := strings.Index(input, ":")
		if c == 0 {
			// key is not passed
			return nil, o.makeError(input, "need k:v pair where v may be quoted")
		} else if c < 0 {
			// only key passed
			result[input] = ""
		} else {
			// both key and value passed
			key := input[:c]
			value := trimQuotes(input[c+1:])
			result[key] = value
		}
	}
	return result, nil
}

func (o *addMetadataOptions) addAnnotations(m *types.Kustomization) error {
	if m.CommonAnnotations == nil {
		m.CommonAnnotations = make(map[string]string)
	}
	return o.writeToMap(m.CommonAnnotations, annotation)
}

func (o *addMetadataOptions) addLabels(m *types.Kustomization) error {
	if m.CommonLabels == nil {
		m.CommonLabels = make(map[string]string)
	}
	return o.writeToMap(m.CommonLabels, label)
}

func (o *addMetadataOptions) writeToMap(m map[string]string, kind kindOfAdd) error {
	for k, v := range o.metadata {
		if _, ok := m[k]; ok && !o.force {
			return fmt.Errorf("%s %s already in kustomization file", kind, k)
		}
		m[k] = v
	}
	return nil
}

func (o *addMetadataOptions) makeError(input string, message string) error {
	return fmt.Errorf("invalid %s: '%s' (%s)", o.kind, input, message)
}

func trimQuotes(s string) string {
	if len(s) >= 2 {
		if s[0] == '"' && s[len(s)-1] == '"' {
			return s[1 : len(s)-1]
		}
	}
	return s
}
