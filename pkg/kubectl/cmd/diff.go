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
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/apply/parse"
	"k8s.io/kubernetes/pkg/kubectl/apply/strategy"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	diffLong    = templates.LongDesc(i18n.T(`TODO`))
	diffExample = templates.Examples(i18n.T(`TODO`))
)

type DiffOptions struct {
	FilenameOptions resource.FilenameOptions
}

func NewCmdDiff(f cmdutil.Factory, stdout io.Writer) *cobra.Command {
	var options DiffOptions
	cmd := &cobra.Command{
		Use:     "diff -f FILENAME",
		Short:   i18n.T("Diff local configuration against live version"),
		Long:    diffLong,
		Example: diffExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunDiff(f, cmd, stdout, &options, args))
		},
	}

	usage := "that contains the configuration to diff"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.MarkFlagRequired("filename")

	return cmd
}

func getDiffProgram() []string {
	if envDiff := os.Getenv("KUBERNETES_DIFF"); envDiff != "" {
		return []string{envDiff}
	}
	return []string{"diff", "-u", "-N"}
}

func runDiffProgram(from, to string, stdout io.Writer) error {
	diff := getDiffProgram()
	diff = append(diff, from, to)

	cmd := exec.Command(diff[0], diff[1:]...)
	out, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	if _, err := io.Copy(stdout, out); err != nil {
		return err
	}

	cmd.Wait() // Ignore diff return code

	return nil
}

func RunDiff(f cmdutil.Factory, cmd *cobra.Command, stdout io.Writer, options *DiffOptions, args []string) error {
	fromArg, toArg, err := ParseDiffArguments(args)
	if err != nil {
		return err
	}

	resources, err := f.OpenAPISchema()
	if err != nil {
		return err
	}

	from, err := NewDiffVersion(fromArg, f.JSONEncoder(), resources)
	if err != nil {
		return err
	}
	defer from.Dir.Delete()
	to, err := NewDiffVersion(toArg, f.JSONEncoder(), resources)
	if err != nil {
		return err
	}
	defer to.Dir.Delete()

	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		Unstructured(f.UnstructuredClientForMapping, mapper, typer).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	printer := Printer{}
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := info.Get(); err != nil {
			if !errors.IsNotFound(err) {
				return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
			}
			info.Object = nil
		}

		if err := from.Print(info, f.JSONEncoder(), printer); err != nil {
			return err
		}
		if err := to.Print(info, f.JSONEncoder(), printer); err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		return err
	}

	if err := runDiffProgram(from.Dir.Name, to.Dir.Name, stdout); err != nil {
		return err
	}

	return nil
}

func ParseDiffArguments(args []string) (string, string, error) {
	if len(args) > 2 {
		return "", "", fmt.Errorf("Invalid number of arguments: expected at most 2.")
	}
	// Default values
	from := "LOCAL"
	to := "LIVE"
	if len(args) > 0 {
		from = args[0]
	}
	if len(args) > 1 {
		to = args[1]
	}

	return from, to, nil
}

type Printer struct{}

func (p *Printer) Print(obj map[string]interface{}, w io.Writer) error {
	if obj == nil {
		return nil
	}
	data, err := yaml.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

type DiffVersion struct {
	Dir  *Directory
	Name string

	parser  parse.Factory
	encoder runtime.Encoder
}

func NewDiffVersion(name string, encoder runtime.Encoder, openapi openapi.Resources) (*DiffVersion, error) {
	dir, err := CreateDirectory(name)
	if err != nil {
		return nil, err
	}
	return &DiffVersion{
		Dir:     dir,
		Name:    name,
		parser:  parse.Factory{openapi},
		encoder: encoder,
	}, nil
}

func (v *DiffVersion) getObject(obj Object) (map[string]interface{}, error) {
	switch v.Name {
	case "LIVE":
		return obj.Live(v.encoder)
	case "MERGED":
		return obj.Merged(v.encoder, v.parser)
	case "LOCAL":
		return obj.Local(v.encoder)
	case "LAST":
		return obj.Last()
	}
	return nil, fmt.Errorf("Unknown version: %v", v.Name)
}

func (v *DiffVersion) Print(info *resource.Info, encoder runtime.Encoder, printer Printer) error {
	obj, err := v.getObject(Object{Info: info})
	if err != nil {
		return err
	}
	f, err := v.Dir.NewFile(info.Name)
	if err != nil {
		return err
	}
	defer f.Close()
	return printer.Print(obj, f)
}

type Directory struct {
	Name string
}

func CreateDirectory(prefix string) (*Directory, error) {
	name, err := ioutil.TempDir("", prefix+"-")
	if err != nil {
		return nil, err
	}

	return &Directory{
		Name: name,
	}, nil
}

func (d *Directory) NewFile(name string) (*os.File, error) {
	return os.OpenFile(filepath.Join(d.Name, name), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0700)
}

func (d *Directory) Delete() error {
	return os.Remove(d.Name)
}

type Object struct {
	Info *resource.Info
}

func (obj Object) Local(encoder runtime.Encoder) (map[string]interface{}, error) {
	data, err := runtime.Encode(encoder, obj.Info.VersionedObject)
	if err != nil {
		return nil, err
	}
	return ToMap(data)
}

func (obj Object) Live(encoder runtime.Encoder) (map[string]interface{}, error) {
	if obj.Info.Object == nil {
		return nil, nil // Object doesn't exist on cluster.
	}
	data, err := runtime.Encode(encoder, obj.Info.Object)
	if err != nil {
		return nil, err
	}
	return ToMap(data)
}

func ToMap(data []byte) (map[string]interface{}, error) {
	m := map[string]interface{}{}
	if len(data) == 0 {
		return m, nil
	}
	err := json.Unmarshal(data, &m)
	return m, err
}

func (obj Object) Merged(encoder runtime.Encoder, parser parse.Factory) (map[string]interface{}, error) {
	local, err := obj.Local(encoder)
	if err != nil {
		return nil, err
	}

	live, err := obj.Live(encoder)
	if err != nil {
		return nil, err
	}

	last, err := obj.Last()
	if err != nil {
		return nil, err
	}

	if live == nil || last == nil {
		return local, nil // We probably don't have a live verison, merged is local.
	}

	elmt, err := parser.CreateElement(last, local, live)
	if err != nil {
		return nil, err
	}
	result, err := elmt.Merge(strategy.Create(strategy.Options{}))
	return result.MergedResult.(map[string]interface{}), err
}

func (obj Object) Last() (map[string]interface{}, error) {
	if obj.Info.Object == nil {
		return nil, nil // No object is live, return empty
	}
	accessor, err := meta.Accessor(obj.Info.Object)
	if err != nil {
		return nil, err
	}
	annots := accessor.GetAnnotations()
	if annots == nil {
		return nil, nil // Not an error, just empty.
	}

	return ToMap([]byte(annots[api.LastAppliedConfigAnnotation]))
}
