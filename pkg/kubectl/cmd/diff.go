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
	"path/filepath"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	"k8s.io/client-go/dynamic"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubectl/apply/parse"
	"k8s.io/kubernetes/pkg/kubectl/apply/strategy"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/utils/exec"
)

var (
	diffLong = templates.LongDesc(i18n.T(`
		Diff configurations specified by filename or stdin between their local,
		last-applied, live and/or "merged" versions.

		LOCAL and LIVE versions are diffed by default. Other available keywords
		are MERGED and LAST.

		Output is always YAML.

		KUBERNETES_EXTERNAL_DIFF environment variable can be used to select your own
		diff command. By default, the "diff" command available in your path will be
		run with "-u" (unicode) and "-N" (treat new files as empty) options.`))
	diffExample = templates.Examples(i18n.T(`
		# Diff resources included in pod.json. By default, it will diff LOCAL and LIVE versions
		kubectl alpha diff -f pod.json

		# When one version is specified, diff that version against LIVE
		cat service.yaml | kubectl alpha diff -f - MERGED

		# Or specify both versions
		kubectl alpha diff -f pod.json -f service.yaml LAST LOCAL`))
)

type DiffOptions struct {
	FilenameOptions resource.FilenameOptions
}

func isValidArgument(arg string) error {
	switch arg {
	case "LOCAL", "LIVE", "LAST", "MERGED":
		return nil
	default:
		return fmt.Errorf(`Invalid parameter %q, must be either "LOCAL", "LIVE", "LAST" or "MERGED"`, arg)
	}

}

func parseDiffArguments(args []string) (string, string, error) {
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

	if err := isValidArgument(to); err != nil {
		return "", "", err
	}
	if err := isValidArgument(from); err != nil {
		return "", "", err
	}

	return from, to, nil
}

func NewCmdDiff(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	var options DiffOptions
	diff := DiffProgram{
		Exec:      exec.New(),
		IOStreams: streams,
	}
	cmd := &cobra.Command{
		Use: "diff -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Diff different versions of configurations"),
		Long:    diffLong,
		Example: diffExample,
		Run: func(cmd *cobra.Command, args []string) {
			from, to, err := parseDiffArguments(args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(RunDiff(f, &diff, &options, from, to))
		},
	}

	usage := "contains the configuration to diff"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.MarkFlagRequired("filename")

	return cmd
}

// DiffProgram finds and run the diff program. The value of
// KUBERNETES_EXTERNAL_DIFF environment variable will be used a diff
// program. By default, `diff(1)` will be used.
type DiffProgram struct {
	Exec exec.Interface
	genericclioptions.IOStreams
}

func (d *DiffProgram) getCommand(args ...string) exec.Cmd {
	diff := ""
	if envDiff := os.Getenv("KUBERNETES_EXTERNAL_DIFF"); envDiff != "" {
		diff = envDiff
	} else {
		diff = "diff"
		args = append([]string{"-u", "-N"}, args...)
	}

	cmd := d.Exec.Command(diff, args...)
	cmd.SetStdout(d.Out)
	cmd.SetStderr(d.ErrOut)

	return cmd
}

// Run runs the detected diff program. `from` and `to` are the directory to diff.
func (d *DiffProgram) Run(from, to string) error {
	d.getCommand(from, to).Run() // Ignore diff return code
	return nil
}

// Printer is used to print an object.
type Printer struct{}

// Print the object inside the writer w.
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

// DiffVersion gets the proper version of objects, and aggregate them into a directory.
type DiffVersion struct {
	Dir  *Directory
	Name string
}

// NewDiffVersion creates a new DiffVersion with the named version.
func NewDiffVersion(name string) (*DiffVersion, error) {
	dir, err := CreateDirectory(name)
	if err != nil {
		return nil, err
	}
	return &DiffVersion{
		Dir:  dir,
		Name: name,
	}, nil
}

func (v *DiffVersion) getObject(obj Object) (map[string]interface{}, error) {
	switch v.Name {
	case "LIVE":
		return obj.Live()
	case "MERGED":
		return obj.Merged()
	case "LOCAL":
		return obj.Local()
	case "LAST":
		return obj.Last()
	}
	return nil, fmt.Errorf("Unknown version: %v", v.Name)
}

// Print prints the object using the printer into a new file in the directory.
func (v *DiffVersion) Print(obj Object, printer Printer) error {
	vobj, err := v.getObject(obj)
	if err != nil {
		return err
	}
	f, err := v.Dir.NewFile(obj.Name())
	if err != nil {
		return err
	}
	defer f.Close()
	return printer.Print(vobj, f)
}

// Directory creates a new temp directory, and allows to easily create new files.
type Directory struct {
	Name string
}

// CreateDirectory does create the actual disk directory, and return a
// new representation of it.
func CreateDirectory(prefix string) (*Directory, error) {
	name, err := ioutil.TempDir("", prefix+"-")
	if err != nil {
		return nil, err
	}

	return &Directory{
		Name: name,
	}, nil
}

// NewFile creates a new file in the directory.
func (d *Directory) NewFile(name string) (*os.File, error) {
	return os.OpenFile(filepath.Join(d.Name, name), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0700)
}

// Delete removes the directory recursively.
func (d *Directory) Delete() error {
	return os.RemoveAll(d.Name)
}

// Object is an interface that let's you retrieve multiple version of
// it.
type Object interface {
	Local() (map[string]interface{}, error)
	Live() (map[string]interface{}, error)
	Last() (map[string]interface{}, error)
	Merged() (map[string]interface{}, error)

	Name() string
}

// InfoObject is an implementation of the Object interface. It gets all
// the information from the Info object.
type InfoObject struct {
	Remote  *unstructured.Unstructured
	Info    *resource.Info
	Encoder runtime.Encoder
	Parser  *parse.Factory
}

var _ Object = &InfoObject{}

func (obj InfoObject) toMap(data []byte) (map[string]interface{}, error) {
	m := map[string]interface{}{}
	if len(data) == 0 {
		return m, nil
	}
	err := json.Unmarshal(data, &m)
	return m, err
}

func (obj InfoObject) Local() (map[string]interface{}, error) {
	data, err := runtime.Encode(obj.Encoder, obj.Info.Object)
	if err != nil {
		return nil, err
	}
	return obj.toMap(data)
}

func (obj InfoObject) Live() (map[string]interface{}, error) {
	if obj.Remote == nil {
		return nil, nil // Object doesn't exist on cluster.
	}
	return obj.Remote.UnstructuredContent(), nil
}

func (obj InfoObject) Merged() (map[string]interface{}, error) {
	local, err := obj.Local()
	if err != nil {
		return nil, err
	}

	live, err := obj.Live()
	if err != nil {
		return nil, err
	}

	last, err := obj.Last()
	if err != nil {
		return nil, err
	}

	if live == nil || last == nil {
		return local, nil // We probably don't have a live version, merged is local.
	}

	elmt, err := obj.Parser.CreateElement(last, local, live)
	if err != nil {
		return nil, err
	}
	result, err := elmt.Merge(strategy.Create(strategy.Options{}))
	return result.MergedResult.(map[string]interface{}), err
}

func (obj InfoObject) Last() (map[string]interface{}, error) {
	if obj.Remote == nil {
		return nil, nil // No object is live, return empty
	}
	accessor, err := meta.Accessor(obj.Remote)
	if err != nil {
		return nil, err
	}
	annots := accessor.GetAnnotations()
	if annots == nil {
		return nil, nil // Not an error, just empty.
	}

	return obj.toMap([]byte(annots[api.LastAppliedConfigAnnotation]))
}

func (obj InfoObject) Name() string {
	return obj.Info.Name
}

// Differ creates two DiffVersion and diffs them.
type Differ struct {
	From *DiffVersion
	To   *DiffVersion
}

func NewDiffer(from, to string) (*Differ, error) {
	differ := Differ{}
	var err error
	differ.From, err = NewDiffVersion(from)
	if err != nil {
		return nil, err
	}
	differ.To, err = NewDiffVersion(to)
	if err != nil {
		differ.From.Dir.Delete()
		return nil, err
	}

	return &differ, nil
}

// Diff diffs to versions of a specific object, and print both versions to directories.
func (d *Differ) Diff(obj Object, printer Printer) error {
	if err := d.From.Print(obj, printer); err != nil {
		return err
	}
	if err := d.To.Print(obj, printer); err != nil {
		return err
	}
	return nil
}

// Run runs the diff program against both directories.
func (d *Differ) Run(diff *DiffProgram) error {
	return diff.Run(d.From.Dir.Name, d.To.Dir.Name)
}

// TearDown removes both temporary directories recursively.
func (d *Differ) TearDown() {
	d.From.Dir.Delete() // Ignore error
	d.To.Dir.Delete()   // Ignore error
}

type Downloader struct {
	mapper  meta.RESTMapper
	dclient dynamic.Interface
	ns      string
}

func NewDownloader(f cmdutil.Factory) (*Downloader, error) {
	var err error
	var d Downloader

	d.mapper, err = f.ToRESTMapper()
	if err != nil {
		return nil, err
	}
	d.dclient, err = f.DynamicClient()
	if err != nil {
		return nil, err
	}
	d.ns, _, _ = f.ToRawKubeConfigLoader().Namespace()

	return &d, nil
}

func (d *Downloader) Download(info *resource.Info) (*unstructured.Unstructured, error) {
	gvk := info.Object.GetObjectKind().GroupVersionKind()
	mapping, err := d.mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return nil, err
	}

	var resource dynamic.ResourceInterface
	switch mapping.Scope.Name() {
	case meta.RESTScopeNameNamespace:
		if info.Namespace == "" {
			info.Namespace = d.ns
		}
		resource = d.dclient.Resource(mapping.Resource).Namespace(info.Namespace)
	case meta.RESTScopeNameRoot:
		resource = d.dclient.Resource(mapping.Resource)
	}

	return resource.Get(info.Name, metav1.GetOptions{})
}

// RunDiff uses the factory to parse file arguments, find the version to
// diff, and find each Info object for each files, and runs against the
// differ.
func RunDiff(f cmdutil.Factory, diff *DiffProgram, options *DiffOptions, from, to string) error {
	openapi, err := f.OpenAPISchema()
	if err != nil {
		return err
	}
	parser := &parse.Factory{Resources: openapi}

	differ, err := NewDiffer(from, to)
	if err != nil {
		return err
	}
	defer differ.TearDown()

	printer := Printer{}

	cmdNamespace, enforceNamespace, err := f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		Unstructured().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		Local().
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	dl, err := NewDownloader(f)
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		remote, _ := dl.Download(info)
		obj := InfoObject{
			Remote:  remote,
			Info:    info,
			Parser:  parser,
			Encoder: cmdutil.InternalVersionJSONEncoder(),
		}

		return differ.Diff(obj, printer)
	})
	if err != nil {
		return err
	}

	differ.Run(diff)

	return nil
}
