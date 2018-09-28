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
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/ghodss/yaml"
	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/utils/exec"
)

var (
	diffLong = templates.LongDesc(i18n.T(`
		Diff configurations specified by filename or stdin between the current online
		configuration, and the configuration as it would be if applied.

		Output is always YAML.

		KUBERNETES_EXTERNAL_DIFF environment variable can be used to select your own
		diff command. By default, the "diff" command available in your path will be
		run with "-u" (unicode) and "-N" (treat new files as empty) options.`))
	diffExample = templates.Examples(i18n.T(`
		# Diff resources included in pod.json.
		kubectl diff -f pod.json

		# Diff file read from stdin
		cat service.yaml | kubectl diff -f -`))
)

type DiffOptions struct {
	FilenameOptions resource.FilenameOptions
}

func checkDiffArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}
	return nil
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
		Short:   i18n.T("Diff live version against would-be applied version"),
		Long:    diffLong,
		Example: diffExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(checkDiffArgs(cmd, args))
			cmdutil.CheckErr(RunDiff(f, &diff, &options))
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
func (p *Printer) Print(obj runtime.Object, w io.Writer) error {
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

func (v *DiffVersion) getObject(obj Object) (runtime.Object, error) {
	switch v.Name {
	case "LIVE":
		return obj.Live(), nil
	case "MERGED":
		return obj.Merged()
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
	Live() runtime.Object
	Merged() (runtime.Object, error)

	Name() string
}

// InfoObject is an implementation of the Object interface. It gets all
// the information from the Info object.
type InfoObject struct {
	LocalObj runtime.Object
	Info     *resource.Info
	Encoder  runtime.Encoder
	OpenAPI  openapi.Resources
}

var _ Object = &InfoObject{}

// Returns the live version of the object
func (obj InfoObject) Live() runtime.Object {
	return obj.Info.Object
}

// Returns the "merged" object, as it would look like if applied or
// created.
func (obj InfoObject) Merged() (runtime.Object, error) {
	// Build the patcher, and then apply the patch with dry-run, unless the object doesn't exist, in which case we need to create it.
	if obj.Live() == nil {
		// Dry-run create if the object doesn't exist.
		return resource.NewHelper(obj.Info.Client, obj.Info.Mapping).Create(
			obj.Info.Namespace,
			true,
			obj.LocalObj,
			&metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
		)
	}

	modified, err := kubectl.GetModifiedConfiguration(obj.LocalObj, false, unstructured.UnstructuredJSONScheme)
	if err != nil {
		return nil, err
	}

	// This is using the patcher from apply, to keep the same behavior.
	// We plan on replacing this with server-side apply when it becomes available.
	patcher := &patcher{
		mapping:       obj.Info.Mapping,
		helper:        resource.NewHelper(obj.Info.Client, obj.Info.Mapping),
		overwrite:     true,
		backOff:       clockwork.NewRealClock(),
		serverDryRun:  true,
		openapiSchema: obj.OpenAPI,
	}

	_, result, err := patcher.patch(obj.Info.Object, modified, obj.Info.Source, obj.Info.Namespace, obj.Info.Name, nil)
	return result, err
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

// RunDiff uses the factory to parse file arguments, find the version to
// diff, and find each Info object for each files, and runs against the
// differ.
func RunDiff(f cmdutil.Factory, diff *DiffProgram, options *DiffOptions) error {
	schema, err := f.OpenAPISchema()
	if err != nil {
		return err
	}

	differ, err := NewDiffer("LIVE", "MERGED")
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
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		local := info.Object.DeepCopyObject()
		if err := info.Get(); err != nil {
			if !errors.IsNotFound(err) {
				return err
			}
			info.Object = nil
		}

		obj := InfoObject{
			LocalObj: local,
			Info:     info,
			Encoder:  scheme.DefaultJSONEncoder(),
			OpenAPI:  schema,
		}

		return differ.Diff(obj, printer)
	})
	if err != nil {
		return err
	}

	// Error ignore on purpose. diff(1) for example, returns an error if there is any diff.
	_ = differ.Run(diff)

	return nil
}
