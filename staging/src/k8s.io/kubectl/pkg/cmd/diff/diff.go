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
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/klog"
	"k8s.io/kubectl/pkg/cmd/apply"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/util/templates"
	"k8s.io/utils/exec"
	"sigs.k8s.io/yaml"
)

var (
	diffLong = templates.LongDesc(i18n.T(`
		Diff configurations specified by filename or stdin between the current online
		configuration, and the configuration as it would be if applied.

		Output is always YAML.

		KUBECTL_EXTERNAL_DIFF environment variable can be used to select your own
		diff command. By default, the "diff" command available in your path will be
		run with "-u" (unified diff) and "-N" (treat absent files as empty) options.`))
	diffExample = templates.Examples(i18n.T(`
		# Diff resources included in pod.json.
		kubectl diff -f pod.json

		# Diff file read from stdin
		cat service.yaml | kubectl diff -f -`))
)

// Number of times we try to diff before giving-up
const maxRetries = 4

type DiffOptions struct {
	FilenameOptions resource.FilenameOptions

	ServerSideApply bool
	ForceConflicts  bool

	OpenAPISchema    openapi.Resources
	DiscoveryClient  discovery.DiscoveryInterface
	DynamicClient    dynamic.Interface
	DryRunVerifier   *resource.DryRunVerifier
	CmdNamespace     string
	EnforceNamespace bool
	Builder          *resource.Builder
	Diff             *DiffProgram
}

func validateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}
	return nil
}

func NewDiffOptions(ioStreams genericclioptions.IOStreams) *DiffOptions {
	return &DiffOptions{
		Diff: &DiffProgram{
			Exec:      exec.New(),
			IOStreams: ioStreams,
		},
	}
}

func NewCmdDiff(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	options := NewDiffOptions(streams)
	cmd := &cobra.Command{
		Use:                   "diff -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Diff live version against would-be applied version"),
		Long:                  diffLong,
		Example:               diffExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd))
			cmdutil.CheckErr(validateArgs(cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	usage := "contains the configuration to diff"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddServerSideApplyFlags(cmd)

	return cmd
}

// DiffProgram finds and run the diff program. The value of
// KUBECTL_EXTERNAL_DIFF environment variable will be used a diff
// program. By default, `diff(1)` will be used.
type DiffProgram struct {
	Exec exec.Interface
	genericclioptions.IOStreams
}

func (d *DiffProgram) getCommand(args ...string) (string, exec.Cmd) {
	diff := ""
	if envDiff := os.Getenv("KUBECTL_EXTERNAL_DIFF"); envDiff != "" {
		diff = envDiff
	} else {
		diff = "diff"
		args = append([]string{"-u", "-N"}, args...)
	}

	cmd := d.Exec.Command(diff, args...)
	cmd.SetStdout(d.Out)
	cmd.SetStderr(d.ErrOut)

	return diff, cmd
}

// Run runs the detected diff program. `from` and `to` are the directory to diff.
func (d *DiffProgram) Run(from, to string) error {
	diff, cmd := d.getCommand(from, to)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to run %q: %v", diff, err)
	}
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
	LocalObj        runtime.Object
	Info            *resource.Info
	Encoder         runtime.Encoder
	OpenAPI         openapi.Resources
	Force           bool
	ServerSideApply bool
	ForceConflicts  bool
	genericclioptions.IOStreams
}

var _ Object = &InfoObject{}

// Returns the live version of the object
func (obj InfoObject) Live() runtime.Object {
	return obj.Info.Object
}

// Returns the "merged" object, as it would look like if applied or
// created.
func (obj InfoObject) Merged() (runtime.Object, error) {
	if obj.ServerSideApply {
		data, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj.LocalObj)
		if err != nil {
			return nil, err
		}
		options := metav1.PatchOptions{
			Force:  &obj.ForceConflicts,
			DryRun: []string{metav1.DryRunAll},
		}
		return resource.NewHelper(obj.Info.Client, obj.Info.Mapping).Patch(
			obj.Info.Namespace,
			obj.Info.Name,
			types.ApplyPatchType,
			data,
			&options,
		)
	}

	// Build the patcher, and then apply the patch with dry-run, unless the object doesn't exist, in which case we need to create it.
	if obj.Live() == nil {
		// Dry-run create if the object doesn't exist.
		return resource.NewHelper(obj.Info.Client, obj.Info.Mapping).CreateWithOptions(
			obj.Info.Namespace,
			true,
			obj.LocalObj,
			&metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
		)
	}

	var resourceVersion *string
	if !obj.Force {
		accessor, err := meta.Accessor(obj.Info.Object)
		if err != nil {
			return nil, err
		}
		str := accessor.GetResourceVersion()
		resourceVersion = &str
	}

	modified, err := util.GetModifiedConfiguration(obj.LocalObj, false, unstructured.UnstructuredJSONScheme)
	if err != nil {
		return nil, err
	}

	// This is using the patcher from apply, to keep the same behavior.
	// We plan on replacing this with server-side apply when it becomes available.
	patcher := &apply.Patcher{
		Mapping:         obj.Info.Mapping,
		Helper:          resource.NewHelper(obj.Info.Client, obj.Info.Mapping),
		Overwrite:       true,
		BackOff:         clockwork.NewRealClock(),
		ServerDryRun:    true,
		OpenapiSchema:   obj.OpenAPI,
		ResourceVersion: resourceVersion,
	}

	_, result, err := patcher.Patch(obj.Info.Object, modified, obj.Info.Source, obj.Info.Namespace, obj.Info.Name, obj.ErrOut)
	return result, err
}

func (obj InfoObject) Name() string {
	group := ""
	if obj.Info.Mapping.GroupVersionKind.Group != "" {
		group = fmt.Sprintf("%v.", obj.Info.Mapping.GroupVersionKind.Group)
	}
	return group + fmt.Sprintf(
		"%v.%v.%v.%v",
		obj.Info.Mapping.GroupVersionKind.Version,
		obj.Info.Mapping.GroupVersionKind.Kind,
		obj.Info.Namespace,
		obj.Info.Name,
	)
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

func isConflict(err error) bool {
	return err != nil && errors.IsConflict(err)
}

func (o *DiffOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error

	err = o.FilenameOptions.RequireFilenameOrKustomize()
	if err != nil {
		return err
	}

	o.ServerSideApply = cmdutil.GetServerSideApplyFlag(cmd)
	o.ForceConflicts = cmdutil.GetForceConflictsFlag(cmd)
	if o.ForceConflicts && !o.ServerSideApply {
		return fmt.Errorf("--force-conflicts only works with --server-side")
	}

	if !o.ServerSideApply {
		o.OpenAPISchema, err = f.OpenAPISchema()
		if err != nil {
			return err
		}
	}

	o.DiscoveryClient, err = f.ToDiscoveryClient()
	if err != nil {
		return err
	}

	o.DynamicClient, err = f.DynamicClient()
	if err != nil {
		return err
	}

	o.DryRunVerifier = resource.NewDryRunVerifier(o.DynamicClient, o.DiscoveryClient)

	o.CmdNamespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.Builder = f.NewBuilder()
	return nil
}

// RunDiff uses the factory to parse file arguments, find the version to
// diff, and find each Info object for each files, and runs against the
// differ.
func (o *DiffOptions) Run() error {
	differ, err := NewDiffer("LIVE", "MERGED")
	if err != nil {
		return err
	}
	defer differ.TearDown()

	printer := Printer{}

	r := o.Builder.
		Unstructured().
		NamespaceParam(o.CmdNamespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := o.DryRunVerifier.HasSupport(info.Mapping.GroupVersionKind); err != nil {
			return err
		}

		local := info.Object.DeepCopyObject()
		for i := 1; i <= maxRetries; i++ {
			if err = info.Get(); err != nil {
				if !errors.IsNotFound(err) {
					return err
				}
				info.Object = nil
			}

			force := i == maxRetries
			if force {
				klog.Warningf(
					"Object (%v: %v) keeps changing, diffing without lock",
					info.Object.GetObjectKind().GroupVersionKind(),
					info.Name,
				)
			}
			obj := InfoObject{
				LocalObj:        local,
				Info:            info,
				Encoder:         scheme.DefaultJSONEncoder(),
				OpenAPI:         o.OpenAPISchema,
				Force:           force,
				ServerSideApply: o.ServerSideApply,
				ForceConflicts:  o.ForceConflicts,
				IOStreams:       o.Diff.IOStreams,
			}

			err = differ.Diff(obj, printer)
			if !isConflict(err) {
				break
			}
		}
		return err
	})
	if err != nil {
		return err
	}

	return differ.Run(o.Diff)
}
