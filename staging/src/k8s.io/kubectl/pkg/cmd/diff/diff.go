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
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/openapi3"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/cmd/apply"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/util/prune"
	"k8s.io/kubectl/pkg/util/templates"
	"k8s.io/utils/exec"
	"sigs.k8s.io/yaml"
)

var (
	diffLong = templates.LongDesc(i18n.T(`
		Diff configurations specified by file name or stdin between the current online
		configuration, and the configuration as it would be if applied.

		The output is always YAML.

		KUBECTL_EXTERNAL_DIFF environment variable can be used to select your own
		diff command. Users can use external commands with params too, example:
		KUBECTL_EXTERNAL_DIFF="colordiff -N -u"

		By default, the "diff" command available in your path will be
		run with the "-u" (unified diff) and "-N" (treat absent files as empty) options.

		Exit status:
		 0
		No differences were found.
		 1
		Differences were found.
		 >1
		Kubectl or diff failed with an error.

		Note: KUBECTL_EXTERNAL_DIFF, if used, is expected to follow that convention.`))

	diffExample = templates.Examples(i18n.T(`
		# Diff resources included in pod.json
		kubectl diff -f pod.json

		# Diff file read from stdin
		cat service.yaml | kubectl diff -f -`))
)

// Number of times we try to diff before giving-up
const maxRetries = 4

// Constants for masking sensitive values
const (
	sensitiveMaskDefault = "***"
	sensitiveMaskBefore  = "*** (before)"
	sensitiveMaskAfter   = "*** (after)"
)

// diffError returns the ExitError if the status code is less than 1,
// nil otherwise.
func diffError(err error) exec.ExitError {
	if err, ok := err.(exec.ExitError); ok && err.ExitStatus() <= 1 {
		return err
	}
	return nil
}

type DiffOptions struct {
	FilenameOptions resource.FilenameOptions

	ServerSideApply   bool
	FieldManager      string
	ForceConflicts    bool
	ShowManagedFields bool

	Concurrency      int
	Selector         string
	OpenAPIGetter    openapi.OpenAPIResourcesGetter
	OpenAPIV3Root    openapi3.Root
	DynamicClient    dynamic.Interface
	CmdNamespace     string
	EnforceNamespace bool
	Builder          *resource.Builder
	Diff             *DiffProgram

	pruner  *pruner
	tracker *tracker
}

func NewDiffOptions(ioStreams genericiooptions.IOStreams) *DiffOptions {
	return &DiffOptions{
		Diff: &DiffProgram{
			Exec:      exec.New(),
			IOStreams: ioStreams,
		},
	}
}

func NewCmdDiff(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	options := NewDiffOptions(streams)
	cmd := &cobra.Command{
		Use:                   "diff -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Diff the live version against a would-be applied version"),
		Long:                  diffLong,
		Example:               diffExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckDiffErr(options.Complete(f, cmd, args))
			cmdutil.CheckDiffErr(options.Validate())
			// `kubectl diff` propagates the error code from
			// diff or `KUBECTL_EXTERNAL_DIFF`. Also, we
			// don't want to print an error if diff returns
			// error code 1, which simply means that changes
			// were found. We also don't want kubectl to
			// return 1 if there was a problem.
			if err := options.Run(); err != nil {
				if exitErr := diffError(err); exitErr != nil {
					cmdutil.CheckErr(cmdutil.ErrExit)
				}
				cmdutil.CheckDiffErr(err)
			}
		},
	}

	// Flag errors exit with code 1, however according to the diff
	// command it means changes were found.
	// Thus, it should return status code greater than 1.
	cmd.SetFlagErrorFunc(func(command *cobra.Command, err error) error {
		cmdutil.CheckDiffErr(cmdutil.UsageErrorf(cmd, err.Error()))
		return nil
	})

	usage := "contains the configuration to diff"
	cmd.Flags().StringArray("prune-allowlist", []string{}, "Overwrite the default allowlist with <group/version/kind> for --prune")
	cmd.Flags().Bool("prune", false, "Include resources that would be deleted by pruning. Can be used with -l and default shows all resources would be pruned")
	cmd.Flags().BoolVar(&options.ShowManagedFields, "show-managed-fields", options.ShowManagedFields, "If true, include managed fields in the diff.")
	cmd.Flags().IntVar(&options.Concurrency, "concurrency", 1, "Number of objects to process in parallel when diffing against the live version. Larger number = faster, but more memory, I/O and CPU over that shorter period of time.")
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddServerSideApplyFlags(cmd)
	cmdutil.AddFieldManagerFlagVar(cmd, &options.FieldManager, apply.FieldManagerClientSideApply)
	cmdutil.AddLabelSelectorFlagVar(cmd, &options.Selector)

	return cmd
}

// DiffProgram finds and run the diff program. The value of
// KUBECTL_EXTERNAL_DIFF environment variable will be used a diff
// program. By default, `diff(1)` will be used.
type DiffProgram struct {
	Exec exec.Interface
	genericiooptions.IOStreams
}

func (d *DiffProgram) getCommand(args ...string) (string, exec.Cmd) {
	diff := ""
	if envDiff := os.Getenv("KUBECTL_EXTERNAL_DIFF"); envDiff != "" {
		diffCommand := strings.Split(envDiff, " ")
		diff = diffCommand[0]

		if len(diffCommand) > 1 {
			// Regex accepts: Alphanumeric (case-insensitive), dash and equal
			isValidChar := regexp.MustCompile(`^[a-zA-Z0-9-=]+$`).MatchString
			for i := 1; i < len(diffCommand); i++ {
				if isValidChar(diffCommand[i]) {
					args = append(args, diffCommand[i])
				}
			}
		}
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
		// Let's not wrap diff errors, or we won't be able to
		// differentiate them later.
		if diffErr := diffError(err); diffErr != nil {
			return diffErr
		}
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
func (v *DiffVersion) Print(name string, obj runtime.Object, printer Printer) error {
	f, err := v.Dir.NewFile(name)
	if err != nil {
		return err
	}
	defer f.Close()
	return printer.Print(obj, f)
}

// Directory creates a new temp directory, and allows to easily create new files.
type Directory struct {
	Name string
}

// CreateDirectory does create the actual disk directory, and return a
// new representation of it.
func CreateDirectory(prefix string) (*Directory, error) {
	name, err := os.MkdirTemp("", prefix+"-")
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
	OpenAPIGetter   openapi.OpenAPIResourcesGetter
	OpenAPIV3Root   openapi3.Root
	Force           bool
	ServerSideApply bool
	FieldManager    string
	ForceConflicts  bool
	genericiooptions.IOStreams
}

var _ Object = &InfoObject{}

// Returns the live version of the object
func (obj InfoObject) Live() runtime.Object {
	return obj.Info.Object
}

// Returns the "merged" object, as it would look like if applied or
// created.
func (obj InfoObject) Merged() (runtime.Object, error) {
	helper := resource.NewHelper(obj.Info.Client, obj.Info.Mapping).
		DryRun(true).
		WithFieldManager(obj.FieldManager)
	if obj.ServerSideApply {
		data, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj.LocalObj)
		if err != nil {
			return nil, err
		}
		options := metav1.PatchOptions{
			Force:        &obj.ForceConflicts,
			FieldManager: obj.FieldManager,
		}
		return helper.Patch(
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
		return helper.CreateWithOptions(
			obj.Info.Namespace,
			true,
			obj.LocalObj,
			&metav1.CreateOptions{},
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
		Helper:          helper,
		Overwrite:       true,
		BackOff:         clockwork.NewRealClock(),
		OpenAPIGetter:   obj.OpenAPIGetter,
		OpenAPIV3Root:   obj.OpenAPIV3Root,
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

// toUnstructured converts a runtime.Object into an unstructured.Unstructured object.
func toUnstructured(obj runtime.Object) (*unstructured.Unstructured, error) {
	if obj == nil {
		return nil, nil
	}
	c, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj.DeepCopyObject())
	if err != nil {
		return nil, fmt.Errorf("convert to unstructured: %w", err)
	}
	u := &unstructured.Unstructured{}
	u.SetUnstructuredContent(c)
	return u, nil
}

// Masker masks sensitive values in an object while preserving diff-able
// changes.
//
// All sensitive values in the object will be masked with a fixed-length
// asterisk mask. If two values are different, an additional suffix will
// be added so they can be diff-ed.
type Masker struct {
	from *unstructured.Unstructured
	to   *unstructured.Unstructured
}

func NewMasker(from, to runtime.Object) (*Masker, error) {
	// Convert objects to unstructured
	f, err := toUnstructured(from)
	if err != nil {
		return nil, fmt.Errorf("convert to unstructured: %w", err)
	}
	t, err := toUnstructured(to)
	if err != nil {
		return nil, fmt.Errorf("convert to unstructured: %w", err)
	}

	// Run masker
	m := &Masker{
		from: f,
		to:   t,
	}
	if err := m.run(); err != nil {
		return nil, fmt.Errorf("run masker: %w", err)
	}
	return m, nil
}

// dataFromUnstructured returns the underlying nested map in the data key.
func (m Masker) dataFromUnstructured(u *unstructured.Unstructured) (map[string]interface{}, error) {
	if u == nil {
		return nil, nil
	}
	data, found, err := unstructured.NestedMap(u.UnstructuredContent(), "data")
	if err != nil {
		return nil, fmt.Errorf("get nested map: %w", err)
	}
	if !found {
		return nil, nil
	}
	return data, nil
}

// run compares and patches sensitive values.
func (m *Masker) run() error {
	// Extract nested map object
	from, err := m.dataFromUnstructured(m.from)
	if err != nil {
		return fmt.Errorf("extract 'data' field: %w", err)
	}
	to, err := m.dataFromUnstructured(m.to)
	if err != nil {
		return fmt.Errorf("extract 'data' field: %w", err)
	}

	for k := range from {
		// Add before/after suffix when key exists on both
		// objects and are not equal, so that it will be
		// visible in diffs.
		if _, ok := to[k]; ok {
			if from[k] != to[k] {
				from[k] = sensitiveMaskBefore
				to[k] = sensitiveMaskAfter
				continue
			}
			to[k] = sensitiveMaskDefault
		}
		from[k] = sensitiveMaskDefault
	}
	for k := range to {
		// Mask remaining keys that were not in 'from'
		if _, ok := from[k]; !ok {
			to[k] = sensitiveMaskDefault
		}
	}

	// Patch objects with masked data
	if m.from != nil && from != nil {
		if err := unstructured.SetNestedMap(m.from.UnstructuredContent(), from, "data"); err != nil {
			return fmt.Errorf("patch masked data: %w", err)
		}
	}
	if m.to != nil && to != nil {
		if err := unstructured.SetNestedMap(m.to.UnstructuredContent(), to, "data"); err != nil {
			return fmt.Errorf("patch masked data: %w", err)
		}
	}
	return nil
}

// From returns the masked version of the 'from' object.
func (m *Masker) From() runtime.Object {
	return m.from
}

// To returns the masked version of the 'to' object.
func (m *Masker) To() runtime.Object {
	return m.to
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
func (d *Differ) Diff(obj Object, printer Printer, showManagedFields bool) error {
	from, err := d.From.getObject(obj)
	if err != nil {
		return err
	}
	to, err := d.To.getObject(obj)
	if err != nil {
		return err
	}

	if !showManagedFields {
		from = omitManagedFields(from)
		to = omitManagedFields(to)
	}

	// Mask secret values if object is V1Secret
	if gvk := to.GetObjectKind().GroupVersionKind(); gvk.Version == "v1" && gvk.Kind == "Secret" {
		m, err := NewMasker(from, to)
		if err != nil {
			return err
		}
		from, to = m.From(), m.To()
	}

	if err := d.From.Print(obj.Name(), from, printer); err != nil {
		return err
	}
	if err := d.To.Print(obj.Name(), to, printer); err != nil {
		return err
	}
	return nil
}

func omitManagedFields(o runtime.Object) runtime.Object {
	a, err := meta.Accessor(o)
	if err != nil {
		// The object is not a `metav1.Object`, ignore it.
		return o
	}
	a.SetManagedFields(nil)
	return o
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

func (o *DiffOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}

	var err error

	err = o.FilenameOptions.RequireFilenameOrKustomize()
	if err != nil {
		return err
	}

	o.ServerSideApply = cmdutil.GetServerSideApplyFlag(cmd)
	o.FieldManager = apply.GetApplyFieldManagerFlag(cmd, o.ServerSideApply)
	o.ForceConflicts = cmdutil.GetForceConflictsFlag(cmd)
	if o.ForceConflicts && !o.ServerSideApply {
		return fmt.Errorf("--force-conflicts only works with --server-side")
	}

	if !o.ServerSideApply {
		o.OpenAPIGetter = f
		if !cmdutil.OpenAPIV3Patch.IsDisabled() {
			openAPIV3Client, err := f.OpenAPIV3Client()
			if err == nil {
				o.OpenAPIV3Root = openapi3.NewRoot(openAPIV3Client)
			} else {
				klog.V(4).Infof("warning: OpenAPI V3 Patch is enabled but is unable to be loaded. Will fall back to OpenAPI V2")
			}
		}
	}

	o.DynamicClient, err = f.DynamicClient()
	if err != nil {
		return err
	}

	o.CmdNamespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	if cmdutil.GetFlagBool(cmd, "prune") {
		mapper, err := f.ToRESTMapper()
		if err != nil {
			return err
		}

		resources, err := prune.ParseResources(mapper, cmdutil.GetFlagStringArray(cmd, "prune-allowlist"))
		if err != nil {
			return err
		}
		o.tracker = newTracker()
		o.pruner = newPruner(o.DynamicClient, mapper, resources, o.Selector)
	}

	o.Builder = f.NewBuilder()
	return nil
}

// Run uses the factory to parse file arguments, find the version to
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
		VisitorConcurrency(o.Concurrency).
		NamespaceParam(o.CmdNamespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		LabelSelectorParam(o.Selector).
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
				OpenAPIGetter:   o.OpenAPIGetter,
				OpenAPIV3Root:   o.OpenAPIV3Root,
				Force:           force,
				ServerSideApply: o.ServerSideApply,
				FieldManager:    o.FieldManager,
				ForceConflicts:  o.ForceConflicts,
				IOStreams:       o.Diff.IOStreams,
			}

			if o.tracker != nil {
				o.tracker.MarkVisited(info)
			}

			err = differ.Diff(obj, printer, o.ShowManagedFields)
			if !isConflict(err) {
				break
			}
		}

		apply.WarnIfDeleting(info.Object, o.Diff.ErrOut)

		return err
	})

	if o.pruner != nil {
		prunedObjs, err := o.pruner.pruneAll(o.tracker, o.CmdNamespace != "")
		if err != nil {
			klog.Warningf("pruning failed and could not be evaluated err: %v", err)
		}

		// Print pruned objects into old file and thus, diff
		// command will show them as pruned.
		for _, p := range prunedObjs {
			name, err := getObjectName(p)
			if err != nil {
				klog.Warningf("pruning failed and object name could not be retrieved: %v", err)
				continue
			}
			if err := differ.From.Print(name, p, printer); err != nil {
				return err
			}
		}
	}

	if err != nil {
		return err
	}

	return differ.Run(o.Diff)
}

// Validate makes sure provided values for DiffOptions are valid
func (o *DiffOptions) Validate() error {
	return nil
}

func getObjectName(obj runtime.Object) (string, error) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	metadata, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	name := metadata.GetName()
	ns := metadata.GetNamespace()

	group := ""
	if gvk.Group != "" {
		group = fmt.Sprintf("%v.", gvk.Group)
	}
	return group + fmt.Sprintf(
		"%v.%v.%v.%v",
		gvk.Version,
		gvk.Kind,
		ns,
		name,
	), nil
}
