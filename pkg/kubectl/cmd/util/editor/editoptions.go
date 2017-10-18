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

package editor

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/evanphx/json-patch"
	"github.com/golang/glog"
	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/crlf"
	"k8s.io/kubernetes/pkg/printers"
)

// EditOptions contains all the options for running edit cli command.
type EditOptions struct {
	resource.FilenameOptions

	Output             string
	OutputPatch        bool
	WindowsLineEndings bool

	cmdutil.ValidateOptions

	Mapper         meta.RESTMapper
	ResourceMapper *resource.Mapper
	OriginalResult *resource.Result
	Encoder        runtime.Encoder

	EditMode EditMode

	CmdNamespace    string
	ApplyAnnotation bool
	Record          bool
	ChangeCause     string
	Include3rdParty bool

	Out    io.Writer
	ErrOut io.Writer

	f                   cmdutil.Factory
	editPrinterOptions  *editPrinterOptions
	updatedResultGetter func(data []byte) *resource.Result
}

type editPrinterOptions struct {
	printer   printers.ResourcePrinter
	ext       string
	addHeader bool
}

// Complete completes all the required options
func (o *EditOptions) Complete(f cmdutil.Factory, out, errOut io.Writer, args []string, cmd *cobra.Command) error {
	if o.EditMode != NormalEditMode && o.EditMode != EditBeforeCreateMode && o.EditMode != ApplyEditMode {
		return fmt.Errorf("unsupported edit mode %q", o.EditMode)
	}
	if o.Output != "" {
		if o.Output != "yaml" && o.Output != "json" {
			return fmt.Errorf("invalid output format %s, only yaml|json supported", o.Output)
		}
	}
	o.editPrinterOptions = getPrinter(o.Output)

	if o.OutputPatch && o.EditMode != NormalEditMode {
		return fmt.Errorf("the edit mode doesn't support output the patch")
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}

	b := f.NewBuilder().Unstructured(f.UnstructuredClientForMapping, mapper, typer)
	if o.EditMode == NormalEditMode || o.EditMode == ApplyEditMode {
		// when do normal edit or apply edit we need to always retrieve the latest resource from server
		b = b.ResourceTypeOrNameArgs(true, args...).Latest()
	}
	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	r := b.NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		ContinueOnError().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	o.OriginalResult = r

	o.updatedResultGetter = func(data []byte) *resource.Result {
		// resource builder to read objects from edited data
		return resource.NewBuilder(mapper, f.CategoryExpander(), typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme).
			Stream(bytes.NewReader(data), "edited-file").
			IncludeUninitialized(includeUninitialized).
			ContinueOnError().
			Flatten().
			Do()
	}

	o.Mapper = mapper
	o.CmdNamespace = cmdNamespace
	o.Encoder = f.JSONEncoder()
	o.f = f

	// Set up writer
	o.Out = out
	o.ErrOut = errOut

	return nil
}

// Validate checks the EditOptions to see if there is sufficient information to run the command.
func (o *EditOptions) Validate() error {
	return nil
}

func (o *EditOptions) Run() error {
	edit := NewDefaultEditor(o.f.EditorEnvs())
	// editFn is invoked for each edit session (once with a list for normal edit, once for each individual resource in a edit-on-create invocation)
	editFn := func(infos []*resource.Info) error {
		var (
			results  = editResults{}
			original = []byte{}
			edited   = []byte{}
			file     string
			err      error
		)

		containsError := false
		// loop until we succeed or cancel editing
		for {
			// get the object we're going to serialize as input to the editor
			var originalObj runtime.Object
			switch len(infos) {
			case 1:
				originalObj = infos[0].Object
			default:
				l := &unstructured.UnstructuredList{
					Object: map[string]interface{}{
						"kind":       "List",
						"apiVersion": "v1",
						"metadata":   map[string]interface{}{},
					},
				}
				for _, info := range infos {
					l.Items = append(l.Items, *info.Object.(*unstructured.Unstructured))
				}
				originalObj = l
			}

			// generate the file to edit
			buf := &bytes.Buffer{}
			var w io.Writer = buf
			if o.WindowsLineEndings {
				w = crlf.NewCRLFWriter(w)
			}

			if o.editPrinterOptions.addHeader {
				results.header.writeTo(w, o.EditMode)
			}

			if !containsError {
				if err := o.editPrinterOptions.printer.PrintObj(originalObj, w); err != nil {
					return preservedFile(err, results.file, o.ErrOut)
				}
				original = buf.Bytes()
			} else {
				// In case of an error, preserve the edited file.
				// Remove the comments (header) from it since we already
				// have included the latest header in the buffer above.
				buf.Write(cmdutil.ManualStrip(edited))
			}

			// launch the editor
			editedDiff := edited
			edited, file, err = edit.LaunchTempFile(fmt.Sprintf("%s-edit-", filepath.Base(os.Args[0])), o.editPrinterOptions.ext, buf)
			if err != nil {
				return preservedFile(err, results.file, o.ErrOut)
			}
			// If we're retrying the loop because of an error, and no change was made in the file, short-circuit
			if containsError && bytes.Equal(cmdutil.StripComments(editedDiff), cmdutil.StripComments(edited)) {
				return preservedFile(fmt.Errorf("%s", "Edit cancelled, no valid changes were saved."), file, o.ErrOut)
			}
			// cleanup any file from the previous pass
			if len(results.file) > 0 {
				os.Remove(results.file)
			}
			glog.V(4).Infof("User edited:\n%s", string(edited))

			// Apply validation
			schema, err := o.f.Validator(o.EnableValidation)
			if err != nil {
				return preservedFile(err, file, o.ErrOut)
			}
			err = schema.ValidateBytes(cmdutil.StripComments(edited))
			if err != nil {
				results = editResults{
					file: file,
				}
				containsError = true
				fmt.Fprintln(o.ErrOut, results.addError(apierrors.NewInvalid(api.Kind(""), "", field.ErrorList{field.Invalid(nil, "The edited file failed validation", fmt.Sprintf("%v", err))}), infos[0]))
				continue
			}

			// Compare content without comments
			if bytes.Equal(cmdutil.StripComments(original), cmdutil.StripComments(edited)) {
				os.Remove(file)
				fmt.Fprintln(o.ErrOut, "Edit cancelled, no changes made.")
				return nil
			}

			lines, err := hasLines(bytes.NewBuffer(edited))
			if err != nil {
				return preservedFile(err, file, o.ErrOut)
			}
			if !lines {
				os.Remove(file)
				fmt.Fprintln(o.ErrOut, "Edit cancelled, saved file was empty.")
				return nil
			}

			results = editResults{
				file: file,
			}

			// parse the edited file
			updatedInfos, err := o.updatedResultGetter(edited).Infos()
			if err != nil {
				// syntax error
				containsError = true
				results.header.reasons = append(results.header.reasons, editReason{head: fmt.Sprintf("The edited file had a syntax error: %v", err)})
				continue
			}
			// not a syntax error as it turns out...
			containsError = false
			updatedVisitor := resource.InfoListVisitor(updatedInfos)

			// need to make sure the original namespace wasn't changed while editing
			if err := updatedVisitor.Visit(resource.RequireNamespace(o.CmdNamespace)); err != nil {
				return preservedFile(err, file, o.ErrOut)
			}

			// iterate through all items to apply annotations
			if err := o.visitAnnotation(updatedVisitor); err != nil {
				return preservedFile(err, file, o.ErrOut)
			}

			switch o.EditMode {
			case NormalEditMode:
				err = o.visitToPatch(infos, updatedVisitor, &results)
			case ApplyEditMode:
				err = o.visitToApplyEditPatch(infos, updatedVisitor)
			case EditBeforeCreateMode:
				err = o.visitToCreate(updatedVisitor)
			default:
				err = fmt.Errorf("unsupported edit mode %q", o.EditMode)
			}
			if err != nil {
				return preservedFile(err, results.file, o.ErrOut)
			}

			// Handle all possible errors
			//
			// 1. retryable: propose kubectl replace -f
			// 2. notfound: indicate the location of the saved configuration of the deleted resource
			// 3. invalid: retry those on the spot by looping ie. reloading the editor
			if results.retryable > 0 {
				fmt.Fprintf(o.ErrOut, "You can run `%s replace -f %s` to try this update again.\n", filepath.Base(os.Args[0]), file)
				return cmdutil.ErrExit
			}
			if results.notfound > 0 {
				fmt.Fprintf(o.ErrOut, "The edits you made on deleted resources have been saved to %q\n", file)
				return cmdutil.ErrExit
			}

			if len(results.edit) == 0 {
				if results.notfound == 0 {
					os.Remove(file)
				} else {
					fmt.Fprintf(o.Out, "The edits you made on deleted resources have been saved to %q\n", file)
				}
				return nil
			}

			if len(results.header.reasons) > 0 {
				containsError = true
			}
		}
	}

	switch o.EditMode {
	// If doing normal edit we cannot use Visit because we need to edit a list for convenience. Ref: #20519
	case NormalEditMode:
		infos, err := o.OriginalResult.Infos()
		if err != nil {
			return err
		}
		if len(infos) == 0 {
			return errors.New("edit cancelled, no objects found.")
		}
		return editFn(infos)
	case ApplyEditMode:
		infos, err := o.OriginalResult.Infos()
		if err != nil {
			return err
		}
		var annotationInfos []*resource.Info
		for i := range infos {
			data, err := kubectl.GetOriginalConfiguration(infos[i].Mapping, infos[i].Object)
			if err != nil {
				return err
			}
			if data == nil {
				continue
			}

			tempInfos, err := o.updatedResultGetter(data).Infos()
			if err != nil {
				return err
			}
			annotationInfos = append(annotationInfos, tempInfos[0])
		}
		if len(annotationInfos) == 0 {
			return errors.New("no last-applied-configuration annotation found on resources, to create the annotation, use command `kubectl apply set-last-applied --create-annotation`")
		}
		return editFn(annotationInfos)
	// If doing an edit before created, we don't want a list and instead want the normal behavior as kubectl create.
	case EditBeforeCreateMode:
		return o.OriginalResult.Visit(func(info *resource.Info, err error) error {
			return editFn([]*resource.Info{info})
		})
	default:
		return fmt.Errorf("unsupported edit mode %q", o.EditMode)
	}
}

func (o *EditOptions) visitToApplyEditPatch(originalInfos []*resource.Info, patchVisitor resource.Visitor) error {
	err := patchVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		editObjUID, err := meta.NewAccessor().UID(info.Object)
		if err != nil {
			return err
		}

		var originalInfo *resource.Info
		for _, i := range originalInfos {
			originalObjUID, err := meta.NewAccessor().UID(i.Object)
			if err != nil {
				return err
			}
			if editObjUID == originalObjUID {
				originalInfo = i
				break
			}
		}
		if originalInfo == nil {
			return fmt.Errorf("no original object found for %#v", info.Object)
		}

		originalJS, err := encodeToJson(o.Encoder, originalInfo.Object)
		if err != nil {
			return err
		}

		editedJS, err := encodeToJson(o.Encoder, info.Object)
		if err != nil {
			return err
		}

		if reflect.DeepEqual(originalJS, editedJS) {
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "skipped")
			return nil
		} else {
			err := o.annotationPatch(info)
			if err != nil {
				return err
			}
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "edited")
			return nil
		}
	})
	return err
}

func (o *EditOptions) annotationPatch(update *resource.Info) error {
	patch, _, patchType, err := GetApplyPatch(update.Object, o.Encoder)
	if err != nil {
		return err
	}
	mapping := update.ResourceMapping()
	client, err := o.f.UnstructuredClientForMapping(mapping)
	if err != nil {
		return err
	}
	helper := resource.NewHelper(client, mapping)
	_, err = helper.Patch(o.CmdNamespace, update.Name, patchType, patch)
	if err != nil {
		return err
	}
	return nil
}

func GetApplyPatch(obj runtime.Object, codec runtime.Encoder) ([]byte, []byte, types.PatchType, error) {
	beforeJSON, err := encodeToJson(codec, obj)
	if err != nil {
		return nil, []byte(""), types.MergePatchType, err
	}
	objCopy := obj.DeepCopyObject()
	accessor := meta.NewAccessor()
	annotations, err := accessor.Annotations(objCopy)
	if err != nil {
		return nil, beforeJSON, types.MergePatchType, err
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[api.LastAppliedConfigAnnotation] = string(beforeJSON)
	accessor.SetAnnotations(objCopy, annotations)
	afterJSON, err := encodeToJson(codec, objCopy)
	if err != nil {
		return nil, beforeJSON, types.MergePatchType, err
	}
	patch, err := jsonpatch.CreateMergePatch(beforeJSON, afterJSON)
	return patch, beforeJSON, types.MergePatchType, err
}

func encodeToJson(codec runtime.Encoder, obj runtime.Object) ([]byte, error) {
	serialization, err := runtime.Encode(codec, obj)
	if err != nil {
		return nil, err
	}
	js, err := yaml.ToJSON(serialization)
	if err != nil {
		return nil, err
	}
	return js, nil
}

func getPrinter(format string) *editPrinterOptions {
	switch format {
	case "json":
		return &editPrinterOptions{
			printer:   &printers.JSONPrinter{},
			ext:       ".json",
			addHeader: false,
		}
	case "yaml":
		return &editPrinterOptions{
			printer:   &printers.YAMLPrinter{},
			ext:       ".yaml",
			addHeader: true,
		}
	default:
		// if format is not specified, use yaml as default
		return &editPrinterOptions{
			printer:   &printers.YAMLPrinter{},
			ext:       ".yaml",
			addHeader: true,
		}
	}
}

func (o *EditOptions) visitToPatch(originalInfos []*resource.Info, patchVisitor resource.Visitor, results *editResults) error {
	err := patchVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		editObjUID, err := meta.NewAccessor().UID(info.Object)
		if err != nil {
			return err
		}

		var originalInfo *resource.Info
		for _, i := range originalInfos {
			originalObjUID, err := meta.NewAccessor().UID(i.Object)
			if err != nil {
				return err
			}
			if editObjUID == originalObjUID {
				originalInfo = i
				break
			}
		}
		if originalInfo == nil {
			return fmt.Errorf("no original object found for %#v", info.Object)
		}

		originalJS, err := encodeToJson(o.Encoder, originalInfo.Object)
		if err != nil {
			return err
		}

		editedJS, err := encodeToJson(o.Encoder, info.Object)
		if err != nil {
			return err
		}

		if reflect.DeepEqual(originalJS, editedJS) {
			// no edit, so just skip it.
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "skipped")
			return nil
		}

		preconditions := []mergepatch.PreconditionFunc{
			mergepatch.RequireKeyUnchanged("apiVersion"),
			mergepatch.RequireKeyUnchanged("kind"),
			mergepatch.RequireMetadataKeyUnchanged("name"),
		}

		// Create the versioned struct from the type defined in the mapping
		// (which is the API version we'll be submitting the patch to)
		versionedObject, err := legacyscheme.Scheme.New(info.Mapping.GroupVersionKind)
		var patchType types.PatchType
		var patch []byte
		switch {
		case runtime.IsNotRegisteredError(err):
			// fall back to generic JSON merge patch
			patchType = types.MergePatchType
			patch, err = jsonpatch.CreateMergePatch(originalJS, editedJS)
			if err != nil {
				glog.V(4).Infof("Unable to calculate diff, no merge is possible: %v", err)
				return err
			}
			for _, precondition := range preconditions {
				if !precondition(patch) {
					glog.V(4).Infof("Unable to calculate diff, no merge is possible: %v", err)
					return fmt.Errorf("%s", "At least one of apiVersion, kind and name was changed")
				}
			}
		case err != nil:
			return err
		default:
			patchType = types.StrategicMergePatchType
			patch, err = strategicpatch.CreateTwoWayMergePatch(originalJS, editedJS, versionedObject, preconditions...)
			if err != nil {
				glog.V(4).Infof("Unable to calculate diff, no merge is possible: %v", err)
				if mergepatch.IsPreconditionFailed(err) {
					return fmt.Errorf("%s", "At least one of apiVersion, kind and name was changed")
				}
				return err
			}
		}

		if o.OutputPatch {
			fmt.Fprintf(o.Out, "Patch: %s\n", string(patch))
		}

		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, patchType, patch)
		if err != nil {
			fmt.Fprintln(o.ErrOut, results.addError(err, info))
			return nil
		}
		info.Refresh(patched, true)
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "edited")
		return nil
	})
	return err
}

func (o *EditOptions) visitToCreate(createVisitor resource.Visitor) error {
	err := createVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		if err := resource.CreateAndRefresh(info); err != nil {
			return err
		}
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "created")
		return nil
	})
	return err
}

func (o *EditOptions) visitAnnotation(annotationVisitor resource.Visitor) error {
	// iterate through all items to apply annotations
	err := annotationVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		// put configuration annotation in "updates"
		if o.ApplyAnnotation {
			if err := kubectl.CreateOrUpdateAnnotation(true, info, o.Encoder); err != nil {
				return err
			}
		}
		if o.Record || cmdutil.ContainsChangeCause(info) {
			if err := cmdutil.RecordChangeCause(info.Object, o.ChangeCause); err != nil {
				return err
			}
		}

		return nil

	})
	return err
}

type EditMode string

const (
	NormalEditMode       EditMode = "normal_mode"
	EditBeforeCreateMode EditMode = "edit_before_create_mode"
	ApplyEditMode        EditMode = "edit_last_applied_mode"
)

// editReason preserves a message about the reason this file must be edited again
type editReason struct {
	head  string
	other []string
}

// editHeader includes a list of reasons the edit must be retried
type editHeader struct {
	reasons []editReason
}

// writeTo outputs the current header information into a stream
func (h *editHeader) writeTo(w io.Writer, editMode EditMode) error {
	if editMode == ApplyEditMode {
		fmt.Fprint(w, `# Please edit the 'last-applied-configuration' annotations below.
# Lines beginning with a '#' will be ignored, and an empty file will abort the edit.
#
`)
	} else {
		fmt.Fprint(w, `# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
`)
	}

	for _, r := range h.reasons {
		if len(r.other) > 0 {
			fmt.Fprintf(w, "# %s:\n", r.head)
		} else {
			fmt.Fprintf(w, "# %s\n", r.head)
		}
		for _, o := range r.other {
			fmt.Fprintf(w, "# * %s\n", o)
		}
		fmt.Fprintln(w, "#")
	}
	return nil
}

func (h *editHeader) flush() {
	h.reasons = []editReason{}
}

// editResults capture the result of an update
type editResults struct {
	header    editHeader
	retryable int
	notfound  int
	edit      []*resource.Info
	file      string

	version schema.GroupVersion
}

func (r *editResults) addError(err error, info *resource.Info) string {
	switch {
	case apierrors.IsInvalid(err):
		r.edit = append(r.edit, info)
		reason := editReason{
			head: fmt.Sprintf("%s %q was not valid", info.Mapping.Resource, info.Name),
		}
		if err, ok := err.(apierrors.APIStatus); ok {
			if details := err.Status().Details; details != nil {
				for _, cause := range details.Causes {
					reason.other = append(reason.other, fmt.Sprintf("%s: %s", cause.Field, cause.Message))
				}
			}
		}
		r.header.reasons = append(r.header.reasons, reason)
		return fmt.Sprintf("error: %s %q is invalid", info.Mapping.Resource, info.Name)
	case apierrors.IsNotFound(err):
		r.notfound++
		return fmt.Sprintf("error: %s %q could not be found on the server", info.Mapping.Resource, info.Name)
	default:
		r.retryable++
		return fmt.Sprintf("error: %s %q could not be patched: %v", info.Mapping.Resource, info.Name, err)
	}
}

// preservedFile writes out a message about the provided file if it exists to the
// provided output stream when an error happens. Used to notify the user where
// their updates were preserved.
func preservedFile(err error, path string, out io.Writer) error {
	if len(path) > 0 {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			fmt.Fprintf(out, "A copy of your changes has been stored to %q\n", path)
		}
	}
	return err
}

// hasLines returns true if any line in the provided stream is non empty - has non-whitespace
// characters, or the first non-whitespace character is a '#' indicating a comment. Returns
// any errors encountered reading the stream.
func hasLines(r io.Reader) (bool, error) {
	// TODO: if any files we read have > 64KB lines, we'll need to switch to bytes.ReadLine
	// TODO: probably going to be secrets
	s := bufio.NewScanner(r)
	for s.Scan() {
		if line := strings.TrimSpace(s.Text()); len(line) > 0 && line[0] != '#' {
			return true, nil
		}
	}
	if err := s.Err(); err != nil && err != io.EOF {
		return false, err
	}
	return false, nil
}
