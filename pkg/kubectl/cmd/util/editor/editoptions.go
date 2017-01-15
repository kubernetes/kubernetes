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
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/crlf"
	"k8s.io/kubernetes/pkg/util/strategicpatch"

	"github.com/golang/glog"
)

// EditOptions contains all the options for running edit cli command.
type EditOptions struct {
	resource.FilenameOptions

	Output             string
	OutputVersion      string
	WindowsLineEndings bool

	cmdutil.ValidateOptions

	Mapper         meta.RESTMapper
	ResourceMapper *resource.Mapper
	Result         *resource.Result

	EditMode EditMode

	CmdNamespace    string
	ApplyAnnotation bool
	Record          bool
	Include3rdParty bool

	Out    io.Writer
	ErrOut io.Writer

	f                  cmdutil.Factory
	defaultVersion     schema.GroupVersion
	editPrinterOptions *editPrinterOptions
}

type editPrinterOptions struct {
	printer   kubectl.ResourcePrinter
	ext       string
	addHeader bool
}

// Complete completes all the required options
func (o *EditOptions) Complete(f cmdutil.Factory, out, errOut io.Writer, args []string) error {
	o.editPrinterOptions = getPrinter(o.Output)

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	var mapper meta.RESTMapper
	var typer runtime.ObjectTyper
	switch o.EditMode {
	case NormalEditMode:
		mapper, typer = f.Object()
	case EditBeforeCreateMode:
		mapper, typer, err = f.UnstructuredObject()
	default:
		return fmt.Errorf("Not supported edit mode %q", o.EditMode)
	}
	if err != nil {
		return err
	}
	resourceMapper := &resource.Mapper{
		ObjectTyper:  typer,
		RESTMapper:   mapper,
		ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),

		// NB: we use `f.Decoder(false)` to get a plain deserializer for
		// the resourceMapper, since it's used to read in edits and
		// we don't want to convert into the internal version when
		// reading in edits (this would cause us to potentially try to
		// compare two different GroupVersions).
		Decoder: f.Decoder(false),
	}
	var b *resource.Builder
	switch o.EditMode {
	case NormalEditMode:
		b = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
			ResourceTypeOrNameArgs(true, args...).
			Latest()
	case EditBeforeCreateMode:
		b = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme)
	default:
		return fmt.Errorf("Not supported edit mode %q", o.EditMode)
	}
	r := b.NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ContinueOnError().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	o.Result = r

	// determine default version
	clientConfig, err := f.ClientConfig()
	if err != nil {
		return err
	}
	defaultVersion, err := cmdutil.OutputVersionAlias(o.OutputVersion, clientConfig.GroupVersion)
	if err != nil {
		return err
	}
	o.defaultVersion = defaultVersion

	o.Mapper = mapper
	o.ResourceMapper = resourceMapper
	o.CmdNamespace = cmdNamespace

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
	encoder := o.f.JSONEncoder()
	normalEditInfos, err := o.Result.Infos()
	if err != nil {
		return err
	}

	edit := NewDefaultEditor(o.f.EditorEnvs())
	editFn := func(info *resource.Info, err error) error {
		var (
			results  = editResults{}
			original = []byte{}
			edited   = []byte{}
			file     string
		)

		containsError := false
		var infos []*resource.Info
		for {
			switch o.EditMode {
			case NormalEditMode:
				infos = normalEditInfos
			case EditBeforeCreateMode:
				infos = []*resource.Info{info}
			default:
				err = fmt.Errorf("Not supported edit mode %q", o.EditMode)
			}
			originalObj, err := resource.AsVersionedObject(infos, false, o.defaultVersion, encoder)
			if err != nil {
				return err
			}

			objToEdit := originalObj

			// generate the file to edit
			buf := &bytes.Buffer{}
			var w io.Writer = buf
			if o.WindowsLineEndings {
				w = crlf.NewCRLFWriter(w)
			}

			if o.editPrinterOptions.addHeader {
				results.header.writeTo(w)
			}

			if !containsError {
				if err := o.editPrinterOptions.printer.PrintObj(objToEdit, w); err != nil {
					return preservedFile(err, results.file, o.ErrOut)
				}
				original = buf.Bytes()
			} else {
				// In case of an error, preserve the edited file.
				// Remove the comments (header) from it since we already
				// have included the latest header in the buffer above.
				buf.Write(manualStrip(edited))
			}

			// launch the editor
			editedDiff := edited
			edited, file, err = edit.LaunchTempFile(fmt.Sprintf("%s-edit-", filepath.Base(os.Args[0])), o.editPrinterOptions.ext, buf)
			if err != nil {
				return preservedFile(err, results.file, o.ErrOut)
			}
			if o.EditMode == NormalEditMode || containsError {
				if bytes.Equal(stripComments(editedDiff), stripComments(edited)) {
					// Ugly hack right here. We will hit this either (1) when we try to
					// save the same changes we tried to save in the previous iteration
					// which means our changes are invalid or (2) when we exit the second
					// time. The second case is more usual so we can probably live with it.
					// TODO: A less hacky fix would be welcome :)
					return preservedFile(fmt.Errorf("%s", "Edit cancelled, no valid changes were saved."), file, o.ErrOut)
				}
			}

			// cleanup any file from the previous pass
			if len(results.file) > 0 {
				os.Remove(results.file)
			}
			glog.V(4).Infof("User edited:\n%s", string(edited))

			// Apply validation
			validator, err := o.f.Validator(o.EnableValidation, o.SchemaCacheDir)
			if err != nil {
				return preservedFile(err, file, o.ErrOut)
			}
			err = validator.ValidateBytes(stripComments(edited))
			if err != nil {
				results = editResults{
					file: file,
				}
				containsError = true
				fmt.Fprintln(o.Out, results.addError(errors.NewInvalid(api.Kind(""), "", field.ErrorList{field.Invalid(nil, "The edited file failed validation", fmt.Sprintf("%v", err))}), infos[0]))
				continue
			}

			// Compare content without comments
			if bytes.Equal(stripComments(original), stripComments(edited)) {
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
			updates, err := o.ResourceMapper.InfoForData(edited, "edited-file")
			if err != nil {
				// syntax error
				containsError = true
				results.header.reasons = append(results.header.reasons, editReason{head: fmt.Sprintf("The edited file had a syntax error: %v", err)})
				continue
			}
			// not a syntax error as it turns out...
			containsError = false

			namespaceVisitor := resource.NewFlattenListVisitor(updates, o.ResourceMapper)
			// need to make sure the original namespace wasn't changed while editing
			if err = namespaceVisitor.Visit(resource.RequireNamespace(o.CmdNamespace)); err != nil {
				return preservedFile(err, file, o.ErrOut)
			}

			// iterate through all items to apply annotations
			mutatedObjects, err := visitAnnotation(o.ApplyAnnotation, o.Record, o.f, updates, o.ResourceMapper, encoder)
			if err != nil {
				return preservedFile(err, file, o.ErrOut)
			}

			// if we mutated a list in the visitor, persist the changes on the overall object
			if meta.IsListType(updates.Object) {
				meta.SetList(updates.Object, mutatedObjects)
			}

			switch o.EditMode {
			case NormalEditMode:
				err = visitToPatch(originalObj, updates, o.Mapper, o.ResourceMapper, encoder, o.Out, o.defaultVersion, &results)
			case EditBeforeCreateMode:
				err = visitToCreate(updates, o.Mapper, o.ResourceMapper, o.Out, o.defaultVersion, &results)
			default:
				err = fmt.Errorf("Not supported edit mode %q", o.EditMode)
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
		return editFn(nil, nil)
	// If doing an edit before created, we don't want a list and instead want the normal behavior as kubectl create.
	case EditBeforeCreateMode:
		return o.Result.Visit(editFn)
	default:
		return fmt.Errorf("Not supported edit mode %q", o.EditMode)
	}
}

func getPrinter(format string) *editPrinterOptions {
	switch format {
	case "json":
		return &editPrinterOptions{
			printer:   &kubectl.JSONPrinter{},
			ext:       ".json",
			addHeader: false,
		}
	case "yaml":
		return &editPrinterOptions{
			printer:   &kubectl.YAMLPrinter{},
			ext:       ".yaml",
			addHeader: true,
		}
	default:
		// if format is not specified, use yaml as default
		return &editPrinterOptions{
			printer:   &kubectl.YAMLPrinter{},
			ext:       ".yaml",
			addHeader: true,
		}
	}
}

func visitToPatch(
	originalObj runtime.Object,
	updates *resource.Info,
	mapper meta.RESTMapper,
	resourceMapper *resource.Mapper,
	encoder runtime.Encoder,
	out io.Writer,
	defaultVersion schema.GroupVersion,
	results *editResults,
) error {

	patchVisitor := resource.NewFlattenListVisitor(updates, resourceMapper)
	err := patchVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		currOriginalObj := originalObj

		// if we're editing a list, then navigate the list to find the item that we're currently trying to edit
		if meta.IsListType(originalObj) {
			currOriginalObj = nil
			editObjUID, err := meta.NewAccessor().UID(info.Object)
			if err != nil {
				return err
			}

			listItems, err := meta.ExtractList(originalObj)
			if err != nil {
				return err
			}

			// iterate through the list to find the item with the matching UID
			for i := range listItems {
				originalObjUID, err := meta.NewAccessor().UID(listItems[i])
				if err != nil {
					return err
				}
				if editObjUID == originalObjUID {
					currOriginalObj = listItems[i]
					break
				}
			}
			if currOriginalObj == nil {
				return fmt.Errorf("no original object found for %#v", info.Object)
			}

		}

		originalSerialization, err := runtime.Encode(encoder, currOriginalObj)
		if err != nil {
			return err
		}
		editedSerialization, err := runtime.Encode(encoder, info.Object)
		if err != nil {
			return err
		}

		// compute the patch on a per-item basis
		// use strategic merge to create a patch
		originalJS, err := yaml.ToJSON(originalSerialization)
		if err != nil {
			return err
		}
		editedJS, err := yaml.ToJSON(editedSerialization)
		if err != nil {
			return err
		}

		if reflect.DeepEqual(originalJS, editedJS) {
			// no edit, so just skip it.
			cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, false, "skipped")
			return nil
		}

		preconditions := []strategicpatch.PreconditionFunc{strategicpatch.RequireKeyUnchanged("apiVersion"),
			strategicpatch.RequireKeyUnchanged("kind"), strategicpatch.RequireMetadataKeyUnchanged("name")}
		patch, err := strategicpatch.CreateTwoWayMergePatch(originalJS, editedJS, currOriginalObj, preconditions...)
		if err != nil {
			glog.V(4).Infof("Unable to calculate diff, no merge is possible: %v", err)
			if strategicpatch.IsPreconditionFailed(err) {
				return fmt.Errorf("%s", "At least one of apiVersion, kind and name was changed")
			}
			return err
		}

		results.version = defaultVersion
		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch)
		if err != nil {
			fmt.Fprintln(out, results.addError(err, info))
			return nil
		}
		info.Refresh(patched, true)
		cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, false, "edited")
		return nil
	})
	return err
}

func visitToCreate(updates *resource.Info, mapper meta.RESTMapper, resourceMapper *resource.Mapper, out io.Writer, defaultVersion schema.GroupVersion, results *editResults) error {
	createVisitor := resource.NewFlattenListVisitor(updates, resourceMapper)
	err := createVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		results.version = defaultVersion
		if err := resource.CreateAndRefresh(info); err != nil {
			return err
		}
		cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, false, "created")
		return nil
	})
	return err
}

func visitAnnotation(applyAnnotation, record bool, f cmdutil.Factory, updates *resource.Info, resourceMapper *resource.Mapper, encoder runtime.Encoder) ([]runtime.Object, error) {
	mutatedObjects := []runtime.Object{}
	annotationVisitor := resource.NewFlattenListVisitor(updates, resourceMapper)
	// iterate through all items to apply annotations
	err := annotationVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		// put configuration annotation in "updates"
		if err := kubectl.CreateOrUpdateAnnotation(applyAnnotation, info, encoder); err != nil {
			return err
		}
		if record {
			if err := cmdutil.RecordChangeCause(info.Object, f.Command()); err != nil {
				return err
			}
		}
		mutatedObjects = append(mutatedObjects, info.Object)

		return nil

	})
	return mutatedObjects, err
}

type EditMode string

const (
	NormalEditMode       EditMode = "normal_mode"
	EditBeforeCreateMode EditMode = "edit_before_create_mode"
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
func (h *editHeader) writeTo(w io.Writer) error {
	fmt.Fprint(w, `# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
`)
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
	case errors.IsInvalid(err):
		r.edit = append(r.edit, info)
		reason := editReason{
			head: fmt.Sprintf("%s %q was not valid", info.Mapping.Resource, info.Name),
		}
		if err, ok := err.(errors.APIStatus); ok {
			if details := err.Status().Details; details != nil {
				for _, cause := range details.Causes {
					reason.other = append(reason.other, fmt.Sprintf("%s: %s", cause.Field, cause.Message))
				}
			}
		}
		r.header.reasons = append(r.header.reasons, reason)
		return fmt.Sprintf("error: %s %q is invalid", info.Mapping.Resource, info.Name)
	case errors.IsNotFound(err):
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

// stripComments will transform a YAML file into JSON, thus dropping any comments
// in it. Note that if the given file has a syntax error, the transformation will
// fail and we will manually drop all comments from the file.
func stripComments(file []byte) []byte {
	stripped := file
	stripped, err := yaml.ToJSON(stripped)
	if err != nil {
		stripped = manualStrip(file)
	}
	return stripped
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

// manualStrip is used for dropping comments from a YAML file
func manualStrip(file []byte) []byte {
	stripped := []byte{}
	lines := bytes.Split(file, []byte("\n"))
	for i, line := range lines {
		if bytes.HasPrefix(bytes.TrimSpace(line), []byte("#")) {
			continue
		}
		stripped = append(stripped, line...)
		if i < len(lines)-1 {
			stripped = append(stripped, '\n')
		}
	}
	return stripped
}
