/*
Copyright 2015 The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	gruntime "runtime"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/crlf"
	"k8s.io/kubernetes/pkg/util/strategicpatch"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

var (
	editLong = templates.LongDesc(`
		Edit a resource from the default editor.

		The edit command allows you to directly edit any API resource you can retrieve via the
		command line tools. It will open the editor defined by your KUBE_EDITOR, or EDITOR
		environment variables, or fall back to 'vi' for Linux or 'notepad' for Windows.
		You can edit multiple objects, although changes are applied one at a time. The command
		accepts filenames as well as command line arguments, although the files you point to must
		be previously saved versions of resources.

		The files to edit will be output in the default API version, or a version specified
		by --output-version. The default format is YAML - if you would like to edit in JSON
		pass -o json. The flag --windows-line-endings can be used to force Windows line endings,
		otherwise the default for your operating system will be used.

		In the event an error occurs while updating, a temporary file will be created on disk
		that contains your unapplied changes. The most common error when updating a resource
		is another editor changing the resource on the server. When this occurs, you will have
		to apply your changes to the newer version of the resource, or update your temporary
		saved copy to include the latest resource version.`)

	editExample = templates.Examples(`
		# Edit the service named 'docker-registry':
		kubectl edit svc/docker-registry

		# Use an alternative editor
		KUBE_EDITOR="nano" kubectl edit svc/docker-registry

		# Edit the service 'docker-registry' in JSON using the v1 API format:
		kubectl edit svc/docker-registry --output-version=v1 -o json`)
)

func NewCmdEdit(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "edit (RESOURCE/NAME | -f FILENAME)",
		Short:   "Edit a resource on the server",
		Long:    editLong,
		Example: fmt.Sprintf(editExample),
		Run: func(cmd *cobra.Command, args []string) {
			err := RunEdit(f, out, errOut, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	usage := "to use to edit the resource"
	cmdutil.AddFilenameOptionFlags(cmd, options, usage)
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringP("output", "o", "yaml", "Output format. One of: yaml|json.")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given group version (for ex: 'extensions/v1beta1').")
	cmd.Flags().Bool("windows-line-endings", gruntime.GOOS == "windows", "Use Windows line-endings (default Unix line-endings)")
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func RunEdit(f cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, args []string, options *resource.FilenameOptions) error {
	return runEdit(f, out, errOut, cmd, args, options, NormalEditMode)
}

func runEdit(f cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, args []string, options *resource.FilenameOptions, editMode EditMode) error {
	o, err := getPrinter(cmd)
	if err != nil {
		return err
	}

	mapper, resourceMapper, r, cmdNamespace, err := getMapperAndResult(f, args, options, editMode)
	if err != nil {
		return err
	}

	clientConfig, err := f.ClientConfig()
	if err != nil {
		return err
	}

	encoder := f.JSONEncoder()
	defaultVersion, err := cmdutil.OutputVersion(cmd, clientConfig.GroupVersion)
	if err != nil {
		return err
	}

	normalEditInfos, err := r.Infos()
	if err != nil {
		return err
	}

	var (
		windowsLineEndings = cmdutil.GetFlagBool(cmd, "windows-line-endings")
		edit               = editor.NewDefaultEditor(f.EditorEnvs())
	)

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
			switch editMode {
			case NormalEditMode:
				infos = normalEditInfos
			case EditBeforeCreateMode:
				infos = []*resource.Info{info}
			default:
				err = fmt.Errorf("Not supported edit mode %q", editMode)
			}
			originalObj, err := resource.AsVersionedObject(infos, false, defaultVersion, encoder)
			if err != nil {
				return err
			}

			objToEdit := originalObj

			// generate the file to edit
			buf := &bytes.Buffer{}
			var w io.Writer = buf
			if windowsLineEndings {
				w = crlf.NewCRLFWriter(w)
			}

			if o.addHeader {
				results.header.writeTo(w)
			}

			if !containsError {
				if err := o.printer.PrintObj(objToEdit, w); err != nil {
					return preservedFile(err, results.file, errOut)
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
			edited, file, err = edit.LaunchTempFile(fmt.Sprintf("%s-edit-", filepath.Base(os.Args[0])), o.ext, buf)
			if err != nil {
				return preservedFile(err, results.file, errOut)
			}
			if editMode == NormalEditMode || containsError {
				if bytes.Equal(stripComments(editedDiff), stripComments(edited)) {
					// Ugly hack right here. We will hit this either (1) when we try to
					// save the same changes we tried to save in the previous iteration
					// which means our changes are invalid or (2) when we exit the second
					// time. The second case is more usual so we can probably live with it.
					// TODO: A less hacky fix would be welcome :)
					return preservedFile(fmt.Errorf("%s", "Edit cancelled, no valid changes were saved."), file, errOut)
				}
			}

			// cleanup any file from the previous pass
			if len(results.file) > 0 {
				os.Remove(results.file)
			}
			glog.V(4).Infof("User edited:\n%s", string(edited))

			// Apply validation
			schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
			if err != nil {
				return preservedFile(err, file, errOut)
			}
			err = schema.ValidateBytes(stripComments(edited))
			if err != nil {
				results = editResults{
					file: file,
				}
				containsError = true
				fmt.Fprintln(out, results.addError(errors.NewInvalid(api.Kind(""), "", field.ErrorList{field.Invalid(nil, "The edited file failed validation", fmt.Sprintf("%v", err))}), infos[0]))
				continue
			}

			// Compare content without comments
			if bytes.Equal(stripComments(original), stripComments(edited)) {
				os.Remove(file)
				fmt.Fprintln(errOut, "Edit cancelled, no changes made.")
				return nil
			}

			lines, err := hasLines(bytes.NewBuffer(edited))
			if err != nil {
				return preservedFile(err, file, errOut)
			}
			if !lines {
				os.Remove(file)
				fmt.Fprintln(errOut, "Edit cancelled, saved file was empty.")
				return nil
			}

			results = editResults{
				file: file,
			}

			// parse the edited file
			updates, err := resourceMapper.InfoForData(edited, "edited-file")
			if err != nil {
				// syntax error
				containsError = true
				results.header.reasons = append(results.header.reasons, editReason{head: fmt.Sprintf("The edited file had a syntax error: %v", err)})
				continue
			}
			// not a syntax error as it turns out...
			containsError = false

			namespaceVisitor := resource.NewFlattenListVisitor(updates, resourceMapper)
			// need to make sure the original namespace wasn't changed while editing
			if err = namespaceVisitor.Visit(resource.RequireNamespace(cmdNamespace)); err != nil {
				return preservedFile(err, file, errOut)
			}

			// iterate through all items to apply annotations
			mutatedObjects, err := visitAnnotation(cmd, f, updates, resourceMapper, encoder)
			if err != nil {
				return preservedFile(err, file, errOut)
			}

			// if we mutated a list in the visitor, persist the changes on the overall object
			if meta.IsListType(updates.Object) {
				meta.SetList(updates.Object, mutatedObjects)
			}

			switch editMode {
			case NormalEditMode:
				err = visitToPatch(originalObj, updates, mapper, resourceMapper, encoder, out, errOut, defaultVersion, &results, file)
			case EditBeforeCreateMode:
				err = visitToCreate(updates, mapper, resourceMapper, out, errOut, defaultVersion, &results, file)
			default:
				err = fmt.Errorf("Not supported edit mode %q", editMode)
			}
			if err != nil {
				return preservedFile(err, results.file, errOut)
			}

			// Handle all possible errors
			//
			// 1. retryable: propose kubectl replace -f
			// 2. notfound: indicate the location of the saved configuration of the deleted resource
			// 3. invalid: retry those on the spot by looping ie. reloading the editor
			if results.retryable > 0 {
				fmt.Fprintf(errOut, "You can run `%s replace -f %s` to try this update again.\n", filepath.Base(os.Args[0]), file)
				return cmdutil.ErrExit
			}
			if results.notfound > 0 {
				fmt.Fprintf(errOut, "The edits you made on deleted resources have been saved to %q\n", file)
				return cmdutil.ErrExit
			}

			if len(results.edit) == 0 {
				if results.notfound == 0 {
					os.Remove(file)
				} else {
					fmt.Fprintf(out, "The edits you made on deleted resources have been saved to %q\n", file)
				}
				return nil
			}

			if len(results.header.reasons) > 0 {
				containsError = true
			}
		}
	}

	switch editMode {
	// If doing normal edit we cannot use Visit because we need to edit a list for convenience. Ref: #20519
	case NormalEditMode:
		return editFn(nil, nil)
	// If doing an edit before created, we don't want a list and instead want the normal behavior as kubectl create.
	case EditBeforeCreateMode:
		return r.Visit(editFn)
	default:
		return fmt.Errorf("Not supported edit mode %q", editMode)
	}
}

func getPrinter(cmd *cobra.Command) (*editPrinterOptions, error) {
	switch format := cmdutil.GetFlagString(cmd, "output"); format {
	case "json":
		return &editPrinterOptions{
			printer:   &kubectl.JSONPrinter{},
			ext:       ".json",
			addHeader: false,
		}, nil
	// If flag -o is not specified, use yaml as default
	case "yaml", "":
		return &editPrinterOptions{
			printer:   &kubectl.YAMLPrinter{},
			ext:       ".yaml",
			addHeader: true,
		}, nil
	default:
		return nil, cmdutil.UsageError(cmd, "The flag 'output' must be one of yaml|json")
	}
}

func getMapperAndResult(f cmdutil.Factory, args []string, options *resource.FilenameOptions, editMode EditMode) (meta.RESTMapper, *resource.Mapper, *resource.Result, string, error) {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return nil, nil, nil, "", err
	}
	var mapper meta.RESTMapper
	var typer runtime.ObjectTyper
	switch editMode {
	case NormalEditMode:
		mapper, typer = f.Object()
	case EditBeforeCreateMode:
		mapper, typer, err = f.UnstructuredObject()
	default:
		return nil, nil, nil, "", fmt.Errorf("Not supported edit mode %q", editMode)
	}
	if err != nil {
		return nil, nil, nil, "", err
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
	switch editMode {
	case NormalEditMode:
		b = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
			ResourceTypeOrNameArgs(true, args...).
			Latest()
	case EditBeforeCreateMode:
		b = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme)
	default:
		return nil, nil, nil, "", fmt.Errorf("Not supported edit mode %q", editMode)
	}
	r := b.NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		ContinueOnError().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return nil, nil, nil, "", err
	}
	return mapper, resourceMapper, r, cmdNamespace, err
}

func visitToPatch(
	originalObj runtime.Object,
	updates *resource.Info,
	mapper meta.RESTMapper,
	resourceMapper *resource.Mapper,
	encoder runtime.Encoder,
	out, errOut io.Writer,
	defaultVersion schema.GroupVersion,
	results *editResults,
	file string,
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

func visitToCreate(updates *resource.Info, mapper meta.RESTMapper, resourceMapper *resource.Mapper, out, errOut io.Writer, defaultVersion schema.GroupVersion, results *editResults, file string) error {
	createVisitor := resource.NewFlattenListVisitor(updates, resourceMapper)
	err := createVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		results.version = defaultVersion
		if err := createAndRefresh(info); err != nil {
			return err
		}
		cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, false, "created")
		return nil
	})
	return err
}

func visitAnnotation(cmd *cobra.Command, f cmdutil.Factory, updates *resource.Info, resourceMapper *resource.Mapper, encoder runtime.Encoder) ([]runtime.Object, error) {
	mutatedObjects := []runtime.Object{}
	annotationVisitor := resource.NewFlattenListVisitor(updates, resourceMapper)
	// iterate through all items to apply annotations
	err := annotationVisitor.Visit(func(info *resource.Info, incomingErr error) error {
		// put configuration annotation in "updates"
		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, encoder); err != nil {
			return err
		}
		if cmdutil.ShouldRecord(cmd, info) {
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

type editPrinterOptions struct {
	printer   kubectl.ResourcePrinter
	ext       string
	addHeader bool
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
