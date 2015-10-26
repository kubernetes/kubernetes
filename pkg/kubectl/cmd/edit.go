/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/jsonmerge"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
	"k8s.io/kubernetes/pkg/util/yaml"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	editLong = `Edit a resource from the default editor.

The edit command allows you to directly edit any API resource you can retrieve via the
command line tools. It will open the editor defined by your KUBE_EDITOR, GIT_EDITOR,
or EDITOR environment variables, or fall back to 'vi'. You can edit multiple objects,
although changes are applied one at a time. The command accepts filenames as well as
command line arguments, although the files you point to must be previously saved
versions of resources.

The files to edit will be output in the default API version, or a version specified
by --output-version. The default format is YAML - if you would like to edit in JSON
pass -o json.

In the event an error occurs while updating, a temporary file will be created on disk
that contains your unapplied changes. The most common error when updating a resource
is another editor changing the resource on the server. When this occurs, you will have
to apply your changes to the newer version of the resource, or update your temporary
saved copy to include the latest resource version.`

	editExample = `  # Edit the service named 'docker-registry':
  $ kubectl edit svc/docker-registry

  # Use an alternative editor
  $ KUBE_EDITOR="nano" kubectl edit svc/docker-registry

  # Edit the service 'docker-registry' in JSON using the v1 API format:
  $ kubectl edit svc/docker-registry --output-version=v1 -o json`
)

var errExit = fmt.Errorf("exit directly")

func NewCmdEdit(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	filenames := []string{}
	cmd := &cobra.Command{
		Use:     "edit (RESOURCE/NAME | -f FILENAME)",
		Short:   "Edit a resource on the server",
		Long:    editLong,
		Example: fmt.Sprintf(editExample),
		Run: func(cmd *cobra.Command, args []string) {
			err := RunEdit(f, out, cmd, args, filenames)
			if err == errExit {
				os.Exit(1)
			}
			cmdutil.CheckErr(err)
		},
	}
	usage := "Filename, directory, or URL to file to use to edit the resource"
	kubectl.AddJsonFilenameFlag(cmd, &filenames, usage)
	cmd.Flags().StringP("output", "o", "yaml", "Output format. One of: yaml|json.")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given version (default api-version).")
	return cmd
}

func RunEdit(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, filenames []string) error {
	var printer kubectl.ResourcePrinter
	var ext string
	switch format := cmdutil.GetFlagString(cmd, "output"); format {
	case "json":
		printer = &kubectl.JSONPrinter{}
		ext = ".json"
	case "yaml":
		printer = &kubectl.YAMLPrinter{}
		ext = ".yaml"
	default:
		return cmdutil.UsageError(cmd, "The flag 'output' must be one of yaml|json")
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	rmap := &resource.Mapper{
		ObjectTyper:  typer,
		RESTMapper:   mapper,
		ClientMapper: f.ClientMapperForCommand(),
	}

	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, filenames...).
		ResourceTypeOrNameArgs(true, args...).
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	infos, err := r.Infos()
	if err != nil {
		return err
	}

	clientConfig, err := f.ClientConfig()
	if err != nil {
		return err
	}

	defaultVersion := cmdutil.OutputVersion(cmd, clientConfig.Version)
	results := editResults{}
	for {
		obj, err := resource.AsVersionedObject(infos, false, defaultVersion)
		if err != nil {
			return preservedFile(err, results.file, out)
		}

		// TODO: add an annotating YAML printer that can print inline comments on each field,
		//   including descriptions or validation errors

		// generate the file to edit
		buf := &bytes.Buffer{}
		if err := results.header.writeTo(buf); err != nil {
			return preservedFile(err, results.file, out)
		}
		if err := printer.PrintObj(obj, buf); err != nil {
			return preservedFile(err, results.file, out)
		}
		original := buf.Bytes()

		// launch the editor
		edit := editor.NewDefaultEditor()
		edited, file, err := edit.LaunchTempFile("kubectl-edit-", ext, buf)
		if err != nil {
			return preservedFile(err, results.file, out)
		}

		// cleanup any file from the previous pass
		if len(results.file) > 0 {
			os.Remove(results.file)
		}

		glog.V(4).Infof("User edited:\n%s", string(edited))
		fmt.Printf("User edited:\n%s", string(edited))
		lines, err := hasLines(bytes.NewBuffer(edited))
		if err != nil {
			return preservedFile(err, file, out)
		}
		if bytes.Equal(original, edited) {
			if len(results.edit) > 0 {
				preservedFile(nil, file, out)
			} else {
				os.Remove(file)
			}
			fmt.Fprintln(out, "Edit cancelled, no changes made.")
			return nil
		}
		if !lines {
			if len(results.edit) > 0 {
				preservedFile(nil, file, out)
			} else {
				os.Remove(file)
			}
			fmt.Fprintln(out, "Edit cancelled, saved file was empty.")
			return nil
		}

		results = editResults{
			file: file,
		}

		// parse the edited file
		updates, err := rmap.InfoForData(edited, "edited-file")
		if err != nil {
			return preservedFile(err, file, out)
		}

		// annotate the edited object for kubectl apply
		if err := kubectl.UpdateApplyAnnotation(updates); err != nil {
			return preservedFile(err, file, out)
		}

		visitor := resource.NewFlattenListVisitor(updates, rmap)

		// need to make sure the original namespace wasn't changed while editing
		if err = visitor.Visit(resource.RequireNamespace(cmdNamespace)); err != nil {
			return preservedFile(err, file, out)
		}

		// use strategic merge to create a patch
		originalJS, err := yaml.ToJSON(original)
		if err != nil {
			return preservedFile(err, file, out)
		}
		editedJS, err := yaml.ToJSON(edited)
		if err != nil {
			return preservedFile(err, file, out)
		}
		patch, err := strategicpatch.CreateStrategicMergePatch(originalJS, editedJS, obj)
		// TODO: change all jsonmerge to strategicpatch
		// for checking preconditions
		preconditions := []jsonmerge.PreconditionFunc{}
		if err != nil {
			glog.V(4).Infof("Unable to calculate diff, no merge is possible: %v", err)
			return preservedFile(err, file, out)
		} else {
			preconditions = append(preconditions, jsonmerge.RequireKeyUnchanged("apiVersion"))
			preconditions = append(preconditions, jsonmerge.RequireKeyUnchanged("kind"))
			preconditions = append(preconditions, jsonmerge.RequireMetadataKeyUnchanged("name"))
			results.version = defaultVersion
		}

		if hold, msg := jsonmerge.TestPreconditionsHold(patch, preconditions); !hold {
			fmt.Fprintf(out, "error: %s", msg)
			return preservedFile(nil, file, out)
		}

		err = visitor.Visit(func(info *resource.Info, err error) error {
			patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch)
			if err != nil {
				fmt.Fprintln(out, results.addError(err, info))
				return nil
			}
			info.Refresh(patched, true)
			cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, "edited")
			return nil
		})
		if err != nil {
			return preservedFile(err, file, out)
		}

		if results.retryable > 0 {
			fmt.Fprintf(out, "You can run `kubectl replace -f %s` to try this update again.\n", file)
			return errExit
		}
		if results.conflict > 0 {
			fmt.Fprintf(out, "You must update your local resource version and run `kubectl replace -f %s` to overwrite the remote changes.\n", file)
			return errExit
		}
		if len(results.edit) == 0 {
			if results.notfound == 0 {
				os.Remove(file)
			} else {
				fmt.Fprintf(out, "The edits you made on deleted resources have been saved to %q\n", file)
			}
			return nil
		}

		// loop again and edit the remaining items
		infos = results.edit
	}
	return nil
}

// print json file (such as patch file) content for debugging
func printJson(out io.Writer, file []byte) error {
	diff := make(map[string]interface{})
	if err := json.Unmarshal(file, &diff); err != nil {
		return err
	}
	fmt.Fprintf(out, "%v\n", diff)
	return nil
}

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

// editResults capture the result of an update
type editResults struct {
	header    editHeader
	retryable int
	notfound  int
	conflict  int
	edit      []*resource.Info
	file      string

	version string
}

func (r *editResults) addError(err error, info *resource.Info) string {
	switch {
	case errors.IsInvalid(err):
		r.edit = append(r.edit, info)
		reason := editReason{
			head: fmt.Sprintf("%s %s was not valid", info.Mapping.Kind, info.Name),
		}
		if err, ok := err.(client.APIStatus); ok {
			if details := err.Status().Details; details != nil {
				for _, cause := range details.Causes {
					reason.other = append(reason.other, cause.Message)
				}
			}
		}
		r.header.reasons = append(r.header.reasons, reason)
		return fmt.Sprintf("Error: the %s %s is invalid", info.Mapping.Kind, info.Name)
	case errors.IsNotFound(err):
		r.notfound++
		return fmt.Sprintf("Error: the %s %s could not be found on the server", info.Mapping.Kind, info.Name)
	default:
		r.retryable++
		return fmt.Sprintf("Error: the %s %s could not be patched: %v", info.Mapping.Kind, info.Name, err)
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
