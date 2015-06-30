package cmd

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	kclient "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util/editor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util/jsonmerge"

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

	editExample = `  // Edit the service named 'docker-registry':
  $ kubectl edit svc/docker-registry

  // Edit the DeploymentConfig named 'my-deployment':
  $ kubectl edit dc/my-deployment

  // Use an alternative editor
  $ KUBE_EDITOR="nano" kubectl edit dc/my-deployment

  // Edit the service 'docker-registry' in JSON using the v1 API format:
  $ kubectl edit svc/docker-registry --output-version=v1 -o json`
)

var errExit = fmt.Errorf("exit directly")

func NewCmdEdit(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	var filenames util.StringList
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

func RunEdit(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, filenames util.StringList) error {
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

	cmdNamespace, err := f.DefaultNamespace()
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
		FilenameParam(filenames...).
		ResourceTypeOrNameArgs(true, args...).
		Latest().
		Flatten().
		Do()
	if r.err != nil {
		return r.err
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
			return preservedFile(err, results.file, cmd.Out())
		}

		// TODO: add an annotating YAML printer that can print inline comments on each field,
		//   including descriptions or validation errors

		// generate the file to edit
		buf := &bytes.Buffer{}
		if err := results.header.WriteTo(buf); err != nil {
			return preservedFile(err, results.file, cmd.Out())
		}
		if err := printer.PrintObj(obj, buf); err != nil {
			return preservedFile(err, results.file, cmd.Out())
		}
		original := buf.Bytes()

		// launch the editor
		edit := editor.NewDefaultEditor()
		edited, file, err := edit.LaunchTempFile("kubectl-edit-", ext, buf)
		if err != nil {
			return preservedFile(err, results.file, cmd.Out())
		}

		// cleanup any file from the previous pass
		if len(results.file) > 0 {
			os.Remove(results.file)
		}

		glog.V(4).Infof("User edited:\n%s", string(edited))
		lines, err := hasLines(bytes.NewBuffer(edited))
		if err != nil {
			return preservedFile(err, file, cmd.Out())
		}
		if bytes.Equal(original, edited) {
			if len(results.edit) > 0 {
				preservedFile(nil, file, cmd.Out())
			} else {
				os.Remove(file)
			}
			fmt.Fprintln(cmd.Out(), "Edit cancelled, no changes made.")
			return nil
		}
		if !lines {
			if len(results.edit) > 0 {
				preservedFile(nil, file, cmd.Out())
			} else {
				os.Remove(file)
			}
			fmt.Fprintln(cmd.Out(), "Edit cancelled, saved file was empty.")
			return nil
		}

		results = editResults{
			file: file,
		}

		// parse the edited file
		updates, err := rmap.InfoForData(edited, "edited-file")
		if err != nil {
			results.header.reasons = append(results.header.reasons, editReason{
				head: fmt.Sprintf("The edited file had a syntax error: %v", err),
			})
			continue
		}

		visitor := resource.NewFlattenListVisitor(updates, rmap)

		// need to make sure the original namespace wasn't changed while editing
		if err = visitor.Visit(resource.RequireNamespace(cmdNamespace)); err != nil {
			return preservedFile(err, file, cmd.Out())
		}

		// attempt to calculate a delta for merging conflicts
		delta, err := jsonmerge.NewDelta(original, edited)
		if err != nil {
			glog.V(4).Infof("Unable to calculate diff, no merge is possible: %v", err)
			delta = nil
		} else {
			delta.AddPreconditions(jsonmerge.RequireKeyUnchanged("apiVersion"))
			results.delta = delta
			results.version = defaultVersion
		}

		err = visitor.Visit(func(info *resource.Info) error {
			data, err := info.Mapping.Codec.Encode(info.Object)
			if err != nil {
				return err
			}
			updated, err := resource.NewHelper(info.Client, info.Mapping).Update(info.Namespace, info.Name, false, data)
			if err != nil {
				fmt.Fprintln(cmd.Out(), results.AddError(err, info))
				return nil
			}
			info.Refresh(updated, true)
			fmt.Fprintf(out, "%s/%s\n", info.Mapping.Resource, info.Name)
			return nil
		})
		if err != nil {
			return preservedFile(err, file, cmd.Out())
		}

		if results.retryable > 0 {
			fmt.Fprintf(cmd.Out(), "You can run `kubectl update -f %s` to try this update again.\n", file)
			return errExit
		}
		if results.conflict > 0 {
			fmt.Fprintf(cmd.Out(), "You must update your local resource version and run `kubectl update -f %s` to overwrite the remote changes.\n", file)
			return errExit
		}
		if len(results.edit) == 0 {
			if results.notfound == 0 {
				os.Remove(file)
			} else {
				fmt.Fprintf(cmd.Out(), "The edits you made on deleted resources have been saved to %q\n", file)
			}
			return nil
		}

		// loop again and edit the remaining items
		infos = results.edit
	}
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

// WriteTo outputs the current header information into a stream
func (h *editHeader) WriteTo(w io.Writer) error {
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

	delta   *jsonmerge.Delta
	version string
}

func (r *editResults) AddError(err error, info *resource.Info) string {
	switch {
	case errors.IsInvalid(err):
		r.edit = append(r.edit, info)
		reason := editReason{
			head: fmt.Sprintf("%s %s was not valid", info.Mapping.Kind, info.Name),
		}
		if err, ok := err.(kclient.APIStatus); ok {
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
		return fmt.Sprintf("Error: the %s %s has been deleted on the server", info.Mapping.Kind, info.Name)

	case errors.IsConflict(err):
		if r.delta != nil {
			v1 := info.ResourceVersion
			if perr := applyPatch(r.delta, info, r.version); perr != nil {
				// the error was related to the patching process
				if nerr, ok := perr.(patchError); ok {
					r.conflict++
					if jsonmerge.IsPreconditionFailed(nerr.error) {
						return fmt.Sprintf("Error: the API version of the provided object cannot be changed")
					}
					// the patch is in conflict, report to user and exit
					if jsonmerge.IsConflicting(nerr.error) {
						// TODO: read message
						return fmt.Sprintf("Error: a conflicting change was made to the %s %s on the server", info.Mapping.Kind, info.Name)
					}
					glog.V(4).Infof("Attempted to patch the resource, but failed: %v", perr)
					return fmt.Sprintf("Error: %v", err)
				}
				// try processing this server error and unset delta so we don't recurse
				r.delta = nil
				return r.AddError(err, info)
			}
			return fmt.Sprintf("Applied your changes to %s from version %s onto %s", info.Name, v1, info.ResourceVersion)
		}
		// no delta was available
		r.conflict++
		return fmt.Sprintf("Error: %v", err)
	default:
		r.retryable++
		return fmt.Sprintf("Error: the %s %s could not be updated: %v", info.Mapping.Kind, info.Name, err)
	}
}

type patchError struct {
	error
}

// applyPatch reads the latest version of the object, writes it to version, then attempts to merge
// the changes onto it without conflict. If a conflict occurs jsonmerge.IsConflicting(err) is
// true. The info object is mutated
func applyPatch(delta *jsonmerge.Delta, info *resource.Info, version string) error {
	if err := info.Get(); err != nil {
		return patchError{err}
	}
	obj, err := resource.AsVersionedObject([]*resource.Info{info}, false, version)
	if err != nil {
		return patchError{err}
	}
	data, err := info.Mapping.Codec.Encode(obj)
	if err != nil {
		return patchError{err}
	}
	merged, err := delta.Apply(data)
	if err != nil {
		return patchError{err}
	}
	updated, err := resource.NewHelper(info.Client, info.Mapping).Update(info.Namespace, info.Name, false, merged)
	if err != nil {
		return err
	}
	info.Refresh(updated, true)
	return nil
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
