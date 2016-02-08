/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/strategicpatch"

	"github.com/evanphx/json-patch"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	ApplyAnnotationsFlag = "save-config"
)

type debugError interface {
	DebugError() (msg string, args []interface{})
}

// AddSourceToErr adds handleResourcePrefix and source string to error message.
// verb is the string like "creating", "deleting" etc.
// souce is the filename or URL to the template file(*.json or *.yaml), or stdin to use to handle the resource.
func AddSourceToErr(verb string, source string, err error) error {
	if source != "" {
		if statusError, ok := err.(errors.APIStatus); ok {
			status := statusError.Status()
			status.Message = fmt.Sprintf("error when %s %q: %v", verb, source, status.Message)
			return &errors.StatusError{ErrStatus: status}
		}
		return fmt.Errorf("error when %s %q: %v", verb, source, err)
	}
	return err
}

var fatalErrHandler = fatal

// BehaviorOnFatal allows you to override the default behavior when a fatal
// error occurs, which is call os.Exit(1). You can pass 'panic' as a function
// here if you prefer the panic() over os.Exit(1).
func BehaviorOnFatal(f func(string)) {
	fatalErrHandler = f
}

// DefaultBehaviorOnFatal allows you to undo any previous override.  Useful in
// tests.
func DefaultBehaviorOnFatal() {
	fatalErrHandler = fatal
}

// fatal prints the message and then exits. If V(2) or greater, glog.Fatal
// is invoked for extended information.
func fatal(msg string) {
	// add newline if needed
	if !strings.HasSuffix(msg, "\n") {
		msg += "\n"
	}

	if glog.V(2) {
		glog.FatalDepth(2, msg)
	}
	fmt.Fprint(os.Stderr, msg)
	os.Exit(1)
}

// CheckErr prints a user friendly error to STDERR and exits with a non-zero
// exit code. Unrecognized errors will be printed with an "error: " prefix.
//
// This method is generic to the command in use and may be used by non-Kubectl
// commands.
func CheckErr(err error) {
	checkErr(err, fatalErrHandler)
}

func checkErr(err error, handleErr func(string)) {
	if err == nil {
		return
	}

	if errors.IsInvalid(err) {
		details := err.(*errors.StatusError).Status().Details
		prefix := fmt.Sprintf("The %s %q is invalid.\n", details.Kind, details.Name)
		errs := statusCausesToAggrError(details.Causes)
		handleErr(MultilineError(prefix, errs))
	}

	if meta.IsNoResourceMatchError(err) {
		noMatch := err.(*meta.NoResourceMatchError)

		switch {
		case len(noMatch.PartialResource.Group) > 0 && len(noMatch.PartialResource.Version) > 0:
			handleErr(fmt.Sprintf("the server doesn't have a resource type %q in group %q and version %q", noMatch.PartialResource.Resource, noMatch.PartialResource.Group, noMatch.PartialResource.Version))
		case len(noMatch.PartialResource.Group) > 0:
			handleErr(fmt.Sprintf("the server doesn't have a resource type %q in group %q", noMatch.PartialResource.Resource, noMatch.PartialResource.Group))
		case len(noMatch.PartialResource.Version) > 0:
			handleErr(fmt.Sprintf("the server doesn't have a resource type %q in version %q", noMatch.PartialResource.Resource, noMatch.PartialResource.Version))
		default:
			handleErr(fmt.Sprintf("the server doesn't have a resource type %q", noMatch.PartialResource.Resource))
		}
		return
	}

	// handle multiline errors
	if clientcmd.IsConfigurationInvalid(err) {
		handleErr(MultilineError("Error in configuration: ", err))
	}
	if agg, ok := err.(utilerrors.Aggregate); ok && len(agg.Errors()) > 0 {
		handleErr(MultipleErrors("", agg.Errors()))
	}

	msg, ok := StandardErrorMessage(err)
	if !ok {
		msg = err.Error()
		if !strings.HasPrefix(msg, "error: ") {
			msg = fmt.Sprintf("error: %s", msg)
		}
	}
	handleErr(msg)
}

func statusCausesToAggrError(scs []unversioned.StatusCause) utilerrors.Aggregate {
	errs := make([]error, len(scs))
	for i, sc := range scs {
		errs[i] = fmt.Errorf("%s: %s", sc.Field, sc.Message)
	}
	return utilerrors.NewAggregate(errs)
}

// StandardErrorMessage translates common errors into a human readable message, or returns
// false if the error is not one of the recognized types. It may also log extended
// information to glog.
//
// This method is generic to the command in use and may be used by non-Kubectl
// commands.
func StandardErrorMessage(err error) (string, bool) {
	if debugErr, ok := err.(debugError); ok {
		glog.V(4).Infof(debugErr.DebugError())
	}
	_, isStatus := err.(errors.APIStatus)
	switch {
	case isStatus:
		return fmt.Sprintf("Error from server: %s", err.Error()), true
	case errors.IsUnexpectedObjectError(err):
		return fmt.Sprintf("Server returned an unexpected response: %s", err.Error()), true
	}
	switch t := err.(type) {
	case *url.Error:
		glog.V(4).Infof("Connection error: %s %s: %v", t.Op, t.URL, t.Err)
		switch {
		case strings.Contains(t.Err.Error(), "connection refused"):
			host := t.URL
			if server, err := url.Parse(t.URL); err == nil {
				host = server.Host
			}
			return fmt.Sprintf("The connection to the server %s was refused - did you specify the right host or port?", host), true
		}
		return fmt.Sprintf("Unable to connect to the server: %v", t.Err), true
	}
	return "", false
}

// MultilineError returns a string representing an error that splits sub errors into their own
// lines. The returned string will end with a newline.
func MultilineError(prefix string, err error) string {
	if agg, ok := err.(utilerrors.Aggregate); ok {
		errs := utilerrors.Flatten(agg).Errors()
		buf := &bytes.Buffer{}
		switch len(errs) {
		case 0:
			return fmt.Sprintf("%s%v\n", prefix, err)
		case 1:
			return fmt.Sprintf("%s%v\n", prefix, messageForError(errs[0]))
		default:
			fmt.Fprintln(buf, prefix)
			for _, err := range errs {
				fmt.Fprintf(buf, "* %v\n", messageForError(err))
			}
			return buf.String()
		}
	}
	return fmt.Sprintf("%s%s\n", prefix, err)
}

// MultipleErrors returns a newline delimited string containing
// the prefix and referenced errors in standard form.
func MultipleErrors(prefix string, errs []error) string {
	buf := &bytes.Buffer{}
	for _, err := range errs {
		fmt.Fprintf(buf, "%s%v\n", prefix, messageForError(err))
	}
	return buf.String()
}

// messageForError returns the string representing the error.
func messageForError(err error) string {
	msg, ok := StandardErrorMessage(err)
	if !ok {
		msg = err.Error()
	}
	return msg
}

func UsageError(cmd *cobra.Command, format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s\nSee '%s -h' for help and examples.", msg, cmd.CommandPath())
}

// Whether this cmd need watching objects.
func isWatch(cmd *cobra.Command) bool {
	if w, err := cmd.Flags().GetBool("watch"); w && err == nil {
		return true
	}

	if wo, err := cmd.Flags().GetBool("watch-only"); wo && err == nil {
		return true
	}

	return false
}

func getFlag(cmd *cobra.Command, flag string) *pflag.Flag {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	return f
}

func GetFlagString(cmd *cobra.Command, flag string) string {
	s, err := cmd.Flags().GetString(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return s
}

// GetFlagStringList can be used to accept multiple argument with flag repetition (e.g. -f arg1 -f arg2 ...)
func GetFlagStringSlice(cmd *cobra.Command, flag string) []string {
	s, err := cmd.Flags().GetStringSlice(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return s
}

// GetWideFlag is used to determine if "-o wide" is used
func GetWideFlag(cmd *cobra.Command) bool {
	f := cmd.Flags().Lookup("output")
	if f.Value.String() == "wide" {
		return true
	}
	return false
}

func GetFlagBool(cmd *cobra.Command, flag string) bool {
	b, err := cmd.Flags().GetBool(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return b
}

// Assumes the flag has a default value.
func GetFlagInt(cmd *cobra.Command, flag string) int {
	i, err := cmd.Flags().GetInt(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return i
}

// Assumes the flag has a default value.
func GetFlagInt64(cmd *cobra.Command, flag string) int64 {
	i, err := cmd.Flags().GetInt64(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return i
}

func GetFlagDuration(cmd *cobra.Command, flag string) time.Duration {
	d, err := cmd.Flags().GetDuration(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return d
}

func AddValidateFlags(cmd *cobra.Command) {
	cmd.Flags().Bool("validate", true, "If true, use a schema to validate the input before sending it")
	cmd.Flags().String("schema-cache-dir", fmt.Sprintf("~/%s/%s", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName), fmt.Sprintf("If non-empty, load/store cached API schemas in this directory, default is '$HOME/%s/%s'", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName))
}

func AddApplyAnnotationFlags(cmd *cobra.Command) {
	cmd.Flags().Bool(ApplyAnnotationsFlag, false, "If true, the configuration of current object will be saved in its annotation. This is useful when you want to perform kubectl apply on this object in the future.")
}

// AddGeneratorFlags adds flags common to resource generation commands
// TODO: need to take a pass at other generator commands to use this set of flags
func AddGeneratorFlags(cmd *cobra.Command, defaultGenerator string) {
	cmd.Flags().String("generator", defaultGenerator, "The name of the API generator to use.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")
	cmd.Flags().StringP("output", "o", "", "Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/HEAD/docs/user-guide/jsonpath.md].")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given version (default api-version).")
}

func ReadConfigDataFromReader(reader io.Reader, source string) ([]byte, error) {
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, fmt.Errorf(`Read from %s but no data found`, source)
	}

	return data, nil
}

// ReadConfigData reads the bytes from the specified filesytem or network
// location or from stdin if location == "-".
// TODO: replace with resource.Builder
func ReadConfigData(location string) ([]byte, error) {
	if len(location) == 0 {
		return nil, fmt.Errorf("location given but empty")
	}

	if location == "-" {
		// Read from stdin.
		return ReadConfigDataFromReader(os.Stdin, "stdin ('-')")
	}

	// Use the location as a file path or URL.
	return ReadConfigDataFromLocation(location)
}

// TODO: replace with resource.Builder
func ReadConfigDataFromLocation(location string) ([]byte, error) {
	// we look for http:// or https:// to determine if valid URL, otherwise do normal file IO
	if strings.Index(location, "http://") == 0 || strings.Index(location, "https://") == 0 {
		resp, err := http.Get(location)
		if err != nil {
			return nil, fmt.Errorf("unable to access URL %s: %v\n", location, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("unable to read URL, server reported %d %s", resp.StatusCode, resp.Status)
		}
		return ReadConfigDataFromReader(resp.Body, location)
	} else {
		file, err := os.Open(location)
		if err != nil {
			return nil, fmt.Errorf("unable to read %s: %v\n", location, err)
		}
		return ReadConfigDataFromReader(file, location)
	}
}

// Merge requires JSON serialization
// TODO: merge assumes JSON serialization, and does not properly abstract API retrieval
func Merge(codec runtime.Codec, dst runtime.Object, fragment, kind string) (runtime.Object, error) {
	// encode dst into versioned json and apply fragment directly too it
	target, err := runtime.Encode(codec, dst)
	if err != nil {
		return nil, err
	}
	patched, err := jsonpatch.MergePatch(target, []byte(fragment))
	if err != nil {
		return nil, err
	}
	out, err := runtime.Decode(codec, patched)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// DumpReaderToFile writes all data from the given io.Reader to the specified file
// (usually for temporary use).
func DumpReaderToFile(reader io.Reader, filename string) error {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	defer f.Close()
	if err != nil {
		return err
	}
	buffer := make([]byte, 1024)
	for {
		count, err := reader.Read(buffer)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		_, err = f.Write(buffer[:count])
		if err != nil {
			return err
		}
	}
	return nil
}

// UpdateObject updates resource object with updateFn
func UpdateObject(info *resource.Info, codec runtime.Codec, updateFn func(runtime.Object) error) (runtime.Object, error) {
	helper := resource.NewHelper(info.Client, info.Mapping)

	if err := updateFn(info.Object); err != nil {
		return nil, err
	}

	// Update the annotation used by kubectl apply
	if err := kubectl.UpdateApplyAnnotation(info, codec); err != nil {
		return nil, err
	}

	if _, err := helper.Replace(info.Namespace, info.Name, true, info.Object); err != nil {
		return nil, err
	}

	return info.Object, nil
}

// AddCmdRecordFlag adds --record flag to command
func AddRecordFlag(cmd *cobra.Command) {
	cmd.Flags().Bool("record", false, "Record current kubectl command in the resource annotation.")
}

func GetRecordFlag(cmd *cobra.Command) bool {
	return GetFlagBool(cmd, "record")
}

// RecordChangeCause annotate change-cause to input runtime object.
func RecordChangeCause(obj runtime.Object, changeCause string) error {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return err
	}
	if meta.Annotations == nil {
		meta.Annotations = make(map[string]string)
	}
	meta.Annotations[kubectl.ChangeCauseAnnotation] = changeCause
	return nil
}

// ChangeResourcePatch creates a strategic merge patch between the origin input resource info
// and the annotated with change-cause input resource info.
func ChangeResourcePatch(info *resource.Info, changeCause string) ([]byte, error) {
	oldData, err := json.Marshal(info.Object)
	if err != nil {
		return nil, err
	}
	if err := RecordChangeCause(info.Object, changeCause); err != nil {
		return nil, err
	}
	newData, err := json.Marshal(info.Object)
	if err != nil {
		return nil, err
	}
	return strategicpatch.CreateTwoWayMergePatch(oldData, newData, info.Object)
}

// containsChangeCause checks if input resource info contains change-cause annotation.
func ContainsChangeCause(info *resource.Info) bool {
	annotations, err := info.Mapping.MetadataAccessor.Annotations(info.Object)
	if err != nil {
		return false
	}
	return len(annotations[kubectl.ChangeCauseAnnotation]) > 0
}

// ShouldRecord checks if we should record current change cause
func ShouldRecord(cmd *cobra.Command, info *resource.Info) bool {
	return GetRecordFlag(cmd) || ContainsChangeCause(info)
}
