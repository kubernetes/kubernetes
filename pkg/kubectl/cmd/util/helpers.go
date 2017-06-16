/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"strings"
	"time"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/golang/glog"
	"github.com/spf13/cobra"

	kerrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

const (
	ApplyAnnotationsFlag = "save-config"
	DefaultErrorExitCode = 1
)

type debugError interface {
	DebugError() (msg string, args []interface{})
}

// AddSourceToErr adds handleResourcePrefix and source string to error message.
// verb is the string like "creating", "deleting" etc.
// source is the filename or URL to the template file(*.json or *.yaml), or stdin to use to handle the resource.
func AddSourceToErr(verb string, source string, err error) error {
	if source != "" {
		if statusError, ok := err.(kerrors.APIStatus); ok {
			status := statusError.Status()
			status.Message = fmt.Sprintf("error when %s %q: %v", verb, source, status.Message)
			return &kerrors.StatusError{ErrStatus: status}
		}
		return fmt.Errorf("error when %s %q: %v", verb, source, err)
	}
	return err
}

var fatalErrHandler = fatal

// BehaviorOnFatal allows you to override the default behavior when a fatal
// error occurs, which is to call os.Exit(code). You can pass 'panic' as a function
// here if you prefer the panic() over os.Exit(1).
func BehaviorOnFatal(f func(string, int)) {
	fatalErrHandler = f
}

// DefaultBehaviorOnFatal allows you to undo any previous override.  Useful in
// tests.
func DefaultBehaviorOnFatal() {
	fatalErrHandler = fatal
}

// fatal prints the message (if provided) and then exits. If V(2) or greater,
// glog.Fatal is invoked for extended information.
func fatal(msg string, code int) {
	if glog.V(2) {
		glog.FatalDepth(2, msg)
	}
	if len(msg) > 0 {
		// add newline if needed
		if !strings.HasSuffix(msg, "\n") {
			msg += "\n"
		}
		fmt.Fprint(os.Stderr, msg)
	}
	os.Exit(code)
}

// ErrExit may be passed to CheckError to instruct it to output nothing but exit with
// status code 1.
var ErrExit = fmt.Errorf("exit")

// CheckErr prints a user friendly error to STDERR and exits with a non-zero
// exit code. Unrecognized errors will be printed with an "error: " prefix.
//
// This method is generic to the command in use and may be used by non-Kubectl
// commands.
func CheckErr(err error) {
	checkErr("", err, fatalErrHandler)
}

// checkErrWithPrefix works like CheckErr, but adds a caller-defined prefix to non-nil errors
func checkErrWithPrefix(prefix string, err error) {
	checkErr(prefix, err, fatalErrHandler)
}

// checkErr formats a given error as a string and calls the passed handleErr
// func with that string and an kubectl exit code.
func checkErr(prefix string, err error, handleErr func(string, int)) {
	// unwrap aggregates of 1
	if agg, ok := err.(utilerrors.Aggregate); ok && len(agg.Errors()) == 1 {
		err = agg.Errors()[0]
	}

	switch {
	case err == nil:
		return
	case err == ErrExit:
		handleErr("", DefaultErrorExitCode)
		return
	case kerrors.IsInvalid(err):
		details := err.(*kerrors.StatusError).Status().Details
		s := fmt.Sprintf("%sThe %s %q is invalid", prefix, details.Kind, details.Name)
		if len(details.Causes) > 0 {
			errs := statusCausesToAggrError(details.Causes)
			handleErr(MultilineError(s+": ", errs), DefaultErrorExitCode)
		} else {
			handleErr(s, DefaultErrorExitCode)
		}
	case clientcmd.IsConfigurationInvalid(err):
		handleErr(MultilineError(fmt.Sprintf("%sError in configuration: ", prefix), err), DefaultErrorExitCode)
	default:
		switch err := err.(type) {
		case *meta.NoResourceMatchError:
			switch {
			case len(err.PartialResource.Group) > 0 && len(err.PartialResource.Version) > 0:
				handleErr(fmt.Sprintf("%sthe server doesn't have a resource type %q in group %q and version %q", prefix, err.PartialResource.Resource, err.PartialResource.Group, err.PartialResource.Version), DefaultErrorExitCode)
			case len(err.PartialResource.Group) > 0:
				handleErr(fmt.Sprintf("%sthe server doesn't have a resource type %q in group %q", prefix, err.PartialResource.Resource, err.PartialResource.Group), DefaultErrorExitCode)
			case len(err.PartialResource.Version) > 0:
				handleErr(fmt.Sprintf("%sthe server doesn't have a resource type %q in version %q", prefix, err.PartialResource.Resource, err.PartialResource.Version), DefaultErrorExitCode)
			default:
				handleErr(fmt.Sprintf("%sthe server doesn't have a resource type %q", prefix, err.PartialResource.Resource), DefaultErrorExitCode)
			}
		case utilerrors.Aggregate:
			handleErr(MultipleErrors(prefix, err.Errors()), DefaultErrorExitCode)
		case utilexec.ExitError:
			// do not print anything, only terminate with given error
			handleErr("", err.ExitStatus())
		default: // for any other error type
			msg, ok := StandardErrorMessage(err)
			if !ok {
				msg = err.Error()
				if !strings.HasPrefix(msg, "error: ") {
					msg = fmt.Sprintf("error: %s", msg)
				}
			}
			handleErr(msg, DefaultErrorExitCode)
		}
	}
}

func statusCausesToAggrError(scs []metav1.StatusCause) utilerrors.Aggregate {
	errs := make([]error, 0, len(scs))
	errorMsgs := sets.NewString()
	for _, sc := range scs {
		// check for duplicate error messages and skip them
		msg := fmt.Sprintf("%s: %s", sc.Field, sc.Message)
		if errorMsgs.Has(msg) {
			continue
		}
		errorMsgs.Insert(msg)
		errs = append(errs, errors.New(msg))
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
	status, isStatus := err.(kerrors.APIStatus)
	switch {
	case isStatus:
		switch s := status.Status(); {
		case s.Reason == metav1.StatusReasonUnauthorized:
			return fmt.Sprintf("error: You must be logged in to the server (%s)", s.Message), true
		case len(s.Reason) > 0:
			return fmt.Sprintf("Error from server (%s): %s", s.Reason, err.Error()), true
		default:
			return fmt.Sprintf("Error from server: %s", err.Error()), true
		}
	case kerrors.IsUnexpectedObjectError(err):
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

// PrintErrorWithCauses prints an error's kind, name, and each of the error's causes in a new line.
// The returned string will end with a newline.
// Returns true if a case exists to handle the error type, or false otherwise.
func PrintErrorWithCauses(err error, errOut io.Writer) bool {
	switch t := err.(type) {
	case *kerrors.StatusError:
		errorDetails := t.Status().Details
		if errorDetails != nil {
			fmt.Fprintf(errOut, "error: %s %q is invalid\n\n", errorDetails.Kind, errorDetails.Name)
			for _, cause := range errorDetails.Causes {
				fmt.Fprintf(errOut, "* %s: %s\n", cause.Field, cause.Message)
			}
			return true
		}
	}

	fmt.Fprintf(errOut, "error: %v\n", err)
	return false
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

func IsFilenameEmpty(filenames []string) bool {
	return len(filenames) == 0
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

func GetFlagString(cmd *cobra.Command, flag string) string {
	s, err := cmd.Flags().GetString(flag)
	if err != nil {
		glog.Fatalf("error accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return s
}

// GetFlagStringSlice can be used to accept multiple argument with flag repetition (e.g. -f arg1,arg2 -f arg3 ...)
func GetFlagStringSlice(cmd *cobra.Command, flag string) []string {
	s, err := cmd.Flags().GetStringSlice(flag)
	if err != nil {
		glog.Fatalf("error accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return s
}

// GetFlagStringArray can be used to accept multiple argument with flag repetition (e.g. -f arg1 -f arg2 ...)
func GetFlagStringArray(cmd *cobra.Command, flag string) []string {
	s, err := cmd.Flags().GetStringArray(flag)
	if err != nil {
		glog.Fatalf("err accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return s
}

// GetWideFlag is used to determine if "-o wide" is used
func GetWideFlag(cmd *cobra.Command) bool {
	f := cmd.Flags().Lookup("output")
	if f != nil && f.Value != nil && f.Value.String() == "wide" {
		return true
	}
	return false
}

func GetFlagBool(cmd *cobra.Command, flag string) bool {
	b, err := cmd.Flags().GetBool(flag)
	if err != nil {
		glog.Fatalf("error accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return b
}

// Assumes the flag has a default value.
func GetFlagInt(cmd *cobra.Command, flag string) int {
	i, err := cmd.Flags().GetInt(flag)
	if err != nil {
		glog.Fatalf("error accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return i
}

// Assumes the flag has a default value.
func GetFlagInt64(cmd *cobra.Command, flag string) int64 {
	i, err := cmd.Flags().GetInt64(flag)
	if err != nil {
		glog.Fatalf("error accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return i
}

func GetFlagDuration(cmd *cobra.Command, flag string) time.Duration {
	d, err := cmd.Flags().GetDuration(flag)
	if err != nil {
		glog.Fatalf("error accessing flag %s for command %s: %v", flag, cmd.Name(), err)
	}
	return d
}

func GetPodRunningTimeoutFlag(cmd *cobra.Command) (time.Duration, error) {
	timeout := GetFlagDuration(cmd, "pod-running-timeout")
	if timeout <= 0 {
		return timeout, fmt.Errorf("--pod-running-timeout must be higher than zero")
	}
	return timeout, nil
}

func AddValidateFlags(cmd *cobra.Command) {
	cmd.Flags().Bool("validate", true, "If true, use a schema to validate the input before sending it")
	cmd.Flags().String("schema-cache-dir", fmt.Sprintf("~/%s/%s", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName), fmt.Sprintf("If non-empty, load/store cached API schemas in this directory, default is '$HOME/%s/%s'", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName))
	cmd.MarkFlagFilename("schema-cache-dir")
}

func AddValidateOptionFlags(cmd *cobra.Command, options *ValidateOptions) {
	cmd.Flags().BoolVar(&options.EnableValidation, "validate", true, "If true, use a schema to validate the input before sending it")
	cmd.Flags().StringVar(&options.SchemaCacheDir, "schema-cache-dir", fmt.Sprintf("~/%s/%s", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName), fmt.Sprintf("If non-empty, load/store cached API schemas in this directory, default is '$HOME/%s/%s'", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName))
	cmd.MarkFlagFilename("schema-cache-dir")
}

func AddOpenAPIFlags(cmd *cobra.Command) {
	cmd.Flags().String("schema-cache-dir",
		fmt.Sprintf("~/%s/%s", clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName),
		fmt.Sprintf("If non-empty, load/store cached API schemas in this directory, default is '$HOME/%s/%s'",
			clientcmd.RecommendedHomeDir, clientcmd.RecommendedSchemaName),
	)
	cmd.MarkFlagFilename("schema-cache-dir")
}

func GetOpenAPICacheDir(cmd *cobra.Command) string {
	return GetFlagString(cmd, "schema-cache-dir")
}

func AddFilenameOptionFlags(cmd *cobra.Command, options *resource.FilenameOptions, usage string) {
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, "Filename, directory, or URL to files "+usage)
	cmd.Flags().BoolVarP(&options.Recursive, "recursive", "R", options.Recursive, "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.")
}

// AddDryRunFlag adds dry-run flag to a command. Usually used by mutations.
func AddDryRunFlag(cmd *cobra.Command) {
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without sending it.")
}

func AddPodRunningTimeoutFlag(cmd *cobra.Command, defaultTimeout time.Duration) {
	cmd.Flags().Duration("pod-running-timeout", defaultTimeout, "The length of time (like 5s, 2m, or 3h, higher than zero) to wait until at least one pod is running")
}

func AddApplyAnnotationFlags(cmd *cobra.Command) {
	cmd.Flags().Bool(ApplyAnnotationsFlag, false, "If true, the configuration of current object will be saved in its annotation. Otherwise, the annotation will be unchanged. This flag is useful when you want to perform kubectl apply on this object in the future.")
}

func AddApplyAnnotationVarFlags(cmd *cobra.Command, applyAnnotation *bool) {
	cmd.Flags().BoolVar(applyAnnotation, ApplyAnnotationsFlag, false, "If true, the configuration of current object will be saved in its annotation. Otherwise, the annotation will be unchanged. This flag is useful when you want to perform kubectl apply on this object in the future.")
}

// AddGeneratorFlags adds flags common to resource generation commands
// TODO: need to take a pass at other generator commands to use this set of flags
func AddGeneratorFlags(cmd *cobra.Command, defaultGenerator string) {
	cmd.Flags().String("generator", defaultGenerator, "The name of the API generator to use.")
	AddDryRunFlag(cmd)
}

type ValidateOptions struct {
	EnableValidation bool
	SchemaCacheDir   string
}

func ReadConfigDataFromReader(reader io.Reader, source string) ([]byte, error) {
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("Read from %s but no data found", source)
	}

	return data, nil
}

// Merge requires JSON serialization
// TODO: merge assumes JSON serialization, and does not properly abstract API retrieval
func Merge(codec runtime.Codec, dst runtime.Object, fragment string) (runtime.Object, error) {
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
	if err != nil {
		return err
	}
	defer f.Close()

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
	cmd.Flags().Bool("record", false, "Record current kubectl command in the resource annotation. If set to false, do not record the command. If set to true, record the command. If not set, default to updating the existing annotation value only if one already exists.")
}

func AddRecordVarFlag(cmd *cobra.Command, record *bool) {
	cmd.Flags().BoolVar(record, "record", false, "Record current kubectl command in the resource annotation. If set to false, do not record the command. If set to true, record the command. If not set, default to updating the existing annotation value only if one already exists.")
}

func GetRecordFlag(cmd *cobra.Command) bool {
	return GetFlagBool(cmd, "record")
}

func GetDryRunFlag(cmd *cobra.Command) bool {
	return GetFlagBool(cmd, "dry-run")
}

// RecordChangeCause annotate change-cause to input runtime object.
func RecordChangeCause(obj runtime.Object, changeCause string) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[kubectl.ChangeCauseAnnotation] = changeCause
	accessor.SetAnnotations(annotations)
	return nil
}

// ChangeResourcePatch creates a patch between the origin input resource info
// and the annotated with change-cause input resource info.
func ChangeResourcePatch(info *resource.Info, changeCause string) ([]byte, types.PatchType, error) {
	// Get a versioned object
	obj, err := info.Mapping.ConvertToVersion(info.Object, info.Mapping.GroupVersionKind.GroupVersion())
	if err != nil {
		return nil, types.StrategicMergePatchType, err
	}

	oldData, err := json.Marshal(obj)
	if err != nil {
		return nil, types.StrategicMergePatchType, err
	}
	if err := RecordChangeCause(obj, changeCause); err != nil {
		return nil, types.StrategicMergePatchType, err
	}
	newData, err := json.Marshal(obj)
	if err != nil {
		return nil, types.StrategicMergePatchType, err
	}

	switch obj := obj.(type) {
	case *unstructured.Unstructured:
		patch, err := jsonpatch.CreateMergePatch(oldData, newData)
		return patch, types.MergePatchType, err
	default:
		patch, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, obj)
		return patch, types.StrategicMergePatchType, err
	}
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
	return GetRecordFlag(cmd) || (ContainsChangeCause(info) && !cmd.Flags().Changed("record"))
}

func AddInclude3rdPartyFlags(cmd *cobra.Command) {
	cmd.Flags().Bool("include-extended-apis", true, "If true, include definitions of new APIs via calls to the API server. [default true]")
	cmd.Flags().MarkDeprecated("include-extended-apis", "No longer required.")
}

func AddInclude3rdPartyVarFlags(cmd *cobra.Command, include3rdParty *bool) {
	cmd.Flags().BoolVar(include3rdParty, "include-extended-apis", true, "If true, include definitions of new APIs via calls to the API server. [default true]")
	cmd.Flags().MarkDeprecated("include-extended-apis", "No longer required.")
}

// GetResourcesAndPairs retrieves resources and "KEY=VALUE or KEY-" pair args from given args
func GetResourcesAndPairs(args []string, pairType string) (resources []string, pairArgs []string, err error) {
	foundPair := false
	for _, s := range args {
		nonResource := strings.Contains(s, "=") || strings.HasSuffix(s, "-")
		switch {
		case !foundPair && nonResource:
			foundPair = true
			fallthrough
		case foundPair && nonResource:
			pairArgs = append(pairArgs, s)
		case !foundPair && !nonResource:
			resources = append(resources, s)
		case foundPair && !nonResource:
			err = fmt.Errorf("all resources must be specified before %s changes: %s", pairType, s)
			return
		}
	}
	return
}

// ParsePairs retrieves new and remove pairs (if supportRemove is true) from "KEY=VALUE or KEY-" pair args
func ParsePairs(pairArgs []string, pairType string, supportRemove bool) (newPairs map[string]string, removePairs []string, err error) {
	newPairs = map[string]string{}
	if supportRemove {
		removePairs = []string{}
	}
	var invalidBuf bytes.Buffer
	var invalidBufNonEmpty bool
	for _, pairArg := range pairArgs {
		if strings.Contains(pairArg, "=") {
			parts := strings.SplitN(pairArg, "=", 2)
			if len(parts) != 2 {
				if invalidBufNonEmpty {
					invalidBuf.WriteString(", ")
				}
				invalidBuf.WriteString(pairArg)
				invalidBufNonEmpty = true
			} else {
				newPairs[parts[0]] = parts[1]
			}
		} else if supportRemove && strings.HasSuffix(pairArg, "-") {
			removePairs = append(removePairs, pairArg[:len(pairArg)-1])
		} else {
			if invalidBufNonEmpty {
				invalidBuf.WriteString(", ")
			}
			invalidBuf.WriteString(pairArg)
			invalidBufNonEmpty = true
		}
	}
	if invalidBufNonEmpty {
		err = fmt.Errorf("invalid %s format: %s", pairType, invalidBuf.String())
		return
	}

	return
}

// MaybeConvertObject attempts to convert an object to a specific group/version.  If the object is
// a third party resource it is simply passed through.
func MaybeConvertObject(obj runtime.Object, gv schema.GroupVersion, converter runtime.ObjectConvertor) (runtime.Object, error) {
	switch obj.(type) {
	case *extensions.ThirdPartyResourceData:
		// conversion is not supported for 3rd party objects
		return obj, nil
	default:
		return converter.ConvertToVersion(obj, gv)
	}
}

// MustPrintWithKinds determines if printer is dealing
// with multiple resource kinds, in which case it will
// return true, indicating resource kind will be
// included as part of printer output
func MustPrintWithKinds(objs []runtime.Object, infos []*resource.Info, sorter *kubectl.RuntimeSort) bool {
	var lastMap *meta.RESTMapping

	for ix := range objs {
		var mapping *meta.RESTMapping
		if sorter != nil {
			mapping = infos[sorter.OriginalPosition(ix)].Mapping
		} else {
			mapping = infos[ix].Mapping
		}

		// display "kind" only if we have mixed resources
		if lastMap != nil && mapping.Resource != lastMap.Resource {
			return true
		}
		lastMap = mapping
	}

	return false
}

// FilterResourceList receives a list of runtime objects.
// If any objects are filtered, that number is returned along with a modified list.
func FilterResourceList(obj runtime.Object, filterFuncs kubectl.Filters, filterOpts *printers.PrintOptions) (int, []runtime.Object, error) {
	items, err := meta.ExtractList(obj)
	if err != nil {
		return 0, []runtime.Object{obj}, utilerrors.NewAggregate([]error{err})
	}
	if errs := runtime.DecodeList(items, api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme); len(errs) > 0 {
		return 0, []runtime.Object{obj}, utilerrors.NewAggregate(errs)
	}

	filterCount := 0
	list := make([]runtime.Object, 0, len(items))
	for _, obj := range items {
		if isFiltered, err := filterFuncs.Filter(obj, filterOpts); !isFiltered {
			if err != nil {
				glog.V(2).Infof("Unable to filter resource: %v", err)
				continue
			}
			list = append(list, obj)
		} else if isFiltered {
			filterCount++
		}
	}
	return filterCount, list, nil
}

// PrintFilterCount displays informational messages based on the number of resources found, hidden, or
// config flags shown.
func PrintFilterCount(out io.Writer, found, hidden, errors int, options *printers.PrintOptions, ignoreNotFound bool) {
	switch {
	case errors > 0 || ignoreNotFound:
		// print nothing
	case found <= hidden:
		if found == 0 {
			fmt.Fprintln(out, "No resources found.")
		} else {
			fmt.Fprintln(out, "No resources found, use --show-all to see completed objects.")
		}
	case hidden > 0 && !options.ShowAll && !options.NoHeaders:
		if glog.V(2) {
			if hidden > 1 {
				fmt.Fprintf(out, "info: %d objects not shown, use --show-all to see completed objects.\n", hidden)
			} else {
				fmt.Fprintf(out, "info: 1 object not shown, use --show-all to see completed objects.\n")
			}
		}
	}
}

// ObjectListToVersionedObject receives a list of api objects and a group version
// and squashes the list's items into a single versioned runtime.Object.
func ObjectListToVersionedObject(objects []runtime.Object, version schema.GroupVersion) (runtime.Object, error) {
	objectList := &api.List{Items: objects}
	converted, err := resource.TryConvert(api.Scheme, objectList, version, api.Registry.GroupOrDie(api.GroupName).GroupVersion)
	if err != nil {
		return nil, err
	}
	return converted, nil
}

// IsSiblingCommandExists receives a pointer to a cobra command and a target string.
// Returns true if the target string is found in the list of sibling commands.
func IsSiblingCommandExists(cmd *cobra.Command, targetCmdName string) bool {
	for _, c := range cmd.Parent().Commands() {
		if c.Name() == targetCmdName {
			return true
		}
	}

	return false
}

// DefaultSubCommandRun prints a command's help string to the specified output if no
// arguments (sub-commands) are provided, or a usage error otherwise.
func DefaultSubCommandRun(out io.Writer) func(c *cobra.Command, args []string) {
	return func(c *cobra.Command, args []string) {
		c.SetOutput(out)
		RequireNoArguments(c, args)
		c.Help()
	}
}

// RequireNoArguments exits with a usage error if extra arguments are provided.
func RequireNoArguments(c *cobra.Command, args []string) {
	if len(args) > 0 {
		CheckErr(UsageError(c, fmt.Sprintf(`unknown command %q`, strings.Join(args, " "))))
	}
}

// OutputsRawFormat determines if a command's output format is machine parsable
// or returns false if it is human readable (name, wide, etc.)
func OutputsRawFormat(cmd *cobra.Command) bool {
	output := GetFlagString(cmd, "output")
	if output == "json" ||
		output == "yaml" ||
		output == "go-template" ||
		output == "go-template-file" ||
		output == "jsonpath" ||
		output == "jsonpath-file" {
		return true
	}

	return false
}

// StripComments will transform a YAML file into JSON, thus dropping any comments
// in it. Note that if the given file has a syntax error, the transformation will
// fail and we will manually drop all comments from the file.
func StripComments(file []byte) []byte {
	stripped := file
	stripped, err := yaml.ToJSON(stripped)
	if err != nil {
		stripped = ManualStrip(file)
	}
	return stripped
}

// ManualStrip is used for dropping comments from a YAML file
func ManualStrip(file []byte) []byte {
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
