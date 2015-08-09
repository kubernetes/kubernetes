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

	"github.com/evanphx/json-patch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

type debugError interface {
	DebugError() (msg string, args []interface{})
}

// AddSourceToErr adds handleResourcePrefix and source string to error message.
// verb is the string like "creating", "deleting" etc.
// souce is the filename or URL to the template file(*.json or *.yaml), or stdin to use to handle the resource.
func AddSourceToErr(verb string, source string, err error) error {
	if source != "" {
		if statusError, ok := err.(*errors.StatusError); ok {
			status := statusError.Status()
			status.Message = fmt.Sprintf("error when %s %q: %v", verb, source, status.Message)
			return &errors.StatusError{status}
		}
		return fmt.Errorf("error when %s %q: %v", verb, source, err)
	}
	return err
}

// CheckErr prints a user friendly error to STDERR and exits with a non-zero
// exit code. Unrecognized errors will be printed with an "error: " prefix.
//
// This method is generic to the command in use and may be used by non-Kubectl
// commands.
func CheckErr(err error) {
	checkErr(err, fatal)
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

	// handle multiline errors
	if clientcmd.IsConfigurationInvalid(err) {
		handleErr(MultilineError("Error in configuration: ", err))
	}
	if agg, ok := err.(utilerrors.Aggregate); ok && len(agg.Errors()) > 0 {
		handleErr(MultipleErrors("", agg.Errors()))
	}

	msg, ok := StandardErrorMessage(err)
	if !ok {
		msg = fmt.Sprintf("error: %s", err.Error())
	}
	handleErr(msg)
}

func statusCausesToAggrError(scs []api.StatusCause) utilerrors.Aggregate {
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
	_, isStatus := err.(client.APIStatus)
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

func UsageError(cmd *cobra.Command, format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s\nsee '%s -h' for help.", msg, cmd.CommandPath())
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
		glog.Fatalf("err %v accessing flag %s for command %s: %s", err, flag, cmd.Name())
	}
	return s
}

// GetFlagStringList can be used to accept multiple argument with flag repetition (e.g. -f arg1 -f arg2 ...)
func GetFlagStringSlice(cmd *cobra.Command, flag string) []string {
	s, err := cmd.Flags().GetStringSlice(flag)
	if err != nil {
		glog.Fatalf("err %v accessing flag %s for command %s: %s", err, flag, cmd.Name())
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
		glog.Fatalf("err %v accessing flag %s for command %s: %s", err, flag, cmd.Name())
	}
	return b
}

// Assumes the flag has a default value.
func GetFlagInt(cmd *cobra.Command, flag string) int {
	i, err := cmd.Flags().GetInt(flag)
	if err != nil {
		glog.Fatalf("err: %v accessing flag %s for command %s: %s", err, flag, cmd.Name())
	}
	return i
}

func GetFlagDuration(cmd *cobra.Command, flag string) time.Duration {
	d, err := cmd.Flags().GetDuration(flag)
	if err != nil {
		glog.Fatalf("err: %v accessing flag %s for command %s: %s", err, flag, cmd.Name())
	}
	return d
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

func Merge(dst runtime.Object, fragment, kind string) (runtime.Object, error) {
	// Ok, this is a little hairy, we'd rather not force the user to specify a kind for their JSON
	// So we pull it into a map, add the Kind field, and then reserialize.
	// We also pull the apiVersion for proper parsing
	var intermediate interface{}
	if err := json.Unmarshal([]byte(fragment), &intermediate); err != nil {
		return nil, err
	}
	dataMap, ok := intermediate.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("Expected a map, found something else: %s", fragment)
	}
	version, found := dataMap["apiVersion"]
	if !found {
		return nil, fmt.Errorf("Inline JSON requires an apiVersion field")
	}
	versionString, ok := version.(string)
	if !ok {
		return nil, fmt.Errorf("apiVersion must be a string")
	}
	i, err := latest.InterfacesFor(versionString)
	if err != nil {
		return nil, err
	}

	// encode dst into versioned json and apply fragment directly too it
	target, err := i.Codec.Encode(dst)
	if err != nil {
		return nil, err
	}
	patched, err := jsonpatch.MergePatch(target, []byte(fragment))
	if err != nil {
		return nil, err
	}
	out, err := i.Codec.Decode(patched)
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
func UpdateObject(info *resource.Info, updateFn func(runtime.Object) error) (runtime.Object, error) {
	helper := resource.NewHelper(info.Client, info.Mapping)

	err := updateFn(info.Object)
	if err != nil {
		return nil, err
	}
	data, err := helper.Codec.Encode(info.Object)
	if err != nil {
		return nil, err
	}

	_, err = helper.Replace(info.Namespace, info.Name, true, data)
	if err != nil {
		return nil, err
	}
	return info.Object, nil
}
