/*
Copyright 2014 Google Inc. All rights reserved.

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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/golang/glog"
	"github.com/imdario/mergo"
	"github.com/spf13/cobra"
)

func GetFlagString(cmd *cobra.Command, flag string) string {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	return f.Value.String()
}

func GetFlagBool(cmd *cobra.Command, flag string) bool {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	// Caseless compare.
	if strings.ToLower(f.Value.String()) == "true" {
		return true
	}
	return false
}

// Returns nil if the flag wasn't set.
func GetFlagBoolPtr(cmd *cobra.Command, flag string) *bool {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	// Check if flag was not set at all.
	if !f.Changed && f.DefValue == f.Value.String() {
		return nil
	}
	var ret bool
	// Caseless compare.
	if strings.ToLower(f.Value.String()) == "true" {
		ret = true
	} else {
		ret = false
	}
	return &ret
}

// Assumes the flag has a default value.
func GetFlagInt(cmd *cobra.Command, flag string) int {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	v, err := strconv.Atoi(f.Value.String())
	// This is likely not a sufficiently friendly error message, but cobra
	// should prevent non-integer values from reaching here.
	checkErr(err)
	return v
}

func GetFlagDuration(cmd *cobra.Command, flag string) time.Duration {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	v, err := time.ParseDuration(f.Value.String())
	checkErr(err)
	return v
}

// Returns the first non-empty string out of the ones provided. If all
// strings are empty, returns an empty string.
func FirstNonEmptyString(args ...string) string {
	for _, s := range args {
		if len(s) > 0 {
			return s
		}
	}
	return ""
}

// Return a list of file names of a certain type within a given directory.
// TODO: replace with resource.Builder
func GetFilesFromDir(directory string, fileType string) []string {
	files := []string{}

	err := filepath.Walk(directory, func(path string, f os.FileInfo, err error) error {
		if filepath.Ext(path) == fileType {
			files = append(files, path)
		}
		return err
	})

	checkErr(err)
	return files
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
		data, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			return nil, err
		}

		if len(data) == 0 {
			return nil, fmt.Errorf(`Read from stdin specified ("-") but no data found`)
		}

		return data, nil
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
		data, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("unable to read URL %s: %v\n", location, err)
		}
		return data, nil
	} else {
		data, err := ioutil.ReadFile(location)
		if err != nil {
			return nil, fmt.Errorf("unable to read %s: %v\n", location, err)
		}
		return data, nil
	}
}

func Merge(dst runtime.Object, fragment, kind string) error {
	// Ok, this is a little hairy, we'd rather not force the user to specify a kind for their JSON
	// So we pull it into a map, add the Kind field, and then reserialize.
	// We also pull the apiVersion for proper parsing
	var intermediate interface{}
	if err := json.Unmarshal([]byte(fragment), &intermediate); err != nil {
		return err
	}
	dataMap, ok := intermediate.(map[string]interface{})
	if !ok {
		return fmt.Errorf("Expected a map, found something else: %s", fragment)
	}
	version, found := dataMap["apiVersion"]
	if !found {
		return fmt.Errorf("Inline JSON requires an apiVersion field")
	}
	versionString, ok := version.(string)
	if !ok {
		return fmt.Errorf("apiVersion must be a string")
	}
	codec := runtime.CodecFor(api.Scheme, versionString)

	intermediate.(map[string]interface{})["kind"] = kind
	data, err := json.Marshal(intermediate)
	if err != nil {
		return err
	}
	src, err := codec.Decode(data)
	if err != nil {
		return err
	}
	return mergo.Merge(dst, src)
}
