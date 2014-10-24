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

package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/golang/glog"
	"gopkg.in/v1/yaml"
)

const usage = "usage: enscope specFilename configFilename"

func checkErr(err error) {
	if err != nil {
		glog.Fatalf("%v", err)
	}
}

// TODO: If name suffix is not specified, deterministically generate it by hashing the labels.

type EnscopeSpec struct {
	NameSuffix string            `json:"nameSuffix,omitempty" yaml:"nameSuffix,omitempty"`
	Labels     map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}

func main() {
	if len(os.Args) != 3 {
		checkErr(fmt.Errorf(usage))
	}
	specFilename := os.Args[1]
	configFilename := os.Args[2]

	specData, err := ReadConfigData(specFilename)
	checkErr(err)

	spec := EnscopeSpec{}
	err = yaml.Unmarshal(specData, &spec)
	checkErr(err)

	configData, err := ReadConfigData(configFilename)
	checkErr(err)

	var data interface{}

	err = yaml.Unmarshal([]byte(configData), &data)
	checkErr(err)

	xData, err := enscope("", spec, data)
	checkErr(err)

	out, err := yaml.Marshal(xData)
	checkErr(err)

	fmt.Print(string(out))
}

func enscope(parent string, spec EnscopeSpec, in interface{}) (out interface{}, err error) {
	var ok bool
	switch in.(type) {
	case map[interface{}]interface{}:
		o := make(map[interface{}]interface{})
		for k, v := range in.(map[interface{}]interface{}) {
			var kstring string
			if kstring, ok = k.(string); !ok {
				kstring = parent
			}
			v, err = enscope(kstring, spec, v)
			if err != nil {
				return nil, err
			}
			o[k] = v
		}
		var ifc interface{}
		var name string
		// TODO: Figure out a more general way to identify references
		if parent == "metadata" || parent == "template" {
			if ifc, ok = o["name"]; ok {
				if name, ok = ifc.(string); ok {
					o["name"] = name + spec.NameSuffix
				}
			}
			if ifc, ok = o["labels"]; ok {
				var labels map[interface{}]interface{}
				if labels, ok = ifc.(map[interface{}]interface{}); ok {
					for k, v := range spec.Labels {
						labels[k] = v
					}
					o["labels"] = labels
				}
			}
		}
		if parent == "spec" {
			// Note that nodeSelector doesn't match, so we won't modify it
			if ifc, ok = o["selector"]; ok {
				var selector map[interface{}]interface{}
				if selector, ok = ifc.(map[interface{}]interface{}); ok {
					for k, v := range spec.Labels {
						selector[k] = v
					}
					o["selector"] = selector
				}
			}
		}
		return o, nil
	case []interface{}:
		in1 := in.([]interface{})
		len1 := len(in1)
		o := make([]interface{}, len1)
		for i := 0; i < len1; i++ {
			o[i], err = enscope(parent, spec, in1[i])
			if err != nil {
				return nil, err
			}
		}
		return o, nil
	default:
		return in, nil
	}
	return in, nil
}

//////////////////////////////////////////////////////////////////////

// Client tool utility functions copied from kubectl, kubecfg, and podex.
// This should probably be a separate package, but the right solution is
// to refactor the copied code and delete it from here.

func ReadConfigData(location string) ([]byte, error) {
	if len(location) == 0 {
		return nil, fmt.Errorf("Location given but empty")
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
	return readConfigDataFromLocation(location)
}

func readConfigDataFromLocation(location string) ([]byte, error) {
	// we look for http:// or https:// to determine if valid URL, otherwise do normal file IO
	if strings.Index(location, "http://") == 0 || strings.Index(location, "https://") == 0 {
		resp, err := http.Get(location)
		if err != nil {
			return nil, fmt.Errorf("Unable to access URL %s: %v\n", location, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("Unable to read URL, server reported %d %s", resp.StatusCode, resp.Status)
		}
		data, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("Unable to read URL %s: %v\n", location, err)
		}
		return data, nil
	} else {
		data, err := ioutil.ReadFile(location)
		if err != nil {
			return nil, fmt.Errorf("Unable to read %s: %v\n", location, err)
		}
		return data, nil
	}
}
