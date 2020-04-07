/*
Copyright 2020 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"regexp"

	"gopkg.in/yaml.v2"

	"k8s.io/kubernetes/test/conformance/behaviors"
)

func link(o *options) error {
	var behaviorFiles []string
	behaviorsMapping := make(map[string][]string)
	var conformanceDataList []behaviors.ConformanceData

	err := filepath.Walk(o.behaviorsDir,
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			r, _ := regexp.Compile(".+.yaml$")
			if r.MatchString(path) {
				behaviorFiles = append(behaviorFiles, path)
			}
			return nil
		})
	if err != nil {
		return err
	}
	fmt.Println()
	fmt.Printf("Using behaviors from these %d files:\n", len(behaviorFiles))
	for _, f := range behaviorFiles {
		fmt.Println("    ", f)
	}
	fmt.Println()
	if o.listAll {
		fmt.Println("All behaviors:")
	} else {
		fmt.Println("Behaviors not covered by any conformance test:")
	}

	for _, behaviorFile := range behaviorFiles {
		var suite behaviors.Suite

		yamlFile, err := ioutil.ReadFile(behaviorFile)
		if err != nil {
			return err
		}
		err = yaml.UnmarshalStrict(yamlFile, &suite)
		if err != nil {
			return err
		}

		for _, behavior := range suite.Behaviors {
			behaviorsMapping[behavior.ID] = nil
		}
	}

	conformanceYaml, err := ioutil.ReadFile(o.testdata)
	if err != nil {
		return err
	}

	err = yaml.Unmarshal(conformanceYaml, &conformanceDataList)
	if err != nil {
		return err
	}

	for _, data := range conformanceDataList {
		for _, behaviorID := range data.Behaviors {
			if _, ok := behaviorsMapping[behaviorID]; !ok {
				return fmt.Errorf("cannot find behavior %q", behaviorID)
			}
			behaviorsMapping[behaviorID] = append(behaviorsMapping[behaviorID], data.CodeName)
		}
	}
	printBehaviorsMapping(behaviorsMapping, o)
	return nil
}

func printBehaviorsMapping(behaviorsMapping map[string][]string, o *options) {
	for behaviorID, tests := range behaviorsMapping {
		if o.listAll || tests == nil {
			fmt.Println("    ", behaviorID)
		}
	}
}
