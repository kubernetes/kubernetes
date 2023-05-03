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

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	yaml "gopkg.in/yaml.v2"
)

type EditTestCase struct {
	Description string `yaml:"description"`
	// create or edit
	Mode             string   `yaml:"mode"`
	Args             []string `yaml:"args"`
	Filename         string   `yaml:"filename"`
	Output           string   `yaml:"outputFormat"`
	Namespace        string   `yaml:"namespace"`
	ExpectedStdout   []string `yaml:"expectedStdout"`
	ExpectedStderr   []string `yaml:"expectedStderr"`
	ExpectedExitCode int      `yaml:"expectedExitCode"`

	Steps []EditStep `yaml:"steps"`
}

type EditStep struct {
	// edit or request
	StepType string `yaml:"type"`

	// only applies to request
	RequestMethod      string `yaml:"expectedMethod,omitempty"`
	RequestPath        string `yaml:"expectedPath,omitempty"`
	RequestContentType string `yaml:"expectedContentType,omitempty"`
	Input              string `yaml:"expectedInput"`

	// only applies to request
	ResponseStatusCode int `yaml:"resultingStatusCode,omitempty"`

	Output string `yaml:"resultingOutput"`
}

func main() {
	tc := &EditTestCase{
		Description:    "add a testcase description",
		Mode:           "edit",
		Args:           []string{"set", "args"},
		ExpectedStdout: []string{"expected stdout substring"},
		ExpectedStderr: []string{"expected stderr substring"},
	}

	var currentStep *EditStep

	fmt.Println(http.ListenAndServe(":8081", http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {

		// Record non-discovery things
		record := false
		switch segments := strings.Split(strings.Trim(req.URL.Path, "/"), "/"); segments[0] {
		case "api":
			// api, version
			record = len(segments) > 2
		case "apis":
			// apis, group, version
			record = len(segments) > 3
		case "callback":
			record = true
		}

		body, err := io.ReadAll(req.Body)
		checkErr(err)

		switch m, p := req.Method, req.URL.Path; {
		case m == "POST" && p == "/callback/in":
			if currentStep != nil {
				panic("cannot post input with step already in progress")
			}
			filename := fmt.Sprintf("%d.original", len(tc.Steps))
			checkErr(os.WriteFile(filename, body, os.FileMode(0755)))
			currentStep = &EditStep{StepType: "edit", Input: filename}
		case m == "POST" && p == "/callback/out":
			if currentStep == nil || currentStep.StepType != "edit" {
				panic("cannot post output without posting input first")
			}
			filename := fmt.Sprintf("%d.edited", len(tc.Steps))
			checkErr(os.WriteFile(filename, body, os.FileMode(0755)))
			currentStep.Output = filename
			tc.Steps = append(tc.Steps, *currentStep)
			currentStep = nil
		default:
			if currentStep != nil {
				panic("cannot make request with step already in progress")
			}

			urlCopy := *req.URL
			urlCopy.Host = "localhost:8080"
			urlCopy.Scheme = "http"
			proxiedReq, err := http.NewRequest(req.Method, urlCopy.String(), bytes.NewReader(body))
			checkErr(err)
			proxiedReq.Header = req.Header
			resp, err := http.DefaultClient.Do(proxiedReq)
			checkErr(err)
			defer resp.Body.Close()

			bodyOut, err := io.ReadAll(resp.Body)
			checkErr(err)

			for k, vs := range resp.Header {
				for _, v := range vs {
					w.Header().Add(k, v)
				}
			}
			w.WriteHeader(resp.StatusCode)
			w.Write(bodyOut)

			if record {
				infile := fmt.Sprintf("%d.request", len(tc.Steps))
				outfile := fmt.Sprintf("%d.response", len(tc.Steps))
				checkErr(os.WriteFile(infile, tryIndent(body), os.FileMode(0755)))
				checkErr(os.WriteFile(outfile, tryIndent(bodyOut), os.FileMode(0755)))
				tc.Steps = append(tc.Steps, EditStep{
					StepType:           "request",
					Input:              infile,
					Output:             outfile,
					RequestContentType: req.Header.Get("Content-Type"),
					RequestMethod:      req.Method,
					RequestPath:        req.URL.Path,
					ResponseStatusCode: resp.StatusCode,
				})
			}
		}

		tcData, err := yaml.Marshal(tc)
		checkErr(err)
		checkErr(os.WriteFile("test.yaml", tcData, os.FileMode(0755)))
	})))
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}

func tryIndent(data []byte) []byte {
	indented := &bytes.Buffer{}
	if err := json.Indent(indented, data, "", "\t"); err == nil {
		return indented.Bytes()
	}
	return data
}
