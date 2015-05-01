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

// kube-version-change is a simple utility for converting a
// kubernetes object into a different api version.
package main

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"os"
	"runtime"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/ghodss/yaml"
	flag "github.com/spf13/pflag"
)

var (
	inputSource   = flag.StringP("input", "i", "-", "Input source; '-' means stdin")
	outputDest    = flag.StringP("output", "o", "-", "Output destination; '-' means stdout")
	rewrite       = flag.StringP("rewrite", "r", "", "If nonempty, use this as both input and output.")
	outputVersion = flag.StringP("out-version", "v", "v1beta3", "Version to convert input to")
)

// isYAML determines whether data is JSON or YAML formatted by seeing
// if it will parse as json.
func isYAML(data []byte) bool {
	var unused interface{}
	if err := json.Unmarshal(data, &unused); err != nil {
		return true
	}
	return false
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	flag.CommandLine.SetNormalizeFunc(util.WordSepNormalizeFunc)
	flag.Parse()

	if *rewrite != "" {
		*inputSource = *rewrite
		*outputDest = *rewrite
	}

	var in io.Reader
	if *inputSource == "-" {
		in = os.Stdin
	} else {
		f, err := os.Open(*inputSource)
		if err != nil {
			log.Fatalf("Couldn't open %q: %q", *inputSource, err)
		}
		defer f.Close()
		in = f
	}

	data, err := ioutil.ReadAll(in)
	if err != nil {
		log.Fatalf("Couldn't read from input: %q", err)
	}
	isYAML := isYAML(data)

	if isYAML {
		data, err = yaml.YAMLToJSON(data)
		if err != nil {
			log.Fatalf("Failed to convert YAML to JSON: %q", err)
		}
	}
	obj, err := api.Scheme.Decode(data)
	if err != nil {
		log.Fatalf("Couldn't decode input: %q", err)
	}

	outData, err := api.Scheme.EncodeToVersion(obj, *outputVersion)
	if err != nil {
		log.Fatalf("Failed to encode to version %q: %q", *outputVersion, err)
	}

	if isYAML {
		outData, err = yaml.JSONToYAML(outData)
		if err != nil {
			log.Fatalf("Failed to convert to YAML: %q", err)
		}
	} else if true {
		// TODO: figure out if input JSON was pretty.
		var buf bytes.Buffer
		err = json.Indent(&buf, outData, "", "  ")
		if err != nil {
			log.Fatalf("Failed to indent JSON: %q", err)
		}
		outData = buf.Bytes()
	}

	var out io.Writer
	if *outputDest == "-" {
		out = os.Stdout
	} else {
		f, err := os.Create(*outputDest)
		if err != nil {
			log.Fatalf("Couldn't open %q: %q", *outputDest, err)
		}
		defer f.Close()
		out = f
	}

	if _, err = out.Write(outData); err != nil {
		log.Fatalf("Failed to write: %q", err)
	}
}
