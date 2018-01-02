// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// generator generates Protocol Buffer models and support code from
// JSON Schemas. It is used to generate representations of the
// OpenAPI Specification and vendor and specification extensions
// that are added by third-party OpenAPI authors.
package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"runtime"
	"strings"

	"github.com/googleapis/gnostic/jsonschema"
)

// License is the software license applied to generated code.
const License = "" +
	"// Copyright 2017 Google Inc. All Rights Reserved.\n" +
	"//\n" +
	"// Licensed under the Apache License, Version 2.0 (the \"License\");\n" +
	"// you may not use this file except in compliance with the License.\n" +
	"// You may obtain a copy of the License at\n" +
	"//\n" +
	"//    http://www.apache.org/licenses/LICENSE-2.0\n" +
	"//\n" +
	"// Unless required by applicable law or agreed to in writing, software\n" +
	"// distributed under the License is distributed on an \"AS IS\" BASIS,\n" +
	"// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n" +
	"// See the License for the specific language governing permissions and\n" +
	"// limitations under the License.\n"

func protoOptions(packageName string) []ProtoOption {
	return []ProtoOption{
		ProtoOption{
			Name:  "java_multiple_files",
			Value: "true",
			Comment: "// This option lets the proto compiler generate Java code inside the package\n" +
				"// name (see below) instead of inside an outer class. It creates a simpler\n" +
				"// developer experience by reducing one-level of name nesting and be\n" +
				"// consistent with most programming languages that don't support outer classes.",
		},

		ProtoOption{
			Name:  "java_outer_classname",
			Value: "OpenAPIProto",
			Comment: "// The Java outer classname should be the filename in UpperCamelCase. This\n" +
				"// class is only used to hold proto descriptor, so developers don't need to\n" +
				"// work with it directly.",
		},

		ProtoOption{
			Name:    "java_package",
			Value:   "org." + packageName,
			Comment: "// The Java package name must be proto package name with proper prefix.",
		},

		ProtoOption{
			Name:  "objc_class_prefix",
			Value: "OAS",
			Comment: "// A reasonable prefix for the Objective-C symbols generated from the package.\n" +
				"// It should at a minimum be 3 characters long, all uppercase, and convention\n" +
				"// is to use an abbreviation of the package name. Something short, but\n" +
				"// hopefully unique enough to not conflict with things that may come along in\n" +
				"// the future. 'GPB' is reserved for the protocol buffer implementation itself.",
		},
	}
}

func generateOpenAPIModel(version string) error {
	var input string
	var filename string
	var protoPackageName string

	switch version {
	case "v2":
		input = "openapi-2.0.json"
		filename = "OpenAPIv2"
		protoPackageName = "openapi.v2"
	case "v3":
		input = "openapi-3.0.json"
		filename = "OpenAPIv3"
		protoPackageName = "openapi.v3"
	default:
		return fmt.Errorf("Unknown OpenAPI version %s", version)
	}

	goPackageName := strings.Replace(protoPackageName, ".", "_", -1)

	projectRoot := os.Getenv("GOPATH") + "/src/github.com/googleapis/gnostic/"

	baseSchema, err := jsonschema.NewSchemaFromFile(projectRoot + "jsonschema/schema.json")
	if err != nil {
		return err
	}
	baseSchema.ResolveRefs()
	baseSchema.ResolveAllOfs()

	openapiSchema, err := jsonschema.NewSchemaFromFile(projectRoot + filename + "/" + input)
	if err != nil {
		return err
	}
	openapiSchema.ResolveRefs()
	openapiSchema.ResolveAllOfs()

	// build a simplified model of the types described by the schema
	cc := NewDomain(openapiSchema, version)
	// generators will map these patterns to the associated property names
	// these pattern names are a bit of a hack until we find a more automated way to obtain them

	switch version {
	case "v2":
		cc.TypeNameOverrides = map[string]string{
			"VendorExtension": "Any",
		}
		cc.PropertyNameOverrides = map[string]string{
			"PathItem":      "Path",
			"ResponseValue": "ResponseCode",
		}
	case "v3":
		cc.TypeNameOverrides = map[string]string{
			"SpecificationExtension": "Any",
		}
		cc.PropertyNameOverrides = map[string]string{
			"PathItem":      "Path",
			"ResponseValue": "ResponseCode",
		}
	default:
		return fmt.Errorf("Unknown OpenAPI version %s", version)
	}

	err = cc.Build()
	if err != nil {
		return err
	}

	if true {
		log.Printf("Type Model:\n%s", cc.Description())
	}

	// ensure that the target directory exists
	err = os.MkdirAll(projectRoot+filename, 0755)
	if err != nil {
		return err
	}

	// generate the protocol buffer description
	log.Printf("Generating protocol buffer description")
	proto := cc.generateProto(protoPackageName, License,
		protoOptions(goPackageName), []string{"google/protobuf/any.proto"})
	protoFileName := projectRoot + filename + "/" + filename + ".proto"
	err = ioutil.WriteFile(protoFileName, []byte(proto), 0644)
	if err != nil {
		return err
	}

	// generate the compiler
	log.Printf("Generating compiler support code")
	compiler := cc.GenerateCompiler(goPackageName, License, []string{
		"fmt",
		"gopkg.in/yaml.v2",
		"strings",
		"regexp",
		"github.com/googleapis/gnostic/compiler",
	})
	goFileName := projectRoot + filename + "/" + filename + ".go"
	err = ioutil.WriteFile(goFileName, []byte(compiler), 0644)
	if err != nil {
		return err
	}
	// format the compiler
	log.Printf("Formatting compiler support code")
	return exec.Command(runtime.GOROOT()+"/bin/gofmt", "-w", goFileName).Run()
}

func usage() string {
	return fmt.Sprintf(`
Usage: %s [OPTIONS]
Options:
  --v2
    Generate Protocol Buffer representation and support code for OpenAPI v2.
    Files are read from and written to appropriate locations in the gnostic
    project directory.
  --v3
    Generate Protocol Buffer representation and support code for OpenAPI v3
    Files are read from and written to appropriate locations in the gnostic
    project directory.
  --extension EXTENSION_SCHEMA [EXTENSIONOPTIONS]
    Generate a gnostic extension that reads a set of OpenAPI extensions.
    EXTENSION_SCHEMA is the json schema for the OpenAPI extensions to be
    supported.
    EXTENSION_OPTIONS
      --out_dir=PATH: Location for writing extension models and support code.
`, path.Base(os.Args[0]))
}

func main() {
	var openapiVersion = ""
	var generateExtensions = false

	for i, arg := range os.Args {
		if i == 0 {
			continue // skip the tool name
		}
		if arg == "--v2" {
			openapiVersion = "v2"
		} else if arg == "--v3" {
			openapiVersion = "v3"
		} else if arg == "--extension" {
			generateExtensions = true
			break
		} else {
			fmt.Printf("Unknown option: %s.\n%s\n", arg, usage())
			os.Exit(-1)
		}
	}

	if openapiVersion != "" {
		err := generateOpenAPIModel(openapiVersion)
		if err != nil {
			fmt.Printf("%+v\n", err)
		}
	} else if generateExtensions {
		err := processExtensionGenCommandline(usage())
		if err != nil {
			fmt.Printf("%+v\n", err)
		}
	} else {
		fmt.Printf("%s\n", usage())
	}
}
