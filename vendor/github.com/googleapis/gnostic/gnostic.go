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

//go:generate ./COMPILE-PROTOS.sh

// Gnostic is a tool for building better REST APIs through knowledge.
//
// Gnostic reads declarative descriptions of REST APIs that conform
// to the OpenAPI Specification, reports errors, resolves internal
// dependencies, and puts the results in a binary form that can
// be used in any language that is supported by the Protocol Buffer
// tools.
//
// Gnostic models are validated and typed. This allows API tool
// developers to focus on their product and not worry about input
// validation and type checking.
//
// Gnostic calls plugins that implement a variety of API implementation
// and support features including generation of client and server
// support code.
package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/OpenAPIv3"
	"github.com/googleapis/gnostic/compiler"
	"github.com/googleapis/gnostic/jsonwriter"
	plugins "github.com/googleapis/gnostic/plugins"
	"gopkg.in/yaml.v2"
)

const ( // OpenAPI Version
	openAPIvUnknown = 0
	openAPIv2       = 2
	openAPIv3       = 3
)

// Determine the version of an OpenAPI description read from JSON or YAML.
func getOpenAPIVersionFromInfo(info interface{}) int {
	m, ok := compiler.UnpackMap(info)
	if !ok {
		return openAPIvUnknown
	}
	swagger, ok := compiler.MapValueForKey(m, "swagger").(string)
	if ok && strings.HasPrefix(swagger, "2.0") {
		return openAPIv2
	}
	openapi, ok := compiler.MapValueForKey(m, "openapi").(string)
	if ok && strings.HasPrefix(openapi, "3.0") {
		return openAPIv3
	}
	return openAPIvUnknown
}

const (
	pluginPrefix    = "gnostic-"
	extensionPrefix = "gnostic-x-"
)

type pluginCall struct {
	Name       string
	Invocation string
}

// Invokes a plugin.
func (p *pluginCall) perform(document proto.Message, openAPIVersion int, sourceName string) error {
	if p.Name != "" {
		request := &plugins.Request{}

		// Infer the name of the executable by adding the prefix.
		executableName := pluginPrefix + p.Name

		// Validate invocation string with regular expression.
		invocation := p.Invocation

		//
		// Plugin invocations must consist of
		// zero or more comma-separated key=value pairs followed by a path.
		// If pairs are present, a colon separates them from the path.
		// Keys and values must be alphanumeric strings and may contain
		// dashes, underscores, periods, or forward slashes.
		// A path can contain any characters other than the separators ',', ':', and '='.
		//
		invocationRegex := regexp.MustCompile(`^([\w-_\/\.]+=[\w-_\/\.]+(,[\w-_\/\.]+=[\w-_\/\.]+)*:)?[^,:=]+$`)
		if !invocationRegex.Match([]byte(p.Invocation)) {
			return fmt.Errorf("Invalid invocation of %s: %s", executableName, invocation)
		}

		invocationParts := strings.Split(p.Invocation, ":")
		var outputLocation string
		switch len(invocationParts) {
		case 1:
			outputLocation = invocationParts[0]
		case 2:
			parameters := strings.Split(invocationParts[0], ",")
			for _, keyvalue := range parameters {
				pair := strings.Split(keyvalue, "=")
				if len(pair) == 2 {
					request.Parameters = append(request.Parameters, &plugins.Parameter{Name: pair[0], Value: pair[1]})
				}
			}
			outputLocation = invocationParts[1]
		default:
			// badly-formed request
			outputLocation = invocationParts[len(invocationParts)-1]
		}

		version := &plugins.Version{}
		version.Major = 0
		version.Minor = 1
		version.Patch = 0
		request.CompilerVersion = version

		request.OutputPath = outputLocation

		wrapper := &plugins.Wrapper{}
		wrapper.Name = sourceName
		switch openAPIVersion {
		case openAPIv2:
			wrapper.Version = "v2"
		case openAPIv3:
			wrapper.Version = "v3"
		default:
			wrapper.Version = "unknown"
		}
		protoBytes, _ := proto.Marshal(document)
		wrapper.Value = protoBytes
		request.Wrapper = wrapper
		requestBytes, _ := proto.Marshal(request)

		cmd := exec.Command(executableName)
		cmd.Stdin = bytes.NewReader(requestBytes)
		cmd.Stderr = os.Stderr
		output, err := cmd.Output()
		if err != nil {
			return err
		}
		response := &plugins.Response{}
		err = proto.Unmarshal(output, response)
		if err != nil {
			return err
		}

		if response.Errors != nil {
			return fmt.Errorf("Plugin error: %+v", response.Errors)
		}

		// Write files to the specified directory.
		var writer io.Writer
		switch {
		case outputLocation == "!":
			// Write nothing.
		case outputLocation == "-":
			writer = os.Stdout
			for _, file := range response.Files {
				writer.Write([]byte("\n\n" + file.Name + " -------------------- \n"))
				writer.Write(file.Data)
			}
		case isFile(outputLocation):
			return fmt.Errorf("unable to overwrite %s", outputLocation)
		default: // write files into a directory named by outputLocation
			if !isDirectory(outputLocation) {
				os.Mkdir(outputLocation, 0755)
			}
			for _, file := range response.Files {
				p := outputLocation + "/" + file.Name
				dir := path.Dir(p)
				os.MkdirAll(dir, 0755)
				f, _ := os.Create(p)
				defer f.Close()
				f.Write(file.Data)
			}
		}
	}
	return nil
}

func isFile(path string) bool {
	fileInfo, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !fileInfo.IsDir()
}

func isDirectory(path string) bool {
	fileInfo, err := os.Stat(path)
	if err != nil {
		return false
	}
	return fileInfo.IsDir()
}

// Write bytes to a named file.
// Certain names have special meaning:
//   ! writes nothing
//   - writes to stdout
//   = writes to stderr
// If a directory name is given, the file is written there with
// a name derived from the source and extension arguments.
func writeFile(name string, bytes []byte, source string, extension string) {
	var writer io.Writer
	if name == "!" {
		return
	} else if name == "-" {
		writer = os.Stdout
	} else if name == "=" {
		writer = os.Stderr
	} else if isDirectory(name) {
		base := filepath.Base(source)
		// Remove the original source extension.
		base = base[0 : len(base)-len(filepath.Ext(base))]
		// Build the path that puts the result in the passed-in directory.
		filename := name + "/" + base + "." + extension
		file, _ := os.Create(filename)
		defer file.Close()
		writer = file
	} else {
		file, _ := os.Create(name)
		defer file.Close()
		writer = file
	}
	writer.Write(bytes)
	if name == "-" || name == "=" {
		writer.Write([]byte("\n"))
	}
}

// The Gnostic structure holds global state information for gnostic.
type Gnostic struct {
	usage             string
	sourceName        string
	binaryOutputPath  string
	textOutputPath    string
	yamlOutputPath    string
	jsonOutputPath    string
	errorOutputPath   string
	resolveReferences bool
	pluginCalls       []*pluginCall
	extensionHandlers []compiler.ExtensionHandler
	openAPIVersion    int
}

// Initialize a structure to store global application state.
func newGnostic() *Gnostic {
	g := &Gnostic{}
	// Option fields initialize to their default values.
	g.usage = `
Usage: gnostic OPENAPI_SOURCE [OPTIONS]
  OPENAPI_SOURCE is the filename or URL of an OpenAPI description to read.
Options:
  --pb-out=PATH       Write a binary proto to the specified location.
  --text-out=PATH     Write a text proto to the specified location.
  --json-out=PATH     Write a json API description to the specified location.
  --yaml-out=PATH     Write a yaml API description to the specified location.
  --errors-out=PATH   Write compilation errors to the specified location.
  --PLUGIN-out=PATH   Run the plugin named gnostic_PLUGIN and write results
                      to the specified location.
  --x-EXTENSION       Use the extension named gnostic-x-EXTENSION
                      to process OpenAPI specification extensions.
  --resolve-refs      Explicitly resolve $ref references.
                      This could have problems with recursive definitions.
`
	// Initialize internal structures.
	g.pluginCalls = make([]*pluginCall, 0)
	g.extensionHandlers = make([]compiler.ExtensionHandler, 0)
	return g
}

// Parse command-line options.
func (g *Gnostic) readOptions() {
	// plugin processing matches patterns of the form "--PLUGIN-out=PATH" and "--PLUGIN_out=PATH"
	pluginRegex := regexp.MustCompile("--(.+)[-_]out=(.+)")

	// extension processing matches patterns of the form "--x-EXTENSION"
	extensionRegex := regexp.MustCompile("--x-(.+)")

	for i, arg := range os.Args {
		if i == 0 {
			continue // skip the tool name
		}
		var m [][]byte
		if m = pluginRegex.FindSubmatch([]byte(arg)); m != nil {
			pluginName := string(m[1])
			invocation := string(m[2])
			switch pluginName {
			case "pb":
				g.binaryOutputPath = invocation
			case "text":
				g.textOutputPath = invocation
			case "json":
				g.jsonOutputPath = invocation
			case "yaml":
				g.yamlOutputPath = invocation
			case "errors":
				g.errorOutputPath = invocation
			default:
				p := &pluginCall{Name: pluginName, Invocation: invocation}
				g.pluginCalls = append(g.pluginCalls, p)
			}
		} else if m = extensionRegex.FindSubmatch([]byte(arg)); m != nil {
			extensionName := string(m[1])
			extensionHandler := compiler.ExtensionHandler{Name: extensionPrefix + extensionName}
			g.extensionHandlers = append(g.extensionHandlers, extensionHandler)
		} else if arg == "--resolve-refs" {
			g.resolveReferences = true
		} else if arg[0] == '-' {
			fmt.Fprintf(os.Stderr, "Unknown option: %s.\n%s\n", arg, g.usage)
			os.Exit(-1)
		} else {
			g.sourceName = arg
		}
	}
}

// Validate command-line options.
func (g *Gnostic) validateOptions() {
	if g.binaryOutputPath == "" &&
		g.textOutputPath == "" &&
		g.yamlOutputPath == "" &&
		g.jsonOutputPath == "" &&
		g.errorOutputPath == "" &&
		len(g.pluginCalls) == 0 {
		fmt.Fprintf(os.Stderr, "Missing output directives.\n%s\n", g.usage)
		os.Exit(-1)
	}
	if g.sourceName == "" {
		fmt.Fprintf(os.Stderr, "No input specified.\n%s\n", g.usage)
		os.Exit(-1)
	}
	// If we get here and the error output is unspecified, write errors to stderr.
	if g.errorOutputPath == "" {
		g.errorOutputPath = "="
	}
}

// Generate an error message to be written to stderr or a file.
func (g *Gnostic) errorBytes(err error) []byte {
	return []byte("Errors reading " + g.sourceName + "\n" + err.Error())
}

// Read an OpenAPI description from YAML or JSON.
func (g *Gnostic) readOpenAPIText(bytes []byte) (message proto.Message, err error) {
	info, err := compiler.ReadInfoFromBytes(g.sourceName, bytes)
	if err != nil {
		return nil, err
	}
	// Determine the OpenAPI version.
	g.openAPIVersion = getOpenAPIVersionFromInfo(info)
	if g.openAPIVersion == openAPIvUnknown {
		return nil, errors.New("unable to identify OpenAPI version")
	}
	// Compile to the proto model.
	if g.openAPIVersion == openAPIv2 {
		document, err := openapi_v2.NewDocument(info, compiler.NewContextWithExtensions("$root", nil, &g.extensionHandlers))
		if err != nil {
			return nil, err
		}
		message = document
	} else if g.openAPIVersion == openAPIv3 {
		document, err := openapi_v3.NewDocument(info, compiler.NewContextWithExtensions("$root", nil, &g.extensionHandlers))
		if err != nil {
			return nil, err
		}
		message = document
	}
	return message, err
}

// Read an OpenAPI binary file.
func (g *Gnostic) readOpenAPIBinary(data []byte) (message proto.Message, err error) {
	// try to read an OpenAPI v3 document
	documentV3 := &openapi_v3.Document{}
	err = proto.Unmarshal(data, documentV3)
	if err == nil && strings.HasPrefix(documentV3.Openapi, "3.0") {
		g.openAPIVersion = openAPIv3
		return documentV3, nil
	}
	// if that failed, try to read an OpenAPI v2 document
	documentV2 := &openapi_v2.Document{}
	err = proto.Unmarshal(data, documentV2)
	if err == nil && strings.HasPrefix(documentV2.Swagger, "2.0") {
		g.openAPIVersion = openAPIv2
		return documentV2, nil
	}
	return nil, err
}

// Write a binary pb representation.
func (g *Gnostic) writeBinaryOutput(message proto.Message) {
	protoBytes, err := proto.Marshal(message)
	if err != nil {
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		defer os.Exit(-1)
	} else {
		writeFile(g.binaryOutputPath, protoBytes, g.sourceName, "pb")
	}
}

// Write a text pb representation.
func (g *Gnostic) writeTextOutput(message proto.Message) {
	bytes := []byte(proto.MarshalTextString(message))
	writeFile(g.textOutputPath, bytes, g.sourceName, "text")
}

// Write JSON/YAML OpenAPI representations.
func (g *Gnostic) writeJSONYAMLOutput(message proto.Message) {
	// Convert the OpenAPI document into an exportable MapSlice.
	var rawInfo yaml.MapSlice
	var ok bool
	var err error
	if g.openAPIVersion == openAPIv2 {
		document := message.(*openapi_v2.Document)
		rawInfo, ok = document.ToRawInfo().(yaml.MapSlice)
		if !ok {
			rawInfo = nil
		}
	} else if g.openAPIVersion == openAPIv3 {
		document := message.(*openapi_v3.Document)
		rawInfo, ok = document.ToRawInfo().(yaml.MapSlice)
		if !ok {
			rawInfo = nil
		}
	}
	// Optionally write description in yaml format.
	if g.yamlOutputPath != "" {
		var bytes []byte
		if rawInfo != nil {
			bytes, err = yaml.Marshal(rawInfo)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error generating yaml output %s\n", err.Error())
			}
			writeFile(g.yamlOutputPath, bytes, g.sourceName, "yaml")
		} else {
			fmt.Fprintf(os.Stderr, "No yaml output available.\n")
		}
	}
	// Optionally write description in json format.
	if g.jsonOutputPath != "" {
		var bytes []byte
		if rawInfo != nil {
			bytes, _ = jsonwriter.Marshal(rawInfo)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error generating json output %s\n", err.Error())
			}
			writeFile(g.jsonOutputPath, bytes, g.sourceName, "json")
		} else {
			fmt.Fprintf(os.Stderr, "No json output available.\n")
		}
	}
}

// Perform all actions specified in the command-line options.
func (g *Gnostic) performActions(message proto.Message) (err error) {
	// Optionally resolve internal references.
	if g.resolveReferences {
		if g.openAPIVersion == openAPIv2 {
			document := message.(*openapi_v2.Document)
			_, err = document.ResolveReferences(g.sourceName)
		} else if g.openAPIVersion == openAPIv3 {
			document := message.(*openapi_v3.Document)
			_, err = document.ResolveReferences(g.sourceName)
		}
		if err != nil {
			return err
		}
	}
	// Optionally write proto in binary format.
	if g.binaryOutputPath != "" {
		g.writeBinaryOutput(message)
	}
	// Optionally write proto in text format.
	if g.textOutputPath != "" {
		g.writeTextOutput(message)
	}
	// Optionaly write document in yaml and/or json formats.
	if g.yamlOutputPath != "" || g.jsonOutputPath != "" {
		g.writeJSONYAMLOutput(message)
	}
	// Call all specified plugins.
	for _, p := range g.pluginCalls {
		err := p.perform(message, g.openAPIVersion, g.sourceName)
		if err != nil {
			writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
			defer os.Exit(-1) // run all plugins, even when some have errors
		}
	}
	return nil
}

func (g *Gnostic) main() {
	var err error
	g.readOptions()
	g.validateOptions()
	// Read the OpenAPI source.
	bytes, err := compiler.ReadBytesForFile(g.sourceName)
	if err != nil {
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		os.Exit(-1)
	}
	extension := strings.ToLower(filepath.Ext(g.sourceName))
	var message proto.Message
	if extension == ".json" || extension == ".yaml" {
		// Try to read the source as JSON/YAML.
		message, err = g.readOpenAPIText(bytes)
		if err != nil {
			writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
			os.Exit(-1)
		}
	} else if extension == ".pb" {
		// Try to read the source as a binary protocol buffer.
		message, err = g.readOpenAPIBinary(bytes)
		if err != nil {
			writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
			os.Exit(-1)
		}
	} else {
		err = errors.New("unknown file extension. 'json', 'yaml', and 'pb' are accepted")
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		os.Exit(-1)
	}
	// Perform actions specified by command options.
	err = g.performActions(message)
	if err != nil {
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		os.Exit(-1)
	}
}

func main() {
	g := newGnostic()
	g.main()
}
