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

package lib

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	openapi_v3 "github.com/googleapis/gnostic/openapiv3"
	"github.com/googleapis/gnostic/compiler"
	discovery_v1 "github.com/googleapis/gnostic/discovery"
	"github.com/googleapis/gnostic/jsonwriter"
	plugins "github.com/googleapis/gnostic/plugins"
	surface "github.com/googleapis/gnostic/surface"
	"gopkg.in/yaml.v2"
)

const (
	// SourceFormatUnknown represents an unrecognized source format
	SourceFormatUnknown = 0
	// SourceFormatOpenAPI2 represents an OpenAPI v2 document
	SourceFormatOpenAPI2 = 2
	// SourceFormatOpenAPI3 represents an OpenAPI v3 document
	SourceFormatOpenAPI3 = 3
	// SourceFormatDiscovery represents a Google Discovery document
	SourceFormatDiscovery = 4
)

// Determine the version of an OpenAPI description read from JSON or YAML.
func getOpenAPIVersionFromInfo(info interface{}) int {
	m, ok := compiler.UnpackMap(info)
	if !ok {
		return SourceFormatUnknown
	}
	swagger, ok := compiler.MapValueForKey(m, "swagger").(string)
	if ok && strings.HasPrefix(swagger, "2.0") {
		return SourceFormatOpenAPI2
	}
	openapi, ok := compiler.MapValueForKey(m, "openapi").(string)
	if ok && strings.HasPrefix(openapi, "3.0") {
		return SourceFormatOpenAPI3
	}
	kind, ok := compiler.MapValueForKey(m, "kind").(string)
	if ok && kind == "discovery#restDescription" {
		return SourceFormatDiscovery
	}
	return SourceFormatUnknown
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
func (p *pluginCall) perform(document proto.Message, sourceFormat int, sourceName string, timePlugins bool) ([]*plugins.Message, error) {
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
			return nil, fmt.Errorf("Invalid invocation of %s: %s", executableName, invocation)
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

		request.SourceName = sourceName
		switch sourceFormat {
		case SourceFormatOpenAPI2:
			request.AddModel("openapi.v2.Document", document)
			// include experimental API surface model
			surfaceModel, err := surface.NewModelFromOpenAPI2(document.(*openapi_v2.Document), sourceName)
			if err == nil {
				request.AddModel("surface.v1.Model", surfaceModel)
			}
		case SourceFormatOpenAPI3:
			request.AddModel("openapi.v3.Document", document)
			// include experimental API surface model
			surfaceModel, err := surface.NewModelFromOpenAPI3(document.(*openapi_v3.Document), sourceName)
			if err == nil {
				request.AddModel("surface.v1.Model", surfaceModel)
			}
		case SourceFormatDiscovery:
			request.AddModel("discovery.v1.Document", document)
		default:
		}

		requestBytes, _ := proto.Marshal(request)

		cmd := exec.Command(executableName, "-plugin")
		cmd.Stdin = bytes.NewReader(requestBytes)
		cmd.Stderr = os.Stderr
		pluginStartTime := time.Now()
		output, err := cmd.Output()
		pluginElapsedTime := time.Since(pluginStartTime)
		if timePlugins {
			fmt.Printf("> %s (%s)\n", executableName, pluginElapsedTime)
		}
		if err != nil {
			return nil, err
		}
		response := &plugins.Response{}
		err = proto.Unmarshal(output, response)
		if err != nil {
			// Gnostic expects plugins to only write the
			// response message to stdout. Be sure that
			// any logging messages are written to stderr only.
			return nil, errors.New("invalid plugin response (plugins must write log messages to stderr, not stdout)")
		}

		err = plugins.HandleResponse(response, outputLocation)

		return response.Messages, err
	}
	return nil, nil
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
	args              []string
	usage             string
	sourceName        string
	binaryOutputPath  string
	textOutputPath    string
	yamlOutputPath    string
	jsonOutputPath    string
	errorOutputPath   string
	messageOutputPath string
	resolveReferences bool
	pluginCalls       []*pluginCall
	extensionHandlers []compiler.ExtensionHandler
	sourceFormat      int
	timePlugins       bool
}

// NewGnostic initializes a structure to store global application state.
func NewGnostic(args []string) *Gnostic {
	g := &Gnostic{args: args}
	// Option fields initialize to their default values.
	g.usage = `
Usage: gnostic SOURCE [OPTIONS]
  SOURCE is the filename or URL of an API description.
Options:
  --pb-out=PATH       Write a binary proto to the specified location.
  --text-out=PATH     Write a text proto to the specified location.
  --json-out=PATH     Write a json API description to the specified location.
  --yaml-out=PATH     Write a yaml API description to the specified location.
  --errors-out=PATH   Write compilation errors to the specified location.
  --messages-out=PATH Write messages generated by plugins to the specified
                      location. Messages from all plugin invocations are
                      written to a single common file.
  --PLUGIN-out=PATH   Run the plugin named gnostic-PLUGIN and write results
                      to the specified location.
  --PLUGIN            Run the plugin named gnostic-PLUGIN but don't write any
                      results. Used for plugins that return messages only.
                      PLUGIN must not match any other gnostic option.
  --x-EXTENSION       Use the extension named gnostic-x-EXTENSION
                      to process OpenAPI specification extensions.
  --resolve-refs      Explicitly resolve $ref references.
                      This could have problems with recursive definitions.
  --time-plugins      Report plugin runtimes.
`
	// Initialize internal structures.
	g.pluginCalls = make([]*pluginCall, 0)
	g.extensionHandlers = make([]compiler.ExtensionHandler, 0)
	return g
}

// Parse command-line options.
func (g *Gnostic) readOptions() error {
	// plugin processing matches patterns of the form "--PLUGIN-out=PATH" and "--PLUGIN_out=PATH"
	pluginRegex := regexp.MustCompile("--(.+)[-_]out=(.+)")

	// extension processing matches patterns of the form "--x-EXTENSION"
	extensionRegex := regexp.MustCompile("--x-(.+)")

	for i, arg := range g.args {
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
			case "messages":
				g.messageOutputPath = invocation
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
		} else if arg == "--time-plugins" {
			g.timePlugins = true
		} else if arg[0] == '-' && arg[1] == '-' {
			// try letting the option specify a plugin with no output files (or unwanted output files)
			// this is useful for calling plugins like linters that only return messages
			p := &pluginCall{Name: arg[2:len(arg)], Invocation: "!"}
			g.pluginCalls = append(g.pluginCalls, p)
		} else if arg[0] == '-' {
			return fmt.Errorf("unknown option: %s", arg)
		} else {
			g.sourceName = arg
		}
	}
	return nil
}

// Validate command-line options.
func (g *Gnostic) validateOptions() error {
	if g.binaryOutputPath == "" &&
		g.textOutputPath == "" &&
		g.yamlOutputPath == "" &&
		g.jsonOutputPath == "" &&
		g.errorOutputPath == "" &&
		len(g.pluginCalls) == 0 {
		return fmt.Errorf("missing output directives")
	}
	if g.sourceName == "" {
		return fmt.Errorf("no input specified")
	}
	// If we get here and the error output is unspecified, write errors to stderr.
	if g.errorOutputPath == "" {
		g.errorOutputPath = "="
	}
	return nil
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
	g.sourceFormat = getOpenAPIVersionFromInfo(info)
	if g.sourceFormat == SourceFormatUnknown {
		return nil, errors.New("unable to identify OpenAPI version")
	}
	// Compile to the proto model.
	if g.sourceFormat == SourceFormatOpenAPI2 {
		document, err := openapi_v2.NewDocument(info, compiler.NewContextWithExtensions("$root", nil, &g.extensionHandlers))
		if err != nil {
			return nil, err
		}
		message = document
	} else if g.sourceFormat == SourceFormatOpenAPI3 {
		document, err := openapi_v3.NewDocument(info, compiler.NewContextWithExtensions("$root", nil, &g.extensionHandlers))
		if err != nil {
			return nil, err
		}
		message = document
	} else {
		document, err := discovery_v1.NewDocument(info, compiler.NewContextWithExtensions("$root", nil, &g.extensionHandlers))
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
		g.sourceFormat = SourceFormatOpenAPI3
		return documentV3, nil
	}
	// if that failed, try to read an OpenAPI v2 document
	documentV2 := &openapi_v2.Document{}
	err = proto.Unmarshal(data, documentV2)
	if err == nil && strings.HasPrefix(documentV2.Swagger, "2.0") {
		g.sourceFormat = SourceFormatOpenAPI2
		return documentV2, nil
	}
	// if that failed, try to read a Discovery Format document
	discoveryDocument := &discovery_v1.Document{}
	err = proto.Unmarshal(data, discoveryDocument)
	if err == nil { // && strings.HasPrefix(documentV2.Swagger, "2.0") {
		g.sourceFormat = SourceFormatDiscovery
		return discoveryDocument, nil
	}
	return nil, err
}

// Write a binary pb representation.
func (g *Gnostic) writeBinaryOutput(message proto.Message) error {
	protoBytes, err := proto.Marshal(message)
	if err != nil {
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
	} else {
		writeFile(g.binaryOutputPath, protoBytes, g.sourceName, "pb")
	}
	return err
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
	if g.sourceFormat == SourceFormatOpenAPI2 {
		document := message.(*openapi_v2.Document)
		rawInfo, ok = document.ToRawInfo().(yaml.MapSlice)
		if !ok {
			rawInfo = nil
		}
	} else if g.sourceFormat == SourceFormatOpenAPI3 {
		document := message.(*openapi_v3.Document)
		rawInfo, ok = document.ToRawInfo().(yaml.MapSlice)
		if !ok {
			rawInfo = nil
		}
	} else if g.sourceFormat == SourceFormatDiscovery {
		document := message.(*discovery_v1.Document)
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

// Write messages.
func (g *Gnostic) writeMessagesOutput(message proto.Message) error {
	protoBytes, err := proto.Marshal(message)
	if err != nil {
		writeFile(g.messageOutputPath, g.errorBytes(err), g.sourceName, "errors")
	} else {
		writeFile(g.messageOutputPath, protoBytes, g.sourceName, "messages.pb")
	}
	return err
}

// Perform all actions specified in the command-line options.
func (g *Gnostic) performActions(message proto.Message) (err error) {
	// Optionally resolve internal references.
	if g.resolveReferences {
		if g.sourceFormat == SourceFormatOpenAPI2 {
			document := message.(*openapi_v2.Document)
			_, err = document.ResolveReferences(g.sourceName)
		} else if g.sourceFormat == SourceFormatOpenAPI3 {
			document := message.(*openapi_v3.Document)
			_, err = document.ResolveReferences(g.sourceName)
		}
		if err != nil {
			return err
		}
	}
	// Optionally write proto in binary format.
	if g.binaryOutputPath != "" {
		err = g.writeBinaryOutput(message)
		if err != nil {
			return err
		}
	}
	// Optionally write proto in text format.
	if g.textOutputPath != "" {
		g.writeTextOutput(message)
	}
	// Optionally write document in yaml and/or json formats.
	if g.yamlOutputPath != "" || g.jsonOutputPath != "" {
		g.writeJSONYAMLOutput(message)
	}
	// Call all specified plugins.
	messages := make([]*plugins.Message, 0)
	errors := make([]error, 0)
	for _, p := range g.pluginCalls {
		pluginMessages, err := p.perform(message, g.sourceFormat, g.sourceName, g.timePlugins)
		if err != nil {
			// we don't exit or fail here so that we run all plugins even when some have errors
			errors = append(errors, err)
		}
		messages = append(messages, pluginMessages...)
	}
	if g.messageOutputPath != "" {
		err = g.writeMessagesOutput(&plugins.Messages{Messages: messages})
		if err != nil {
			return err
		}
	} else {
		// Print any messages from the plugins
		if len(messages) > 0 {
			for _, message := range messages {
				fmt.Printf("%+v\n", message)
			}
		}
	}
	return compiler.NewErrorGroupOrNil(errors)
}

// Main is the main program for Gnostic.
func (g *Gnostic) Main() error {

	compiler.ClearCaches()

	var err error
	err = g.readOptions()
	if err != nil {
		return err
	}
	err = g.validateOptions()
	if err != nil {
		return err
	}
	// Read the OpenAPI source.
	bytes, err := compiler.ReadBytesForFile(g.sourceName)
	if err != nil {
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		return err
	}
	extension := strings.ToLower(filepath.Ext(g.sourceName))
	var message proto.Message
	if extension == ".json" || extension == ".yaml" {
		// Try to read the source as JSON/YAML.
		message, err = g.readOpenAPIText(bytes)
		if err != nil {
			writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
			return err
		}
	} else if extension == ".pb" {
		// Try to read the source as a binary protocol buffer.
		message, err = g.readOpenAPIBinary(bytes)
		if err != nil {
			writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
			return err
		}
	} else {
		err = errors.New("unknown file extension. 'json', 'yaml', and 'pb' are accepted")
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		return err
	}
	// Perform actions specified by command options.
	err = g.performActions(message)
	if err != nil {
		writeFile(g.errorOutputPath, g.errorBytes(err), g.sourceName, "errors")
		return err
	}
	return nil
}
