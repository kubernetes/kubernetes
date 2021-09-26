package gnostic_plugin_v1

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"strings"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes/any"

	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
	discovery "github.com/googleapis/gnostic/discovery"
	surface "github.com/googleapis/gnostic/surface"
)

// Environment contains the environment of a plugin call.
type Environment struct {
	Request         *Request  // plugin request object
	Response        *Response // response message
	Invocation      string    // string representation of call
	RunningAsPlugin bool      // true if app is being run as a plugin
	Verbose         bool      // if true, plugin should log details to stderr
}

// NewEnvironment creates a plugin context from arguments and standard input.
func NewEnvironment() (env *Environment, err error) {
	env = &Environment{
		Invocation: os.Args[0],
		Response:   &Response{},
	}

	input := flag.String("input", "", "API description (in binary protocol buffer form)")
	output := flag.String("output", "-", "Output file or directory")
	plugin := flag.Bool("plugin", false, "Run as a gnostic plugin (other flags are ignored).")
	verbose := flag.Bool("verbose", false, "Write details to stderr.")
	flag.Parse()

	env.RunningAsPlugin = *plugin
	env.Verbose = *verbose
	programName := path.Base(os.Args[0])

	if (*input == "") && !*plugin {
		flag.Usage = func() {
			fmt.Fprintf(os.Stderr, "\n")
			fmt.Fprintf(os.Stderr, programName+" is a gnostic plugin.\n")
			fmt.Fprintf(os.Stderr, `
When it is run from gnostic, the -plugin option is specified and gnostic
writes a binary request to stdin and waits for a binary response on stdout.

This program can also be run standalone using the other flags listed below.
When the -plugin option is specified, these flags are ignored.`)
			fmt.Fprintf(os.Stderr, "\n\nUsage:\n")
			flag.PrintDefaults()
		}
		flag.Usage()
		os.Exit(0)
	}

	if env.RunningAsPlugin {
		// Handle invocation as a plugin.

		// Read the plugin input.
		pluginData, err := ioutil.ReadAll(os.Stdin)
		env.RespondAndExitIfError(err)
		if len(pluginData) == 0 {
			env.RespondAndExitIfError(fmt.Errorf("no input data"))
		}

		// Deserialize the request from the input.
		request := &Request{}
		err = proto.Unmarshal(pluginData, request)
		env.RespondAndExitIfError(err)

		// Collect parameters passed to the plugin.
		parameters := request.Parameters
		for _, parameter := range parameters {
			env.Invocation += " " + parameter.Name + "=" + parameter.Value
		}

		// Log the invocation.
		//log.Printf("Running plugin %s", env.Invocation)

		env.Request = request

	} else {
		// Handle invocation from the command line.

		// Read the input document.
		apiData, err := ioutil.ReadFile(*input)
		if len(apiData) == 0 {
			env.RespondAndExitIfError(fmt.Errorf("no input data"))
		}

		env.Request = &Request{}
		env.Request.OutputPath = *output
		env.Request.SourceName = path.Base(*input)

		// First try to unmarshal OpenAPI v2.
		documentv2 := &openapiv2.Document{}
		err = proto.Unmarshal(apiData, documentv2)
		if err == nil {
			env.Request.AddModel("openapi.v2.Document", documentv2)
			sourceName := guessSourceName(*input)
			// include experimental API surface model
			surfaceModel, err := surface.NewModelFromOpenAPI2(documentv2, sourceName)
			if err == nil {
				env.Request.AddModel("surface.v1.Model", surfaceModel)
			}
			return env, err
		}
		// If that failed, ignore deserialization errors and try to unmarshal OpenAPI v3.
		documentv3 := &openapiv3.Document{}
		err = proto.Unmarshal(apiData, documentv3)
		if err == nil {
			env.Request.AddModel("openapi.v3.Document", documentv3)
			sourceName := guessSourceName(*input)
			// include experimental API surface model
			surfaceModel, err := surface.NewModelFromOpenAPI3(documentv3, sourceName)
			if err == nil {
				env.Request.AddModel("surface.v1.Model", surfaceModel)
			}
			return env, err
		}
		// If that failed, ignore deserialization errors and try to unmarshal a Discovery document.
		discoveryDocument := &discovery.Document{}
		err = proto.Unmarshal(apiData, discoveryDocument)
		if err == nil {
			env.Request.AddModel("discovery.v1.Document", discoveryDocument)
			return env, err
		}
		// If we get here, we don't know what we got
		err = errors.New("Unrecognized format for input")
		return env, err
	}
	return env, err
}

// RespondAndExitIfError checks an error and if it is non-nil, records it and serializes and returns the response and then exits.
func (env *Environment) RespondAndExitIfError(err error) {
	if err != nil {
		env.Response.Errors = append(env.Response.Errors, err.Error())
		env.RespondAndExit()
	}
}

// RespondAndExit serializes and returns the plugin response and then exits.
func (env *Environment) RespondAndExit() {
	if env.RunningAsPlugin {
		responseBytes, _ := proto.Marshal(env.Response)
		os.Stdout.Write(responseBytes)
	} else {
		err := HandleResponse(env.Response, env.Request.OutputPath)
		if err != nil {
			log.Printf("%s", err.Error())
		}
	}
	os.Exit(0)
}

func HandleResponse(response *Response, outputLocation string) error {
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
	return nil
}

func (request *Request) AddModel(modelType string, model proto.Message) error {
	modelBytes, err := proto.Marshal(model)
	request.Models = append(request.Models, &any.Any{TypeUrl: modelType, Value: modelBytes})
	return err
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

// Guesses the sourceName from the binary input file name. E.g.: given input: some/path/swagger.pb
// check for some/path/swagger.yaml and some/path/swagger.json.
func guessSourceName(input string) string {
	sourceName := strings.Replace(input, ".pb", ".yaml", -1)
	if _, err := os.Stat(sourceName); os.IsNotExist(err) {
		// sourceName does not exist. Lets try .json instead
		sourceName = strings.Replace(input, ".pb", ".json", -1)
		if _, err := os.Stat(sourceName); os.IsNotExist(err) {
			return ""
		}
	}
	return sourceName
}
