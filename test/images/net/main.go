/*
Copyright 2014 The Kubernetes Authors.

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
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"

	"k8s.io/kubernetes/test/images/net/common"
	"k8s.io/kubernetes/test/images/net/nat"
)

type runnerMap map[string]common.Runner

type runRequestJSON struct {
	runner  string
	options interface{}
}

var (
	// flags for the command line. See usage args below for
	// descriptions.
	flags struct {
		Serve   string
		Runner  string
		Options string
	}
	// runners is a map from runner name to runner instance.
	runners = makeRunnerMap()
)

type logOutput struct {
	b bytes.Buffer
}

func main() {
	initFlags()
	log.SetFlags(log.Flags() | log.Lshortfile)

	if flags.Serve == "" {
		output, err := executeRunner(flags.Runner, flags.Options)
		if err == nil {
			fmt.Print("output:\n\n" + output.b.String())
			os.Exit(0)
		} else {
			log.Printf("Error: %v", err)
			fmt.Print("output:\n\n" + output.b.String())
			os.Exit(1)
		}
	} else {
		http.HandleFunc("/run/", handleRunRequest)
		log.Printf("Running server on %v", flags.Serve)
		log.Fatal(http.ListenAndServe(flags.Serve, nil))
	}
}

func initFlags() {
	legalRunners := ""
	for k := range runners {
		legalRunners += " " + k
	}
	flag.StringVar(
		&flags.Serve, "serve", "",
		"Address and port to bind to (e.g. 127.0.0.1:8080). Setting this will "+
			"run the network tester in server mode runner are triggered through "+
			"HTTP requests.")
	flag.StringVar(
		&flags.Runner, "runner", "",
		"Runner to execute (available:"+legalRunners+")")
	flag.StringVar(
		&flags.Options, "options", "",
		"JSON options to the Runner")
	flag.Parse()

	if flags.Runner == "" && flags.Serve == "" {
		log.Fatalf("Must set either -runner or -serve, see --help")
	}
}

func makeRunnerMap() runnerMap {
	// runner name is <pkg>-<file>-<specific>.
	return runnerMap{
		"nat-closewait-client": nat.NewCloseWaitClient(),
		"nat-closewait-server": nat.NewCloseWaitServer(),
	}
}

func executeRunner(name string, rawOptions string) (logOutput, error) {
	runner, ok := runners[name]
	if ok {
		options := runner.NewOptions()
		if err := json.Unmarshal([]byte(rawOptions), options); err != nil {
			return logOutput{}, fmt.Errorf("Invalid options JSON: %v", err)
		}

		log.Printf("Options: %+v", options)

		output := logOutput{}
		logger := log.New(&output.b, "# ", log.Lshortfile)

		return output, runner.Run(logger, options)
	}

	return logOutput{}, fmt.Errorf("Invalid runner: '%v', see --help\n", runner)
}

// handleRunRequest handles a request JSON to the network tester.
func handleRunRequest(w http.ResponseWriter, r *http.Request) {
	log.Printf("handleRunRequest %v", *r)

	urlParts := strings.Split(r.URL.Path, "/")
	if len(urlParts) != 3 {
		http.Error(w, fmt.Sprintf("invalid request to run: %v", urlParts), 400)
		return
	}

	runner := urlParts[2]
	if r.Body == nil {
		http.Error(w, "Missing request body", 400)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("error reading body: %v", err), 400)
		return
	}

	var output logOutput
	if output, err = executeRunner(runner, string(body)); err != nil {
		contents := fmt.Sprintf("Error from runner: %v\noutput:\n\n%s",
			err, output.b.String())
		http.Error(w, contents, 500)
		return
	}

	fmt.Fprintf(w, "ok\noutput:\n\n"+output.b.String())
}

func setupLogger() {
}
