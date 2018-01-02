// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

// cniproxy proxies commands through to the real container plugin
// and logs output for later inspection.

// For an example execution, see NewNetCNIDNSTest

// The following CNI arguments influence its behavior:
//
// X_REAL_PLUGIN=<<name>>: The name of the real plugin to execute
// X_ADD_DNS=1: populate a DNS response
// X_FAIL=exit / crash: "exit" = fail with error message, "crash" = fail without error message
// X_LOG=<<filename>>: write a logfile of what happens in the same directory as the binary

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/containernetworking/cni/pkg/invoke"
)

var cniArgs map[string]string

func main() {
	parseCniArgs()

	x_fail := cniArgs["X_FAIL"]
	if x_fail == "exit" {
		os.Stdout.WriteString(`{"cniVersion": "0.1.0", "code": 100, "msg": "this is a failure message"}`)
		os.Exit(254)
	} else if x_fail == "crash" {
		os.Exit(254)
	}

	realPluginPath := findRealPlugin()

	stdin, plugout, plugerr, exitCode := proxyPlugin(realPluginPath)
	logResult(realPluginPath, stdin, plugout, plugerr, exitCode)

	// Mutate the response
	if exitCode == 0 && cniArgs["X_ADD_DNS"] == "1" {
		addDns(plugout)
	}

	// Pass through the response to rkt
	os.Stdout.Write(plugout.Bytes())
	os.Stderr.Write(plugerr.Bytes())

	os.Exit(exitCode) // pass through exit code
}

// mutate the response to add DNS info
func addDns(data *bytes.Buffer) {
	var result interface{}

	err := json.Unmarshal(data.Bytes(), &result)
	if err != nil {
		fail("could not parse json", err)
	}

	switch result := result.(type) {
	case map[string]interface{}:
		result["dns"] = map[string]interface{}{
			"nameservers": []string{"1.2.3.4", "4.3.2.1"},
			"domain":      "dotawesome.awesome",
			"search":      []string{"cniproxy1, cniproxy2"},
			"options":     []string{"option1", "option2"},
		}
		newbytes, err := json.Marshal(result)
		if err != nil {
			fail("Failed to write json", err)
		}

		data.Reset()
		data.Write(newbytes)

	default:
		fail("json not expected format")
	}
}

func logResult(pluginPath string, stdin, stdout, stderr *bytes.Buffer, exitCode int) {
	logfile, exists := cniArgs["X_LOG"]
	if !exists {
		return
	}

	logfile = filepath.Join(filepath.Dir(os.Args[0]), logfile)
	fp, err := os.Create(logfile)
	if err != nil {
		fail("Could not open log file", logfile, err)
	}
	defer fp.Close()

	result := map[string]interface{}{
		"pluginPath": pluginPath,
		"stdin":      stdin.String(),
		"stdout":     stdout.String(),
		"stderr":     stderr.String(),
		"exitCode":   exitCode,
		"env":        os.Environ(),
	}

	enc := json.NewEncoder(fp)
	err = enc.Encode(result)
	if err != nil {
		fail("Could not write logfile", err)
	}
}

// Proxy execution through the **real** plugin
func proxyPlugin(path string) (stdin, stdout, stderr *bytes.Buffer, exitCode int) {

	// Filter all X_ arguments from the cni argument variable
	filteredArgs := make([]string, 0, len(cniArgs))
	for k, v := range cniArgs {
		if !strings.HasPrefix(k, "X_") {
			filteredArgs = append(filteredArgs, fmt.Sprintf("%s=%s", k, v))
		}
	}
	os.Setenv("CNI_ARGS", strings.Join(filteredArgs, ";"))

	stdin = bytes.NewBuffer([]byte{})
	stdout = bytes.NewBuffer([]byte{})
	stderr = bytes.NewBuffer([]byte{})

	teeIn := io.TeeReader(os.Stdin, stdin)

	cmd := exec.Command(path)
	cmd.Stdin = teeIn
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	exitStatus := cmd.Run()

	if exitStatus == nil {
		exitCode = 0
	} else {
		switch exitStatus := exitStatus.(type) {
		case *exec.ExitError:
			exitCode = exitStatus.Sys().(syscall.WaitStatus).ExitStatus()
		default:
			fail("exec", exitStatus)
		}
	}
	return
}

// Find the proxied plugin from the CNI_PATH environment var
func findRealPlugin() string {
	pluginName := getArgOrFail("X_REAL_PLUGIN")
	paths := strings.Split(os.Getenv("CNI_PATH"), ":")

	pluginPath, err := invoke.FindInPath(pluginName, paths)

	if err != nil {
		fail("Could not find plugin in CNI_PATH", err)
	}
	return pluginPath
}

func fail(msgs ...interface{}) {
	fmt.Fprintln(os.Stderr, append([]interface{}{"CNIPROXY fail:"}, msgs...)...)
	os.Exit(254)
}

func getArgOrFail(key string) string {
	val, exists := cniArgs[key]

	if !exists {
		fail("Needed CNI_ARG", key)
	}
	return val
}

func parseCniArgs() {
	cniArgs = make(map[string]string)

	argStr := os.Getenv("CNI_ARGS")

	for _, arg := range strings.Split(argStr, ";") {
		argkv := strings.Split(arg, "=")
		if len(argkv) != 2 {
			fail("Invalid CNI arg", arg)
		}
		cniArgs[argkv[0]] = argkv[1]
	}
}
