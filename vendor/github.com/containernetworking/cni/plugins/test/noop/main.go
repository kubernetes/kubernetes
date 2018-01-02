// Copyright 2016 CNI authors
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

/*
Noop plugin is a CNI plugin designed for use as a test-double.

When calling, set the CNI_ARGS env var equal to the path of a file containing
the JSON encoding of a Debug.
*/

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/containernetworking/cni/pkg/skel"
	"github.com/containernetworking/cni/pkg/types"
	"github.com/containernetworking/cni/pkg/types/current"
	"github.com/containernetworking/cni/pkg/version"
	noop_debug "github.com/containernetworking/cni/plugins/test/noop/debug"
)

type NetConf struct {
	types.NetConf
	DebugFile  string          `json:"debugFile"`
	PrevResult *current.Result `json:"prevResult,omitempty"`
}

func loadConf(bytes []byte) (*NetConf, error) {
	n := &NetConf{}
	if err := json.Unmarshal(bytes, n); err != nil {
		return nil, fmt.Errorf("failed to load netconf: %v %q", err, string(bytes))
	}
	return n, nil
}

// parse extra args i.e. FOO=BAR;ABC=123
func parseExtraArgs(args string) (map[string]string, error) {
	m := make(map[string]string)
	if len(args) == 0 {
		return m, nil
	}

	items := strings.Split(args, ";")
	for _, item := range items {
		kv := strings.Split(item, "=")
		if len(kv) != 2 {
			return nil, fmt.Errorf("CNI_ARGS invalid key/value pair: %s\n", kv)
		}
		m[kv[0]] = kv[1]
	}
	return m, nil
}

func getConfig(stdinData []byte, args string) (string, *NetConf, error) {
	netConf, err := loadConf(stdinData)
	if err != nil {
		return "", nil, err
	}

	extraArgs, err := parseExtraArgs(args)
	if err != nil {
		return "", nil, err
	}

	debugFilePath, ok := extraArgs["DEBUG"]
	if !ok {
		debugFilePath = netConf.DebugFile
	}

	return debugFilePath, netConf, nil
}

func debugBehavior(args *skel.CmdArgs, command string) error {
	debugFilePath, netConf, err := getConfig(args.StdinData, args.Args)
	if err != nil {
		return err
	}

	if debugFilePath == "" {
		fmt.Printf(`{}`)
		os.Stderr.WriteString("CNI_ARGS or config empty, no debug behavior\n")
		return nil
	}

	debug, err := noop_debug.ReadDebug(debugFilePath)
	if err != nil {
		return err
	}

	debug.CmdArgs = *args
	debug.Command = command

	if debug.ReportResult == "" {
		debug.ReportResult = fmt.Sprintf(` { "result": %q }`, noop_debug.EmptyReportResultMessage)
	}

	err = debug.WriteDebug(debugFilePath)
	if err != nil {
		return err
	}

	os.Stderr.WriteString(debug.ReportStderr)

	if debug.ReportError != "" {
		return errors.New(debug.ReportError)
	} else if debug.ReportResult == "PASSTHROUGH" || debug.ReportResult == "INJECT-DNS" {
		if debug.ReportResult == "INJECT-DNS" {
			newResult, err := current.NewResultFromResult(netConf.PrevResult)
			if err != nil {
				return err
			}
			newResult.DNS.Nameservers = []string{"1.2.3.4"}
			netConf.PrevResult = newResult
		}
		newResult, err := json.Marshal(netConf.PrevResult)
		if err != nil {
			return fmt.Errorf("failed to marshal new result: %v", err)
		}
		os.Stdout.WriteString(string(newResult))
	} else {
		os.Stdout.WriteString(debug.ReportResult)
	}

	return nil
}

func debugGetSupportedVersions(stdinData []byte) []string {
	vers := []string{"0.-42.0", "0.1.0", "0.2.0", "0.3.0", "0.3.1"}
	cniArgs := os.Getenv("CNI_ARGS")
	if cniArgs == "" {
		return vers
	}

	debugFilePath, _, err := getConfig(stdinData, cniArgs)
	if err != nil {
		panic("test setup error: unable to get debug file path: " + err.Error())
	}

	debug, err := noop_debug.ReadDebug(debugFilePath)
	if err != nil {
		panic("test setup error: unable to read debug file: " + err.Error())
	}
	if debug.ReportVersionSupport == nil {
		return vers
	}
	return debug.ReportVersionSupport
}

func cmdAdd(args *skel.CmdArgs) error {
	return debugBehavior(args, "ADD")
}

func cmdDel(args *skel.CmdArgs) error {
	return debugBehavior(args, "DEL")
}

func saveStdin() ([]byte, error) {
	// Read original stdin
	stdinData, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		return nil, err
	}

	// Make a new pipe for stdin, and write original stdin data to it
	r, w, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	if _, err := w.Write(stdinData); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}

	os.Stdin = r
	return stdinData, nil
}

func main() {
	// Grab and read stdin before pkg/skel gets it
	stdinData, err := saveStdin()
	if err != nil {
		panic("test setup error: unable to read stdin: " + err.Error())
	}

	supportedVersions := debugGetSupportedVersions(stdinData)
	skel.PluginMain(cmdAdd, cmdDel, version.PluginSupports(supportedVersions...))
}
