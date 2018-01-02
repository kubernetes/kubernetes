// Copyright 2014-2016 CNI authors
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

// Package skel provides skeleton code for a CNI plugin.
// In particular, it implements argument parsing and validation.
package skel

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"

	"github.com/containernetworking/cni/pkg/types"
	"github.com/containernetworking/cni/pkg/version"
)

// CmdArgs captures all the arguments passed in to the plugin
// via both env vars and stdin
type CmdArgs struct {
	ContainerID string
	Netns       string
	IfName      string
	Args        string
	Path        string
	StdinData   []byte
}

type dispatcher struct {
	Getenv func(string) string
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer

	ConfVersionDecoder version.ConfigDecoder
	VersionReconciler  version.Reconciler
}

type reqForCmdEntry map[string]bool

func (t *dispatcher) getCmdArgsFromEnv() (string, *CmdArgs, error) {
	var cmd, contID, netns, ifName, args, path string

	vars := []struct {
		name      string
		val       *string
		reqForCmd reqForCmdEntry
	}{
		{
			"CNI_COMMAND",
			&cmd,
			reqForCmdEntry{
				"ADD": true,
				"DEL": true,
			},
		},
		{
			"CNI_CONTAINERID",
			&contID,
			reqForCmdEntry{
				"ADD": false,
				"DEL": false,
			},
		},
		{
			"CNI_NETNS",
			&netns,
			reqForCmdEntry{
				"ADD": true,
				"DEL": false,
			},
		},
		{
			"CNI_IFNAME",
			&ifName,
			reqForCmdEntry{
				"ADD": true,
				"DEL": true,
			},
		},
		{
			"CNI_ARGS",
			&args,
			reqForCmdEntry{
				"ADD": false,
				"DEL": false,
			},
		},
		{
			"CNI_PATH",
			&path,
			reqForCmdEntry{
				"ADD": true,
				"DEL": true,
			},
		},
	}

	argsMissing := false
	for _, v := range vars {
		*v.val = t.Getenv(v.name)
		if *v.val == "" {
			if v.reqForCmd[cmd] || v.name == "CNI_COMMAND" {
				fmt.Fprintf(t.Stderr, "%v env variable missing\n", v.name)
				argsMissing = true
			}
		}
	}

	if argsMissing {
		return "", nil, fmt.Errorf("required env variables missing")
	}

	stdinData, err := ioutil.ReadAll(t.Stdin)
	if err != nil {
		return "", nil, fmt.Errorf("error reading from stdin: %v", err)
	}

	cmdArgs := &CmdArgs{
		ContainerID: contID,
		Netns:       netns,
		IfName:      ifName,
		Args:        args,
		Path:        path,
		StdinData:   stdinData,
	}
	return cmd, cmdArgs, nil
}

func createTypedError(f string, args ...interface{}) *types.Error {
	return &types.Error{
		Code: 100,
		Msg:  fmt.Sprintf(f, args...),
	}
}

func (t *dispatcher) checkVersionAndCall(cmdArgs *CmdArgs, pluginVersionInfo version.PluginInfo, toCall func(*CmdArgs) error) error {
	configVersion, err := t.ConfVersionDecoder.Decode(cmdArgs.StdinData)
	if err != nil {
		return err
	}
	verErr := t.VersionReconciler.Check(configVersion, pluginVersionInfo)
	if verErr != nil {
		return &types.Error{
			Code:    types.ErrIncompatibleCNIVersion,
			Msg:     "incompatible CNI versions",
			Details: verErr.Details(),
		}
	}
	return toCall(cmdArgs)
}

func (t *dispatcher) pluginMain(cmdAdd, cmdDel func(_ *CmdArgs) error, versionInfo version.PluginInfo) *types.Error {
	cmd, cmdArgs, err := t.getCmdArgsFromEnv()
	if err != nil {
		return createTypedError(err.Error())
	}

	switch cmd {
	case "ADD":
		err = t.checkVersionAndCall(cmdArgs, versionInfo, cmdAdd)
	case "DEL":
		err = t.checkVersionAndCall(cmdArgs, versionInfo, cmdDel)
	case "VERSION":
		err = versionInfo.Encode(t.Stdout)
	default:
		return createTypedError("unknown CNI_COMMAND: %v", cmd)
	}

	if err != nil {
		if e, ok := err.(*types.Error); ok {
			// don't wrap Error in Error
			return e
		}
		return createTypedError(err.Error())
	}
	return nil
}

// PluginMainWithError is the core "main" for a plugin. It accepts
// callback functions for add and del CNI commands and returns an error.
//
// The caller must also specify what CNI spec versions the plugin supports.
//
// It is the responsibility of the caller to check for non-nil error return.
//
// For a plugin to comply with the CNI spec, it must print any error to stdout
// as JSON and then exit with nonzero status code.
//
// To let this package automatically handle errors and call os.Exit(1) for you,
// use PluginMain() instead.
func PluginMainWithError(cmdAdd, cmdDel func(_ *CmdArgs) error, versionInfo version.PluginInfo) *types.Error {
	return (&dispatcher{
		Getenv: os.Getenv,
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}).pluginMain(cmdAdd, cmdDel, versionInfo)
}

// PluginMain is the core "main" for a plugin which includes automatic error handling.
//
// The caller must also specify what CNI spec versions the plugin supports.
//
// When an error occurs in either cmdAdd or cmdDel, PluginMain will print the error
// as JSON to stdout and call os.Exit(1).
//
// To have more control over error handling, use PluginMainWithError() instead.
func PluginMain(cmdAdd, cmdDel func(_ *CmdArgs) error, versionInfo version.PluginInfo) {
	if e := PluginMainWithError(cmdAdd, cmdDel, versionInfo); e != nil {
		if err := e.Print(); err != nil {
			log.Print("Error writing error JSON to stdout: ", err)
		}
		os.Exit(1)
	}
}
