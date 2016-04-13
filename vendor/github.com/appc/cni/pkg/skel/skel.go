// Copyright 2014 CoreOS, Inc.
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
	"io/ioutil"
	"log"
	"os"

	"github.com/appc/cni/pkg/types"
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

// PluginMain is the "main" for a plugin. It accepts
// two callback functions for add and del commands.
func PluginMain(cmdAdd, cmdDel func(_ *CmdArgs) error) {
	var cmd, contID, netns, ifName, args, path string

	vars := []struct {
		name string
		val  *string
		req  bool
	}{
		{"CNI_COMMAND", &cmd, true},
		{"CNI_CONTAINERID", &contID, false},
		{"CNI_NETNS", &netns, true},
		{"CNI_IFNAME", &ifName, true},
		{"CNI_ARGS", &args, false},
		{"CNI_PATH", &path, true},
	}

	argsMissing := false
	for _, v := range vars {
		*v.val = os.Getenv(v.name)
		if v.req && *v.val == "" {
			log.Printf("%v env variable missing", v.name)
			argsMissing = true
		}
	}

	if argsMissing {
		dieMsg("required env variables missing")
	}

	stdinData, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		dieMsg("error reading from stdin: %v", err)
	}

	cmdArgs := &CmdArgs{
		ContainerID: contID,
		Netns:       netns,
		IfName:      ifName,
		Args:        args,
		Path:        path,
		StdinData:   stdinData,
	}

	switch cmd {
	case "ADD":
		err = cmdAdd(cmdArgs)

	case "DEL":
		err = cmdDel(cmdArgs)

	default:
		dieMsg("unknown CNI_COMMAND: %v", cmd)
	}

	if err != nil {
		if e, ok := err.(*types.Error); ok {
			// don't wrap Error in Error
			dieErr(e)
		}
		dieMsg(err.Error())
	}
}

func dieMsg(f string, args ...interface{}) {
	e := &types.Error{
		Code: 100,
		Msg:  fmt.Sprintf(f, args...),
	}
	dieErr(e)
}

func dieErr(e *types.Error) {
	if err := e.Print(); err != nil {
		log.Print("Error writing error JSON to stdout: ", err)
	}
	os.Exit(1)
}
