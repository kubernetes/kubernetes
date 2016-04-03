// Copyright 2015 CoreOS, Inc.
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

import (
	"fmt"
	"net/rpc"
	"os"
	"path/filepath"

	"github.com/appc/cni/pkg/skel"
	"github.com/appc/cni/pkg/types"
)

const socketPath = "/run/cni/dhcp.sock"

func main() {
	if len(os.Args) > 1 && os.Args[1] == "daemon" {
		runDaemon()
	} else {
		skel.PluginMain(cmdAdd, cmdDel)
	}
}

func cmdAdd(args *skel.CmdArgs) error {
	result := types.Result{}
	if err := rpcCall("DHCP.Allocate", args, &result); err != nil {
		return err
	}
	return result.Print()
}

func cmdDel(args *skel.CmdArgs) error {
	result := struct{}{}
	if err := rpcCall("DHCP.Release", args, &result); err != nil {
		return fmt.Errorf("error dialing DHCP daemon: %v", err)
	}
	return nil
}

func rpcCall(method string, args *skel.CmdArgs, result interface{}) error {
	client, err := rpc.DialHTTP("unix", socketPath)
	if err != nil {
		return fmt.Errorf("error dialing DHCP daemon: %v", err)
	}

	// The daemon may be running under a different working dir
	// so make sure the netns path is absolute.
	netns, err := filepath.Abs(args.Netns)
	if err != nil {
		return fmt.Errorf("failed to make %q an absolute path: %v", args.Netns, err)
	}
	args.Netns = netns

	err = client.Call(method, args, result)
	if err != nil {
		return fmt.Errorf("error calling %v: %v", method, err)
	}

	return nil
}
