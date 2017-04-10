// Copyright 2015 CNI authors
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

package invoke

import (
	"fmt"
	"os"

	"github.com/containernetworking/cni/pkg/types"
	"github.com/containernetworking/cni/pkg/version"
)

func ExecPluginWithResult(pluginPath string, netconf []byte, args CNIArgs) (types.Result, error) {
	return defaultPluginExec.WithResult(pluginPath, netconf, args)
}

func ExecPluginWithoutResult(pluginPath string, netconf []byte, args CNIArgs) error {
	return defaultPluginExec.WithoutResult(pluginPath, netconf, args)
}

func GetVersionInfo(pluginPath string) (version.PluginInfo, error) {
	return defaultPluginExec.GetVersionInfo(pluginPath)
}

var defaultPluginExec = &PluginExec{
	RawExec:        &RawExec{Stderr: os.Stderr},
	VersionDecoder: &version.PluginDecoder{},
}

type PluginExec struct {
	RawExec interface {
		ExecPlugin(pluginPath string, stdinData []byte, environ []string) ([]byte, error)
	}
	VersionDecoder interface {
		Decode(jsonBytes []byte) (version.PluginInfo, error)
	}
}

func (e *PluginExec) WithResult(pluginPath string, netconf []byte, args CNIArgs) (types.Result, error) {
	stdoutBytes, err := e.RawExec.ExecPlugin(pluginPath, netconf, args.AsEnv())
	if err != nil {
		return nil, err
	}

	// Plugin must return result in same version as specified in netconf
	versionDecoder := &version.ConfigDecoder{}
	confVersion, err := versionDecoder.Decode(netconf)
	if err != nil {
		return nil, err
	}

	return version.NewResult(confVersion, stdoutBytes)
}

func (e *PluginExec) WithoutResult(pluginPath string, netconf []byte, args CNIArgs) error {
	_, err := e.RawExec.ExecPlugin(pluginPath, netconf, args.AsEnv())
	return err
}

// GetVersionInfo returns the version information available about the plugin.
// For recent-enough plugins, it uses the information returned by the VERSION
// command.  For older plugins which do not recognize that command, it reports
// version 0.1.0
func (e *PluginExec) GetVersionInfo(pluginPath string) (version.PluginInfo, error) {
	args := &Args{
		Command: "VERSION",

		// set fake values required by plugins built against an older version of skel
		NetNS:  "dummy",
		IfName: "dummy",
		Path:   "dummy",
	}
	stdin := []byte(fmt.Sprintf(`{"cniVersion":%q}`, version.Current()))
	stdoutBytes, err := e.RawExec.ExecPlugin(pluginPath, stdin, args.AsEnv())
	if err != nil {
		if err.Error() == "unknown CNI_COMMAND: VERSION" {
			return version.PluginSupports("0.1.0"), nil
		}
		return nil, err
	}

	return e.VersionDecoder.Decode(stdoutBytes)
}
