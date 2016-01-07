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

package invoke

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/appc/cni/pkg/types"
)

func pluginErr(err error, output []byte) error {
	if _, ok := err.(*exec.ExitError); ok {
		emsg := types.Error{}
		if perr := json.Unmarshal(output, &emsg); perr != nil {
			return fmt.Errorf("netplugin failed but error parsing its diagnostic message %q: %v", string(output), perr)
		}
		details := ""
		if emsg.Details != "" {
			details = fmt.Sprintf("; %v", emsg.Details)
		}
		return fmt.Errorf("%v%v", emsg.Msg, details)
	}

	return err
}

func ExecPluginWithResult(pluginPath string, netconf []byte, args CNIArgs) (*types.Result, error) {
	stdoutBytes, err := execPlugin(pluginPath, netconf, args)
	if err != nil {
		return nil, err
	}

	res := &types.Result{}
	err = json.Unmarshal(stdoutBytes, res)
	return res, err
}

func ExecPluginWithoutResult(pluginPath string, netconf []byte, args CNIArgs) error {
	_, err := execPlugin(pluginPath, netconf, args)
	return err
}

func execPlugin(pluginPath string, netconf []byte, args CNIArgs) ([]byte, error) {
	if pluginPath == "" {
		return nil, fmt.Errorf("could not find %q plugin", filepath.Base(pluginPath))
	}

	stdout := &bytes.Buffer{}

	c := exec.Cmd{
		Env:    args.AsEnv(),
		Path:   pluginPath,
		Args:   []string{pluginPath},
		Stdin:  bytes.NewBuffer(netconf),
		Stdout: stdout,
		Stderr: os.Stderr,
	}
	if err := c.Run(); err != nil {
		return nil, pluginErr(err, stdout.Bytes())
	}

	return stdout.Bytes(), nil
}
