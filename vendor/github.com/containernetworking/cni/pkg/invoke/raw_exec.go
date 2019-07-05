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

package invoke

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"

	"github.com/containernetworking/cni/pkg/types"
)

type RawExec struct {
	Stderr io.Writer
}

func (e *RawExec) ExecPlugin(ctx context.Context, pluginPath string, stdinData []byte, environ []string) ([]byte, error) {
	stdout := &bytes.Buffer{}
	c := exec.CommandContext(ctx, pluginPath)
	c.Env = environ
	c.Stdin = bytes.NewBuffer(stdinData)
	c.Stdout = stdout
	c.Stderr = e.Stderr
	if err := c.Run(); err != nil {
		return nil, pluginErr(err, stdout.Bytes())
	}

	return stdout.Bytes(), nil
}

func pluginErr(err error, output []byte) error {
	if _, ok := err.(*exec.ExitError); ok {
		emsg := types.Error{}
		if len(output) == 0 {
			emsg.Msg = "netplugin failed with no error message"
		} else if perr := json.Unmarshal(output, &emsg); perr != nil {
			emsg.Msg = fmt.Sprintf("netplugin failed but error parsing its diagnostic message %q: %v", string(output), perr)
		}
		return &emsg
	}

	return err
}

func (e *RawExec) FindInPath(plugin string, paths []string) (string, error) {
	return FindInPath(plugin, paths)
}
