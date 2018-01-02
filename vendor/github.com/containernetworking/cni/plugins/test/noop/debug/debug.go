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

// debug supports tests that use the noop plugin
package debug

import (
	"encoding/json"
	"io/ioutil"

	"github.com/containernetworking/cni/pkg/skel"
)

const EmptyReportResultMessage = "set debug.ReportResult and call debug.WriteDebug() before calling this plugin"

// Debug is used to control and record the behavior of the noop plugin
type Debug struct {
	// Report* fields allow the test to control the behavior of the no-op plugin
	ReportResult         string
	ReportError          string
	ReportStderr         string
	ReportVersionSupport []string

	// Command stores the CNI command that the plugin received
	Command string

	// CmdArgs stores the CNI Args and Env Vars that the plugin recieved
	CmdArgs skel.CmdArgs
}

// ReadDebug will return a debug file recorded by the noop plugin
func ReadDebug(debugFilePath string) (*Debug, error) {
	debugBytes, err := ioutil.ReadFile(debugFilePath)
	if err != nil {
		return nil, err
	}

	var debug Debug
	err = json.Unmarshal(debugBytes, &debug)
	if err != nil {
		return nil, err
	}

	return &debug, nil
}

// WriteDebug will create a debug file to control the noop plugin
func (debug *Debug) WriteDebug(debugFilePath string) error {
	debugBytes, err := json.Marshal(debug)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(debugFilePath, debugBytes, 0600)
	if err != nil {
		return err
	}

	return nil
}
