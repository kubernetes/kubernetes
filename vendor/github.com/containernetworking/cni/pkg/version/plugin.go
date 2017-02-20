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

package version

import (
	"encoding/json"
	"fmt"
	"io"
)

// PluginInfo reports information about CNI versioning
type PluginInfo interface {
	// SupportedVersions returns one or more CNI spec versions that the plugin
	// supports.  If input is provided in one of these versions, then the plugin
	// promises to use the same CNI version in its response
	SupportedVersions() []string

	// Encode writes this CNI version information as JSON to the given Writer
	Encode(io.Writer) error
}

type pluginInfo struct {
	CNIVersion_        string   `json:"cniVersion"`
	SupportedVersions_ []string `json:"supportedVersions,omitempty"`
}

// pluginInfo implements the PluginInfo interface
var _ PluginInfo = &pluginInfo{}

func (p *pluginInfo) Encode(w io.Writer) error {
	return json.NewEncoder(w).Encode(p)
}

func (p *pluginInfo) SupportedVersions() []string {
	return p.SupportedVersions_
}

// PluginSupports returns a new PluginInfo that will report the given versions
// as supported
func PluginSupports(supportedVersions ...string) PluginInfo {
	if len(supportedVersions) < 1 {
		panic("programmer error: you must support at least one version")
	}
	return &pluginInfo{
		CNIVersion_:        Current(),
		SupportedVersions_: supportedVersions,
	}
}

// PluginDecoder can decode the response returned by a plugin's VERSION command
type PluginDecoder struct{}

func (*PluginDecoder) Decode(jsonBytes []byte) (PluginInfo, error) {
	var info pluginInfo
	err := json.Unmarshal(jsonBytes, &info)
	if err != nil {
		return nil, fmt.Errorf("decoding version info: %s", err)
	}
	if info.CNIVersion_ == "" {
		return nil, fmt.Errorf("decoding version info: missing field cniVersion")
	}
	if len(info.SupportedVersions_) == 0 {
		if info.CNIVersion_ == "0.2.0" {
			return PluginSupports("0.1.0", "0.2.0"), nil
		}
		return nil, fmt.Errorf("decoding version info: missing field supportedVersions")
	}
	return &info, nil
}
