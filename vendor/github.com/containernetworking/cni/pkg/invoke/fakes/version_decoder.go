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

package fakes

import "github.com/containernetworking/cni/pkg/version"

type VersionDecoder struct {
	DecodeCall struct {
		Received struct {
			JSONBytes []byte
		}
		Returns struct {
			PluginInfo version.PluginInfo
			Error      error
		}
	}
}

func (e *VersionDecoder) Decode(jsonData []byte) (version.PluginInfo, error) {
	e.DecodeCall.Received.JSONBytes = jsonData
	return e.DecodeCall.Returns.PluginInfo, e.DecodeCall.Returns.Error
}
