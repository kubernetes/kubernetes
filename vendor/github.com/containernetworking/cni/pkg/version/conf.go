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
)

// ConfigDecoder can decode the CNI version available in network config data
type ConfigDecoder struct{}

func (*ConfigDecoder) Decode(jsonBytes []byte) (string, error) {
	var conf struct {
		CNIVersion string `json:"cniVersion"`
	}
	err := json.Unmarshal(jsonBytes, &conf)
	if err != nil {
		return "", fmt.Errorf("decoding version from network config: %s", err)
	}
	if conf.CNIVersion == "" {
		return "0.1.0", nil
	}
	return conf.CNIVersion, nil
}
