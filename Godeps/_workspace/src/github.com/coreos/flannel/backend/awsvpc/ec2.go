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

package awsvpc

import (
	"encoding/json"
	"fmt"

	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/mitchellh/goamz/aws"
)

func getInstanceIdentity() (map[string]interface{}, error) {
	url := "http://169.254.169.254/latest/dynamic/instance-identity/document"

	resp, err := aws.RetryingClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		err = fmt.Errorf("Code %d returned for url %s", resp.StatusCode, url)
		return nil, err
	}

	dec := json.NewDecoder(resp.Body)
	identity := make(map[string]interface{})

	if err := dec.Decode(&identity); err != nil {
		return nil, err
	}

	return identity, nil
}
