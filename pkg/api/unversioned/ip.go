/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package unversioned

import (
	"encoding/json"
	"fmt"
	"net"
)

// IP is a wrapper around net.IP which supports correct marshaling to YAML and
// JSON.
type IP struct {
	net.IP
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (ip *IP) UnmarshalJSON(b []byte) error {
	var str string
	err := json.Unmarshal(b, &str)
	if err != nil {
		return err
	}

	pip := net.ParseIP(str)
	if pip == nil {
		return fmt.Errorf("%q is not a valid representation of an IP address", str)
	}

	ip.IP = pip
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (ip IP) MarshalJSON() ([]byte, error) {
	return json.Marshal(ip.IP.String())
}
