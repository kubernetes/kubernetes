// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"net"
	"strconv"
)

// SplitHostPort splits a network address into a host and a port.
// The port is -1 when there is no port to be found
func SplitHostPort(addr string) (host string, port int, err error) {
	h, p, err := net.SplitHostPort(addr)
	if err != nil {
		return "", -1, err
	}
	if p == "" {
		return "", -1, &net.AddrError{Err: "missing port in address", Addr: addr}
	}

	pi, err := strconv.Atoi(p)
	if err != nil {
		return "", -1, err
	}
	return h, pi, nil
}
