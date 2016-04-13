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

package flags

import (
	"errors"
	"net"
	"strconv"
	"strings"
)

// IPAddressPort implements the flag.Value interface. The argument
// is validated as "ip:port".
type IPAddressPort struct {
	IP   string
	Port int
}

func (a *IPAddressPort) Set(arg string) error {
	arg = strings.TrimSpace(arg)

	host, portStr, err := net.SplitHostPort(arg)
	if err != nil {
		return err
	}

	if net.ParseIP(host) == nil {
		return errors.New("bad IP in address specification")
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return errors.New("bad port in address specification")
	}

	a.IP = host
	a.Port = port

	return nil
}

func (a *IPAddressPort) String() string {
	return net.JoinHostPort(a.IP, strconv.Itoa(a.Port))
}
