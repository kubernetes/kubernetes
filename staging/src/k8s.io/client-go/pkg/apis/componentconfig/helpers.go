/*
Copyright 2016 The Kubernetes Authors.

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

package componentconfig

import (
	"fmt"
	"net"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

// used for validating command line opts
// TODO(mikedanese): remove these when we remove command line flags

type IPVar struct {
	Val *string
}

func (v IPVar) Set(s string) error {
	if net.ParseIP(s) == nil {
		return fmt.Errorf("%q is not a valid IP address", s)
	}
	if v.Val == nil {
		// it's okay to panic here since this is programmer error
		panic("the string pointer passed into IPVar should not be nil")
	}
	*v.Val = s
	return nil
}

func (v IPVar) String() string {
	if v.Val == nil {
		return ""
	}
	return *v.Val
}

func (v IPVar) Type() string {
	return "ip"
}

func (m *ProxyMode) Set(s string) error {
	*m = ProxyMode(s)
	return nil
}

func (m *ProxyMode) String() string {
	if m != nil {
		return string(*m)
	}
	return ""
}

func (m *ProxyMode) Type() string {
	return "ProxyMode"
}

type PortRangeVar struct {
	Val *string
}

func (v PortRangeVar) Set(s string) error {
	if _, err := utilnet.ParsePortRange(s); err != nil {
		return fmt.Errorf("%q is not a valid port range: %v", s, err)
	}
	if v.Val == nil {
		// it's okay to panic here since this is programmer error
		panic("the string pointer passed into PortRangeVar should not be nil")
	}
	*v.Val = s
	return nil
}

func (v PortRangeVar) String() string {
	if v.Val == nil {
		return ""
	}
	return *v.Val
}

func (v PortRangeVar) Type() string {
	return "port-range"
}
