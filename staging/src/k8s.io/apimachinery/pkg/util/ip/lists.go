/*
Copyright 2024 The Kubernetes Authors.

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

package ip

import (
	"fmt"
	"strings"
)

type parser[T any] func(string) (T, error)

// Split parses a list of IPs or CIDRs delimited by sep, using the provided parser
// (ParseIP, ParseCIDR, ParseValidIP, ParseValidCIDR, ParseCanonicalIP, or
// ParseCanonicalCIDR). If list is the empty string, this will return an empty list of
// IPs/CIDRs.
func Split[T any](list, sep string, parser parser[T]) ([]T, error) {
	var err error

	if list == "" {
		return []T{}, nil
	}

	strs := strings.Split(list, sep)
	out := make([]T, len(strs))
	for i := range strs {
		out[i], err = parser(strs[i])
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Join joins a list of net.IP, net.IPNet, netip.Addr, or netip.Prefix, using the given
// separator
func Join[T fmt.Stringer](vals []T, sep string) string {
	var b strings.Builder

	for i := range vals {
		if i > 0 {
			b.WriteString(sep)
		}
		b.WriteString(vals[i].String())
	}
	return b.String()
}
