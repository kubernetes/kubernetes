// Copyright 2016 The rkt Authors
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

package common

import (
	"strings"
	"testing"

	cnitypes "github.com/containernetworking/cni/pkg/types"
)

func TestMakeResolvConf(t *testing.T) {
	tests := []struct {
		dns     cnitypes.DNS
		comment string
		result  string
	}{
		{
			cnitypes.DNS{
				Nameservers: []string{"1.2.3.4"},
				Domain:      "mydomain",
				Search:      []string{"example.com"},
				Options:     []string{"testoption"},
			},
			"foobar",
			`
# foobar

search example.com
nameserver 1.2.3.4
options testoption
domain mydomain
`,
		},
		{
			cnitypes.DNS{
				Nameservers: []string{"1.2.3.4", "1.2.3.5"},
				Domain:      "mydomain",
				Search:      []string{"example.com", "example.org"},
				Options:     []string{"testoption", "ipv6"},
			},
			"this is a comment",
			`
# this is a comment

search example.com example.org
nameserver 1.2.3.4
nameserver 1.2.3.5
options testoption ipv6
domain mydomain
`,
		},
		{
			cnitypes.DNS{
				Nameservers: []string{"1.2.3.4", "1.2.3.5"},
			},
			"",
			`
nameserver 1.2.3.4
nameserver 1.2.3.5
`,
		},
	}

	// trim whitespace for legibility
	for i, tt := range tests {
		result := strings.TrimSpace(MakeResolvConf(tt.dns, tt.comment))
		want := strings.TrimSpace(tt.result)
		if result != want {
			t.Errorf("#%d: got %v, want %v", i, result, want)
		}
	}
}
