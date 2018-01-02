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
	"fmt"
	"strings"

	cnitypes "github.com/containernetworking/cni/pkg/types"
)

// MakeResolvConf generates resolv.conf contents given a cni DNS configuration
func MakeResolvConf(dns cnitypes.DNS, comment string) string {
	content := ""
	if len(comment) > 0 {
		content += fmt.Sprintf("# %s\n\n", comment)
	}

	if len(dns.Search) > 0 {
		content += fmt.Sprintf("search %s\n", strings.Join(dns.Search, " "))
	}

	for _, ns := range dns.Nameservers {
		content += fmt.Sprintf("nameserver %s\n", ns)
	}
	if len(dns.Options) > 0 {
		content += fmt.Sprintf("options %s\n", strings.Join(dns.Options, " "))
	}

	if len(dns.Domain) > 0 {
		content += fmt.Sprintf("domain %s\n", dns.Domain)
	}
	return content
}

/*
 * TODO(cdc) move this to cnitypes
 */
// IsDNSZero checks if the DNS configuration has any information
func IsDNSZero(dns *cnitypes.DNS) bool {
	return len(dns.Nameservers) == 0 &&
		len(dns.Domain) == 0 &&
		len(dns.Search) == 0 &&
		len(dns.Options) == 0
}
