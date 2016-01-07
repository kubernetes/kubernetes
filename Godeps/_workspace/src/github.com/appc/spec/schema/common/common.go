// Copyright 2015 The appc Authors
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
	"net/url"
	"strings"
)

// MakeQueryString takes a comma-separated LABEL=VALUE string and returns an
// "&"-separated string with URL escaped values.
//
// Examples:
// 	version=1.0.0,label=v1+v2 -> version=1.0.0&label=v1%2Bv2
// 	name=db,source=/tmp$1 -> name=db&source=%2Ftmp%241
func MakeQueryString(app string) (string, error) {
	parts := strings.Split(app, ",")
	escapedParts := make([]string, len(parts))
	for i, s := range parts {
		p := strings.SplitN(s, "=", 2)
		if len(p) != 2 {
			return "", fmt.Errorf("malformed string %q - has a label without a value: %s", app, p[0])
		}
		escapedParts[i] = fmt.Sprintf("%s=%s", p[0], url.QueryEscape(p[1]))
	}
	return strings.Join(escapedParts, "&"), nil
}
