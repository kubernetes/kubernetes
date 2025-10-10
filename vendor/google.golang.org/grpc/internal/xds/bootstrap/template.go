/*
 *
 * Copyright 2021 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package bootstrap

import (
	"net/url"
	"strings"
)

// PopulateResourceTemplate populates the given template using the target
// string. "%s", if exists in the template, will be replaced with target.
//
// If the template starts with "xdstp:", the replaced string will be %-encoded.
// But note that "/" is not percent encoded.
func PopulateResourceTemplate(template, target string) string {
	if !strings.Contains(template, "%s") {
		return template
	}
	if strings.HasPrefix(template, "xdstp:") {
		target = percentEncode(target)
	}
	return strings.ReplaceAll(template, "%s", target)
}

// percentEncode percent encode t, except for "/". See the tests for examples.
func percentEncode(t string) string {
	segs := strings.Split(t, "/")
	for i := range segs {
		segs[i] = url.PathEscape(segs[i])
	}
	return strings.Join(segs, "/")
}
