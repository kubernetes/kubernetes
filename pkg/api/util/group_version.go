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

// TODO: This GetVersion/GetGroup arrangement is temporary and will be replaced
// with a GroupAndVersion type.
package util

import "strings"

func GetVersion(groupVersion string) string {
	s := strings.Split(groupVersion, "/")
	if len(s) != 2 {
		// e.g. return "v1" for groupVersion="v1"
		return s[len(s)-1]
	}
	return s[1]
}

func GetGroup(groupVersion string) string {
	s := strings.Split(groupVersion, "/")
	if len(s) == 1 {
		// e.g. return "" for groupVersion="v1"
		return ""
	}
	return s[0]
}
