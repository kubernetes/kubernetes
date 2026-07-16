/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"strings"
)

// PluralExceptionListToMapOrDie converts the list in "Type:PluralType" to map[string]string.
// This is used for pluralizer.
// If the format is wrong, this function will panic.
func PluralExceptionListToMapOrDie(pluralExceptions []string) map[string]string {
	pluralExceptionMap := make(map[string]string, len(pluralExceptions))
	for i := range pluralExceptions {
		parts := strings.Split(pluralExceptions[i], ":")
		if len(parts) != 2 {
			panic(fmt.Sprintf("invalid plural exception definition: %s", pluralExceptions[i]))
		}
		pluralExceptionMap[parts[0]] = parts[1]
	}
	return pluralExceptionMap
}
