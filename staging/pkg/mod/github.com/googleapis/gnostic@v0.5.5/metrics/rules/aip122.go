// Copyright 2020 Google LLC. All Rights Reserved.
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

package rules

import (
	"fmt"
	"strings"
)

type Field struct {
	Name string
	Path []string
}

type MessageType struct {
	Message []string
	Path    []string
}

// checkNameSuffix ensures that the name of the field does not
// end in "_name"
func checkNameSuffix(name string) (bool, string) {
	if strings.HasSuffix(name, "_name") {
		return true, name[:len(name)-5]
	}
	return false, name
}

// AIP122Driver calls all functions for AIP rule 122
func AIP122Driver(f Field) []MessageType {
	messages := make([]MessageType, 0)
	val, sugg := checkNameSuffix(f.Name)
	if val {
		m := []string{"Error", "Message: Parameters must not use the suffix \"_name\"\n",
			fmt.Sprintf("Suggestion: Rename field %s to %s\n", f.Name, sugg)}
		temp := MessageType{Message: m, Path: f.Path}
		messages = append(messages, temp)

	}
	return messages
}
