/*
Copyright 2021 The Kubernetes Authors.

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

package routes

import (
	"fmt"
	"os"
	"testing"
)

func TestPreCheckLogFileNameLength(t *testing.T) {
	oversizeFileName := fmt.Sprintf("%0256s", "a")
	normalFileName := fmt.Sprintf("%0255s", "a")

	// check file with oversize name.
	if !logFileNameIsTooLong(oversizeFileName) {
		t.Error("failed to check oversize filename")
	}

	// check file with normal name which doesn't exist.
	if logFileNameIsTooLong(normalFileName) {
		t.Error("failed to check normal filename")
	}

	// check file with normal name which does exist.
	_, err := os.Create(normalFileName)
	if err != nil {
		t.Error("failed to create test file")
	}
	defer os.Remove(normalFileName)
	if logFileNameIsTooLong(normalFileName) {
		t.Error("failed to check normal filename")
	}
}
