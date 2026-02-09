//go:build windows

/*
Copyright 2022 The Kubernetes Authors.

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
	"os/exec"
)

// CopyDir copies the content of a folder
func CopyDir(src string, dst string) ([]byte, error) {
	// /E Copies directories and subdirectories, including empty ones.
	// /H Copies hidden and system files also.
	return exec.Command("xcopy", "/E", "/H", src, dst).CombinedOutput()
}
