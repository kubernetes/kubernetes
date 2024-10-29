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

package llcalign

import (
	"os"
)

var (
	llcAlignmentEnabled  bool
	llcAlignmentFilename = "/etc/kubernetes/openshift-llc-alignment"
)

func init() {
	readEnablementFile()
}

func readEnablementFile() {
	if _, err := os.Stat(llcAlignmentFilename); err == nil {
		llcAlignmentEnabled = true
	}
}

func IsEnabled() bool {
	return llcAlignmentEnabled
}

func TestOnlySetEnabled(enabled bool) bool {
	oldEnabled := llcAlignmentEnabled
	llcAlignmentEnabled = enabled
	return oldEnabled
}
