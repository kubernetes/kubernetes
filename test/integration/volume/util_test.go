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

package volume

import (
	"encoding/json"
	"os"
	"path/filepath"

	"k8s.io/klog/v2"
)

func reportToArtifacts(filename string, obj any) {
	if os.Getenv("ARTIFACTS") == "" {
		return
	}
	path := filepath.Join(os.Getenv("ARTIFACTS"), filename)
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		klog.Error("Error opening file:", err)
		return
	}
	defer func() {
		if err := file.Close(); err != nil {
			klog.Error("Error closing file:", err)
		}
	}()

	content, err := json.Marshal(obj)
	if err != nil {
		klog.Error("Error marshalling to json:", err)
		return
	}
	content = append(content, '\n')
	_, err = file.Write(content)
	if err != nil {
		klog.Error("Error writing to file:", err)
		return
	}
}
