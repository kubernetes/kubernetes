/*
Copyright 2018 The Kubernetes Authors.

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

package add

import (
	"log"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func globPatterns(fsys fs.FileSystem, patterns []string) ([]string, error) {
	var result []string
	for _, pattern := range patterns {
		files, err := fsys.Glob(pattern)
		if err != nil {
			return nil, err
		}
		if len(files) == 0 {
			log.Printf("%s has no match", pattern)
			continue
		}
		result = append(result, files...)
	}
	return result, nil
}
