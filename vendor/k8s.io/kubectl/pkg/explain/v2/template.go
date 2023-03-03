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

package v2

import (
	"embed"
	"path/filepath"
	"strings"
)

//go:embed templates/*.tmpl
var rawBuiltinTemplates embed.FS

func registerBuiltinTemplates(gen Generator) error {
	files, err := rawBuiltinTemplates.ReadDir("templates")
	if err != nil {
		return err
	}

	for _, entry := range files {
		contents, err := rawBuiltinTemplates.ReadFile("templates/" + entry.Name())
		if err != nil {
			return err
		}

		err = gen.AddTemplate(
			strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name())),
			string(contents))

		if err != nil {
			return err
		}
	}

	return nil
}
