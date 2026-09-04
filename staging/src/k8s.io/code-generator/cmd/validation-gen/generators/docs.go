/*
Copyright The Kubernetes Authors.

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

package generators

import (
	"cmp"
	"encoding/json"
	"io"
	"slices"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
)

// PrintDocs writes the documentation of every registered validation tag, as
// JSON, to w. Tag names are qualified with tagPrefix. It initializes the
// global tag registry, so it must not be combined with GetTargets in the same
// process.
func PrintDocs(w io.Writer, tagPrefix string) error {
	// We need a fake context to init the validator plugins.
	c := &generator.Context{
		Namers:    namer.NameSystems{},
		Universe:  types.Universe{},
		FileTypes: map[string]generator.FileType{},
	}

	// Initialize all registered validators.
	validator := validators.InitGlobalValidator(c, nil, tagPrefix)

	docs := validator.Docs()
	for i := range docs {
		d := &docs[i]
		slices.Sort(d.Scopes)
		if d.Usage == "" {
			// Try to generate a usage string if none was provided.
			usage := d.Tag
			if len(d.Args) > 0 {
				usage += "("
				for i := range d.Args {
					if i > 0 {
						usage += ", "
					}
					usage += d.Args[i].Description
				}
				usage += ")"
			}
			if len(d.Payloads) > 0 {
				usage += "="
				if len(d.Payloads) == 1 {
					usage += d.Payloads[0].Description
				} else {
					usage += "<payload>"
				}
			}
			d.Usage = usage
		}
	}
	slices.SortFunc(docs, func(a, b validators.TagDoc) int {
		return cmp.Compare(a.Tag, b.Tag)
	})

	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "    ")
	return encoder.Encode(docs)
}
