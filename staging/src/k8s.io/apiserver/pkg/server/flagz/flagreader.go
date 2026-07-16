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

package flagz

import (
	"github.com/spf13/pflag"
	cliflag "k8s.io/component-base/cli/flag"
)

type Reader interface {
	GetFlagz() map[string]string
}

// NamedFlagSetsReader implements Reader for cliflag.NamedFlagSets
type NamedFlagSetsReader struct {
	FlagSets cliflag.NamedFlagSets
}

func (n NamedFlagSetsReader) GetFlagz() map[string]string {
	return convertNamedFlagSetToFlags(&n.FlagSets)
}

func convertNamedFlagSetToFlags(flagSets *cliflag.NamedFlagSets) map[string]string {
	flags := make(map[string]string)
	for _, fs := range flagSets.FlagSets {
		fs.VisitAll(func(flag *pflag.Flag) {
			if flag.Value != nil {
				value := flag.Value.String()
				if set, ok := flag.Annotations["classified"]; ok && len(set) > 0 {
					value = "CLASSIFIED"
				}
				flags[flag.Name] = value
			}
		})
	}

	return flags
}
