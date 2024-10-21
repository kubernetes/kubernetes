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
	"fmt"
	"io"
	"sort"

	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
)

type Flag interface {
	Name() string
	Value() string
	String() string
}

type BaseFlag struct {
	FName  string
	FValue string
}

func (f BaseFlag) Name() string {
	return f.FName
}

func (f BaseFlag) Value() string {
	return f.FValue
}

func (f BaseFlag) String() string {
	return fmt.Sprintf("%s=%s", f.FName, f.FValue)
}

func ConvertNamedFlagSetToFlags(flagSets []*cliflag.NamedFlagSets) []Flag {
	var flags []Flag
	for _, flagset := range flagSets {
		for _, fs := range flagset.FlagSets {
			fs.VisitAll(func(flag *pflag.Flag) {
				if flag.Value != nil {
					value := flag.Value.String()
					if set, ok := flag.Annotations["classified"]; ok && len(set) > 0 {
						value = "CLASSIFIED"
					}
					flags = append(flags, BaseFlag{FName: flag.Name, FValue: value})
				}
			})
		}
	}

	sort.Slice(flags, func(i, j int) bool {
		return flags[i].Name() < flags[j].Name()
	})

	return flags
}

func printFlags(w io.Writer, flags []Flag) {
	for _, flag := range flags {
		fmt.Fprint(w, flag.String(), "\n")
	}
}
