/*
Copyright 2023 The Kubernetes Authors.

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

package helpers

import (
	"reflect"

	"github.com/spf13/pflag"
	"k8s.io/code-generator/pkg/codegen/helpers"
)

func (c Command) Name() string {
	return "gen-helpers"
}

func (c Command) OneLine() string {
	return "Generate tagged helper code: conversions, deepcopy, and defaults"
}

func (c Command) Help() string {
	args := &helpers.Args{}
	fs := pflag.NewFlagSet("help", pflag.ContinueOnError)
	defineFlags(fs, args)
	return "Usage: code-generator " + c.Name() + " [options] <input-dir>\n" +
		"\n" +
		c.OneLine() + "\n" +
		"\n" +
		"Arguments:\n" +
		"      <input-dir>                " + inputDirDoc(args) + "\n" +
		"\n" +
		"Options:\n" + fs.FlagUsagesWrapped(100)
}

func inputDirDoc(args *helpers.Args) string {
	ty := reflect.TypeOf(*args)
	if f, ok := ty.FieldByName("InputDir"); ok {
		if usage, ook := f.Tag.Lookup("input-dir"); ook {
			return usage
		}
	}
	return "The input directory to read the input files from"
}
