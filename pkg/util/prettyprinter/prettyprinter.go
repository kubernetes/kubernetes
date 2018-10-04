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

package prettyprinter

import (
	"github.com/davecgh/go-spew/spew"
)

var spewWithoutMethods = spew.ConfigState{DisableMethods: true}

// Sprint will return a pretified string version of an object
func Sprint(vals ...interface{}) string {
	format := ""
	for i := 0; i < len(vals); i++ {
		format += "%#v "
	}
	return spew.Sprintf(format, vals...)
}

// Sprintf will return a pretified formated string version of an object
func Sprintf(format string, vals ...interface{}) string {
	return spew.Sprintf(format, vals...)
}

// SprintWithoutMethods will return a pretified string version of an object
func SprintWithoutMethods(vals ...interface{}) string {
	format := ""
	for i := 0; i < len(vals); i++ {
		format += "%#v "
	}
	return spewWithoutMethods.Sprintf(format, vals...)
}

// SprintfWithoutMethods will return a pretified formated string version of an object
func SprintfWithoutMethods(format string, vals ...interface{}) string {
	return spewWithoutMethods.Sprintf(format, vals...)
}
