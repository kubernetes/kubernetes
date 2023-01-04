/*
Copyright 2021 The Kubernetes Authors.

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

package dump

import (
	"github.com/davecgh/go-spew/spew"
)

var prettyPrintConfig = &spew.ConfigState{
	Indent:                  "  ",
	DisableMethods:          true,
	DisablePointerAddresses: true,
	DisableCapacities:       true,
}

// The config MUST NOT be changed because that could change the result of a hash operation
var prettyPrintConfigForHash = &spew.ConfigState{
	Indent:                  " ",
	SortKeys:                true,
	DisableMethods:          true,
	SpewKeys:                true,
	DisablePointerAddresses: true,
	DisableCapacities:       true,
}

// Pretty wrap the spew.Sdump with Indent, and disabled methods like error() and String()
// The output may change over time, so for guaranteed output please take more direct control
func Pretty(a interface{}) string {
	return prettyPrintConfig.Sdump(a)
}

// ForHash keeps the original Spew.Sprintf format to ensure the same checksum
func ForHash(a interface{}) string {
	return prettyPrintConfigForHash.Sprintf("%#v", a)
}

// OneLine outputs the object in one line
func OneLine(a interface{}) string {
	return prettyPrintConfig.Sprintf("%#v", a)
}
