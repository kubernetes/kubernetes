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

package modes

import (
	"github.com/fxamacker/cbor/v2"
)

var Diagnostic cbor.DiagMode = func() cbor.DiagMode {
	opts := Decode.DecOptions()
	diagnostic, err := cbor.DiagOptions{
		ByteStringText: true,

		MaxNestedLevels:  opts.MaxNestedLevels,
		MaxArrayElements: opts.MaxArrayElements,
		MaxMapPairs:      opts.MaxMapPairs,
	}.DiagMode()
	if err != nil {
		panic(err)
	}
	return diagnostic
}()
