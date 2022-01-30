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

package framework

import (
	"encoding/json"
	"fmt"
)

// FormatJSON is a helper that prints the inner object in json format.
//
// It can be used with log statements, for example:
//  klog.Infof("object is %s", framework.FormatJSON(o))
//
// It returns an object, to avoid json overhead unless the value is actually printed.
func FormatJSON(o interface{}) formatJSON {
	return formatJSON{object: o}
}

// formatJSON is the helper type that defers json rendering until it is needed.
type formatJSON struct {
	object interface{}
}

// formatJSON implements fmt.Stringer
var _ fmt.Stringer = formatJSON{}

// String implements the Stringer interface
func (f formatJSON) String() string {
	b, err := json.Marshal(f.object)
	if err != nil {
		return fmt.Sprintf("<error converting to json: %v>", err)
	}
	return string(b)
}
