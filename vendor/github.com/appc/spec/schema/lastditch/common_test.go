// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package lastditch

import (
	"fmt"
	"strings"
)

// extJ returns a JSON snippet describing an extra field with a given
// name
func extJ(name string) string {
	return fmt.Sprintf(`"%s": [],`, name)
}

// labsJ returns a labels array JSON snippet with given labels
func labsJ(labels ...string) string {
	return fmt.Sprintf("[%s]", strings.Join(labels, ","))
}

// labsI returns a labels array instance with given labels
func labsI(labels ...Label) Labels {
	if labels == nil {
		return Labels{}
	}
	return labels
}

// labJ returns a label JSON snippet with given name and value
func labJ(name, value, extra string) string {
	return fmt.Sprintf(`
		{
		    %s
		    "name": "%s",
		    "value": "%s"
		}`, extra, name, value)
}

// labI returns a label instance with given name and value
func labI(name, value string) Label {
	return Label{
		Name:  name,
		Value: value,
	}
}
