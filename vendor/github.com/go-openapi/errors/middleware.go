// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package errors

import (
	"bytes"
	"fmt"
	"strings"
)

// APIVerificationFailed is an error that contains all the missing info for a mismatched section
// between the api registrations and the api spec
type APIVerificationFailed struct {
	Section              string
	MissingSpecification []string
	MissingRegistration  []string
}

//
func (v *APIVerificationFailed) Error() string {
	buf := bytes.NewBuffer(nil)

	hasRegMissing := len(v.MissingRegistration) > 0
	hasSpecMissing := len(v.MissingSpecification) > 0

	if hasRegMissing {
		buf.WriteString(fmt.Sprintf("missing [%s] %s registrations", strings.Join(v.MissingRegistration, ", "), v.Section))
	}

	if hasRegMissing && hasSpecMissing {
		buf.WriteString("\n")
	}

	if hasSpecMissing {
		buf.WriteString(fmt.Sprintf("missing from spec file [%s] %s", strings.Join(v.MissingSpecification, ", "), v.Section))
	}

	return buf.String()
}
