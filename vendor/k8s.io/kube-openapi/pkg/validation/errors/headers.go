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

// Validation represents a failure of a precondition
type Validation struct {
	code    int32
	Name    string
	In      string
	Value   interface{}
	Valid   interface{}
	message string
	Values  []interface{}
}

func (e *Validation) Error() string {
	return e.message
}

// Code the error code
func (e *Validation) Code() int32 {
	return e.code
}

// ValidateName produces an error message name for an aliased property
func (e *Validation) ValidateName(name string) *Validation {
	if e.Name == "" && name != "" {
		e.Name = name
		e.message = name + e.message
	}
	return e
}
