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

package types

import "fmt"

// An ACKindError is returned when the wrong ACKind is set in a manifest
type ACKindError string

func (e ACKindError) Error() string {
	return string(e)
}

func InvalidACKindError(kind ACKind) ACKindError {
	return ACKindError(fmt.Sprintf("missing or bad ACKind (must be %#v)", kind))
}

// An ACVersionError is returned when a bad ACVersion is set in a manifest
type ACVersionError string

func (e ACVersionError) Error() string {
	return string(e)
}

// An ACIdentifierError is returned when a bad value is used for an ACIdentifier
type ACIdentifierError string

func (e ACIdentifierError) Error() string {
	return string(e)
}

// An ACNameError is returned when a bad value is used for an ACName
type ACNameError string

func (e ACNameError) Error() string {
	return string(e)
}
