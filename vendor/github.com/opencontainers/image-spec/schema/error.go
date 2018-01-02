// Copyright 2016 The Linux Foundation
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

package schema

import (
	"encoding/json"
	"io"

	"go4.org/errorutil"
)

// A SyntaxError is a description of a JSON syntax error
// including line, column and offset in the JSON file.
type SyntaxError struct {
	msg       string
	Line, Col int
	Offset    int64
}

func (e *SyntaxError) Error() string { return e.msg }

// WrapSyntaxError checks whether the given error is a *json.SyntaxError
// and converts it into a *schema.SyntaxError containing line/col information using the given reader.
// If the given error is not a *json.SyntaxError it is returned unchanged.
func WrapSyntaxError(r io.Reader, err error) error {
	if serr, ok := err.(*json.SyntaxError); ok {
		line, col, _ := errorutil.HighlightBytePosition(r, serr.Offset)
		return &SyntaxError{serr.Error(), line, col, serr.Offset}
	}

	return err
}
