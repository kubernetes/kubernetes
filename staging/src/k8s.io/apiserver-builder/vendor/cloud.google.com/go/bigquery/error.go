// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import (
	"fmt"

	bq "google.golang.org/api/bigquery/v2"
)

// An Error contains detailed information about a failed bigquery operation.
type Error struct {
	// Mirrors bq.ErrorProto, but drops DebugInfo
	Location, Message, Reason string
}

func (e Error) Error() string {
	return fmt.Sprintf("{Location: %q; Message: %q; Reason: %q}", e.Location, e.Message, e.Reason)
}

func errorFromErrorProto(ep *bq.ErrorProto) *Error {
	if ep == nil {
		return nil
	}
	return &Error{
		Location: ep.Location,
		Message:  ep.Message,
		Reason:   ep.Reason,
	}
}

// A MultiError contains multiple related errors.
type MultiError []error

func (m MultiError) Error() string {
	switch len(m) {
	case 0:
		return "(0 errors)"
	case 1:
		return m[0].Error()
	case 2:
		return m[0].Error() + " (and 1 other error)"
	}
	return fmt.Sprintf("%s (and %d other errors)", m[0].Error(), len(m)-1)
}

// RowInsertionError contains all errors that occurred when attempting to insert a row.
type RowInsertionError struct {
	InsertID string // The InsertID associated with the affected row.
	RowIndex int    // The 0-based index of the affected row in the batch of rows being inserted.
	Errors   MultiError
}

func (e *RowInsertionError) Error() string {
	errFmt := "insertion of row [insertID: %q; insertIndex: %v] failed with error: %s"
	return fmt.Sprintf(errFmt, e.InsertID, e.RowIndex, e.Errors.Error())
}

// PutMultiError contains an error for each row which was not successfully inserted
// into a BigQuery table.
type PutMultiError []RowInsertionError

func (pme PutMultiError) Error() string {
	plural := "s"
	if len(pme) == 1 {
		plural = ""
	}

	return fmt.Sprintf("%v row insertion%s failed", len(pme), plural)
}
