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

// An Error contains detailed information about an error encountered while processing a job.
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
