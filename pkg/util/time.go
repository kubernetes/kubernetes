/*
Copyright 2014 Google Inc. All rights reserved.

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

package util

import (
	"encoding/json"
	"time"
)

// Time is a wrapper around time.Time which supports correct
// marshaling to YAML and JSON.  Wrappers are provided for many
// of the factory methods that the time package offers.
type Time struct {
	time.Time
}

// Date returns the Time corresponding to the supplied parameters
// by wrapping time.Date.
func Date(year int, month time.Month, day, hour, min, sec, nsec int, loc *time.Location) Time {
	return Time{time.Date(year, month, day, hour, min, sec, nsec, loc)}
}

// Now returns the current local time.
func Now() Time {
	return Time{time.Now()}
}

// Unix returns the local time corresponding to the given Unix time
// by wrapping time.Unix.
func Unix(sec int64, nsec int64) Time {
	return Time{time.Unix(sec, nsec)}
}

// Rfc3339Copy returns a copy of the Time at second-level precision.
func (t Time) Rfc3339Copy() Time {
	copied, _ := time.Parse(time.RFC3339, t.Format(time.RFC3339))
	return Time{copied}
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (t *Time) UnmarshalJSON(b []byte) error {
	if len(b) == 4 && string(b) == "null" {
		t.Time = time.Time{}
		return nil
	}

	var str string
	json.Unmarshal(b, &str)

	pt, err := time.Parse(time.RFC3339, str)
	if err != nil {
		return err
	}

	t.Time = pt
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (t Time) MarshalJSON() ([]byte, error) {
	if t.IsZero() {
		// Encode unset/nil objects as JSON's "null".
		return []byte("null"), nil
	}

	return json.Marshal(t.Format(time.RFC3339))
}

// SetYAML implements the yaml.Setter interface.
func (t *Time) SetYAML(tag string, value interface{}) bool {
	if value == nil {
		t.Time = time.Time{}
		return true
	}

	str, ok := value.(string)
	if !ok {
		return false
	}

	pt, err := time.Parse(time.RFC3339, str)
	if err != nil {
		return false
	}

	t.Time = pt
	return true
}

// GetYAML implements the yaml.Getter interface.
func (t Time) GetYAML() (tag string, value interface{}) {
	if t.IsZero() {
		value = "null"
		return
	}

	value = t.Format(time.RFC3339)
	return tag, value
}
