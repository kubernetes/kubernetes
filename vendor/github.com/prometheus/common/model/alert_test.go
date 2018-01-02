// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"strings"
	"testing"
	"time"
)

func TestAlertValidate(t *testing.T) {
	ts := time.Now()

	var cases = []struct {
		alert *Alert
		err   string
	}{
		{
			alert: &Alert{
				Labels:   LabelSet{"a": "b"},
				StartsAt: ts,
			},
		},
		{
			alert: &Alert{
				Labels: LabelSet{"a": "b"},
			},
			err: "start time missing",
		},
		{
			alert: &Alert{
				Labels:   LabelSet{"a": "b"},
				StartsAt: ts,
				EndsAt:   ts,
			},
		},
		{
			alert: &Alert{
				Labels:   LabelSet{"a": "b"},
				StartsAt: ts,
				EndsAt:   ts.Add(1 * time.Minute),
			},
		},
		{
			alert: &Alert{
				Labels:   LabelSet{"a": "b"},
				StartsAt: ts,
				EndsAt:   ts.Add(-1 * time.Minute),
			},
			err: "start time must be before end time",
		},
		{
			alert: &Alert{
				StartsAt: ts,
			},
			err: "at least one label pair required",
		},
		{
			alert: &Alert{
				Labels:   LabelSet{"a": "b", "!bad": "label"},
				StartsAt: ts,
			},
			err: "invalid label set: invalid name",
		},
		{
			alert: &Alert{
				Labels:   LabelSet{"a": "b", "bad": "\xfflabel"},
				StartsAt: ts,
			},
			err: "invalid label set: invalid value",
		},
		{
			alert: &Alert{
				Labels:      LabelSet{"a": "b"},
				Annotations: LabelSet{"!bad": "label"},
				StartsAt:    ts,
			},
			err: "invalid annotations: invalid name",
		},
		{
			alert: &Alert{
				Labels:      LabelSet{"a": "b"},
				Annotations: LabelSet{"bad": "\xfflabel"},
				StartsAt:    ts,
			},
			err: "invalid annotations: invalid value",
		},
	}

	for i, c := range cases {
		err := c.alert.Validate()
		if err == nil {
			if c.err == "" {
				continue
			}
			t.Errorf("%d. Expected error %q but got none", i, c.err)
			continue
		}
		if c.err == "" && err != nil {
			t.Errorf("%d. Expected no error but got %q", i, err)
			continue
		}
		if !strings.Contains(err.Error(), c.err) {
			t.Errorf("%d. Expected error to contain %q but got %q", i, c.err, err)
		}
	}
}
