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
	"fmt"
	"sort"
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
		if c.err == "" {
			t.Errorf("%d. Expected no error but got %q", i, err)
			continue
		}
		if !strings.Contains(err.Error(), c.err) {
			t.Errorf("%d. Expected error to contain %q but got %q", i, c.err, err)
		}
	}
}

func TestAlert(t *testing.T) {
	// Verifying that an alert with no EndsAt field is unresolved and has firing status.
	alert := &Alert{
		Labels:   LabelSet{"foo": "bar", "lorem": "ipsum"},
		StartsAt: time.Now(),
	}

	actual := fmt.Sprint(alert)
	expected := "[d181d0f][active]"

	if actual != expected {
		t.Errorf("expected %s, but got %s", expected, actual)
	}

	actualStatus := string(alert.Status())
	expectedStatus := "firing"

	if actualStatus != expectedStatus {
		t.Errorf("expected alertStatus %s, but got %s", expectedStatus, actualStatus)
	}

	// Verifying that an alert with an EndsAt time before the current time is resolved and has resolved status.
	ts := time.Now()
	ts1 := ts.Add(-2 * time.Minute)
	ts2 := ts.Add(-1 * time.Minute)
	alert = &Alert{
		Labels:   LabelSet{"foo": "bar", "lorem": "ipsum"},
		StartsAt: ts1,
		EndsAt:   ts2,
	}

	actual = fmt.Sprint(alert)
	expected = "[d181d0f][resolved]"

	if actual != expected {
		t.Errorf("expected %s, but got %s", expected, actual)
	}

	actualStatus = string(alert.Status())
	expectedStatus = "resolved"

	if actualStatus != expectedStatus {
		t.Errorf("expected alertStatus %s, but got %s", expectedStatus, actualStatus)
	}
}

func TestSortAlerts(t *testing.T) {
	ts := time.Now()
	alerts := Alerts{
		{
			Labels: LabelSet{
				"alertname": "InternalError",
				"dev":       "sda3",
			},
			StartsAt: ts.Add(-6 * time.Minute),
			EndsAt:   ts.Add(-3 * time.Minute),
		},
		{
			Labels: LabelSet{
				"alertname": "DiskFull",
				"dev":       "sda1",
			},
			StartsAt: ts.Add(-5 * time.Minute),
			EndsAt:   ts.Add(-4 * time.Minute),
		},
		{
			Labels: LabelSet{
				"alertname": "OutOfMemory",
				"dev":       "sda1",
			},
			StartsAt: ts.Add(-2 * time.Minute),
			EndsAt:   ts.Add(-1 * time.Minute),
		},
		{
			Labels: LabelSet{
				"alertname": "DiskFull",
				"dev":       "sda2",
			},
			StartsAt: ts.Add(-2 * time.Minute),
			EndsAt:   ts.Add(-3 * time.Minute),
		},
		{
			Labels: LabelSet{
				"alertname": "OutOfMemory",
				"dev":       "sda2",
			},
			StartsAt: ts.Add(-5 * time.Minute),
			EndsAt:   ts.Add(-2 * time.Minute),
		},
	}

	sort.Sort(alerts)

	expected := []string{
		"DiskFull[5ffe595][resolved]",
		"InternalError[09cfd46][resolved]",
		"OutOfMemory[d43a602][resolved]",
		"DiskFull[5ff4595][resolved]",
		"OutOfMemory[d444602][resolved]",
	}

	for i := range alerts {
		if alerts[i].String() != expected[i] {
			t.Errorf("expected alert %s at index %d, but got %s", expected[i], i, alerts[i].String())
		}
	}
}

func TestAlertsStatus(t *testing.T) {
	firingAlerts := Alerts{
		{
			Labels: LabelSet{
				"foo": "bar",
			},
			StartsAt: time.Now(),
		},
		{
			Labels: LabelSet{
				"bar": "baz",
			},
			StartsAt: time.Now(),
		},
	}

	actualStatus := firingAlerts.Status()
	expectedStatus := AlertFiring

	if actualStatus != expectedStatus {
		t.Errorf("expected status %s, but got %s", expectedStatus, actualStatus)
	}

	ts := time.Now()
	resolvedAlerts := Alerts{
		{
			Labels: LabelSet{
				"foo": "bar",
			},
			StartsAt: ts.Add(-1 * time.Minute),
			EndsAt:   ts,
		},
		{
			Labels: LabelSet{
				"bar": "baz",
			},
			StartsAt: ts.Add(-1 * time.Minute),
			EndsAt:   ts,
		},
	}

	actualStatus = resolvedAlerts.Status()
	expectedStatus = AlertResolved

	if actualStatus != expectedStatus {
		t.Errorf("expected status %s, but got %s", expectedStatus, actualStatus)
	}
}
