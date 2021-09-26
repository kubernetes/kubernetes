// Copyright 2015 The Prometheus Authors
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

func TestMatcherValidate(t *testing.T) {
	var cases = []struct {
		matcher *Matcher
		err     string
	}{
		{
			matcher: &Matcher{
				Name:  "name",
				Value: "value",
			},
		},
		{
			matcher: &Matcher{
				Name:    "name",
				Value:   "value",
				IsRegex: true,
			},
		},
		{
			matcher: &Matcher{
				Name:  "name!",
				Value: "value",
			},
			err: "invalid name",
		},
		{
			matcher: &Matcher{
				Name:  "",
				Value: "value",
			},
			err: "invalid name",
		},
		{
			matcher: &Matcher{
				Name:  "name",
				Value: "value\xff",
			},
			err: "invalid value",
		},
		{
			matcher: &Matcher{
				Name:  "name",
				Value: "",
			},
			err: "invalid value",
		},
	}

	for i, c := range cases {
		err := c.matcher.Validate()
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

func TestSilenceValidate(t *testing.T) {
	ts := time.Now()

	var cases = []struct {
		sil *Silence
		err string
	}{
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedAt: ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
					{Name: "name", Value: "value"},
					{Name: "name", Value: "value"},
					{Name: "name", Value: "value", IsRegex: true},
				},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedAt: ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				StartsAt:  ts,
				EndsAt:    ts.Add(-1 * time.Minute),
				CreatedAt: ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
			err: "start time must be before end time",
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				StartsAt:  ts,
				CreatedAt: ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
			err: "end time missing",
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				EndsAt:    ts,
				CreatedAt: ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
			err: "start time missing",
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "!name", Value: "value"},
				},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedAt: ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
			err: "invalid matcher",
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedAt: ts,
				CreatedBy: "name",
			},
			err: "comment missing",
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedBy: "name",
				Comment:   "comment",
			},
			err: "creation timestamp missing",
		},
		{
			sil: &Silence{
				Matchers: []*Matcher{
					{Name: "name", Value: "value"},
				},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedAt: ts,
				Comment:   "comment",
			},
			err: "creator information missing",
		},
		{
			sil: &Silence{
				Matchers:  []*Matcher{},
				StartsAt:  ts,
				EndsAt:    ts,
				CreatedAt: ts,
				Comment:   "comment",
			},
			err: "at least one matcher required",
		},
	}

	for i, c := range cases {
		err := c.sil.Validate()
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
