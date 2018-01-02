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

import (
	"encoding/json"
	"testing"
	"time"
)

var (
	pst = time.FixedZone("Pacific", -8*60*60)
)

func TestUnmarshalDate(t *testing.T) {
	tests := []struct {
		in string

		wt time.Time
	}{
		{
			`"2004-05-14T23:11:14+00:00"`,

			time.Date(2004, 05, 14, 23, 11, 14, 0, time.UTC),
		},
		{
			`"2001-02-03T04:05:06Z"`,

			time.Date(2001, 02, 03, 04, 05, 06, 0, time.UTC),
		},
		{
			`"2014-11-14T17:36:54-08:00"`,

			time.Date(2014, 11, 14, 17, 36, 54, 0, pst),
		},
		{
			`"2004-05-14T23:11:14+00:00"`,

			time.Date(2004, 05, 14, 23, 11, 14, 0, time.UTC),
		},
	}
	for i, tt := range tests {
		var d Date
		if err := json.Unmarshal([]byte(tt.in), &d); err != nil {
			t.Errorf("#%d: got err=%v, want nil", i, err)
		}
		if gt := time.Time(d); !gt.Equal(tt.wt) {
			t.Errorf("#%d: got time=%v, want %v", i, gt, tt.wt)
		}
	}
}

func TestUnmarshalDateBad(t *testing.T) {
	tests := []string{
		`not a json string`,
		`2014-11-14T17:36:54-08:00`,
		`"garbage"`,
		`"1416015188"`,
		`"Fri Nov 14 17:53:02 PST 2014"`,
		`"2014-11-1417:36:54"`,
	}
	for i, tt := range tests {
		var d Date
		if err := json.Unmarshal([]byte(tt), &d); err == nil {
			t.Errorf("#%d: unexpected nil err", i)
		}
	}
}
