/*
Copyright 2018 The Kubernetes Authors.

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

package duration

import (
	"testing"
	"time"
)

func TestHumanDuration(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{d: time.Second, want: "1s"},
		{d: 70 * time.Second, want: "70s"},
		{d: 190 * time.Second, want: "3m10s"},
		{d: 70 * time.Minute, want: "70m"},
		{d: 47 * time.Hour, want: "47h"},
		{d: 49 * time.Hour, want: "2d1h"},
		{d: (8*24 + 2) * time.Hour, want: "8d"},
		{d: (367 * 24) * time.Hour, want: "367d"},
		{d: (365*2*24 + 25) * time.Hour, want: "2y1d"},
		{d: (365*8*24 + 2) * time.Hour, want: "8y"},
	}
	for _, tt := range tests {
		t.Run(tt.d.String(), func(t *testing.T) {
			if got := HumanDuration(tt.d); got != tt.want {
				t.Errorf("HumanDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHumanDurationBoundaries(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{d: -2 * time.Second, want: "<invalid>"},
		{d: -2*time.Second + 1, want: "0s"},
		{d: 0, want: "0s"},
		{d: time.Second - time.Millisecond, want: "0s"},
		{d: 2*time.Minute - time.Millisecond, want: "119s"},
		{d: 2 * time.Minute, want: "2m"},
		{d: 2*time.Minute + time.Second, want: "2m1s"},
		{d: 10*time.Minute - time.Millisecond, want: "9m59s"},
		{d: 10 * time.Minute, want: "10m"},
		{d: 10*time.Minute + time.Second, want: "10m"},
		{d: 3*time.Hour - time.Millisecond, want: "179m"},
		{d: 3 * time.Hour, want: "3h"},
		{d: 3*time.Hour + time.Minute, want: "3h1m"},
		{d: 8*time.Hour - time.Millisecond, want: "7h59m"},
		{d: 8 * time.Hour, want: "8h"},
		{d: 8*time.Hour + 59*time.Minute, want: "8h"},
		{d: 2*24*time.Hour - time.Millisecond, want: "47h"},
		{d: 2 * 24 * time.Hour, want: "2d"},
		{d: 2*24*time.Hour + time.Hour, want: "2d1h"},
		{d: 8*24*time.Hour - time.Millisecond, want: "7d23h"},
		{d: 8 * 24 * time.Hour, want: "8d"},
		{d: 8*24*time.Hour + 23*time.Hour, want: "8d"},
		{d: 2*365*24*time.Hour - time.Millisecond, want: "729d"},
		{d: 2 * 365 * 24 * time.Hour, want: "2y"},
		{d: 2*365*24*time.Hour + 23*time.Hour, want: "2y"},
		{d: 2*365*24*time.Hour + 23*time.Hour + 59*time.Minute, want: "2y"},
		{d: 2*365*24*time.Hour + 24*time.Hour - time.Millisecond, want: "2y"},
		{d: 2*365*24*time.Hour + 24*time.Hour, want: "2y1d"},
		{d: 3 * 365 * 24 * time.Hour, want: "3y"},
		{d: 4 * 365 * 24 * time.Hour, want: "4y"},
		{d: 5 * 365 * 24 * time.Hour, want: "5y"},
		{d: 6 * 365 * 24 * time.Hour, want: "6y"},
		{d: 7 * 365 * 24 * time.Hour, want: "7y"},
		{d: 8*365*24*time.Hour - time.Millisecond, want: "7y364d"},
		{d: 8 * 365 * 24 * time.Hour, want: "8y"},
		{d: 8*365*24*time.Hour + 364*24*time.Hour, want: "8y"},
		{d: 9 * 365 * 24 * time.Hour, want: "9y"},
	}
	for _, tt := range tests {
		t.Run(tt.d.String(), func(t *testing.T) {
			if got := HumanDuration(tt.d); got != tt.want {
				t.Errorf("HumanDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestShortHumanDurationBoundaries(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{d: -2 * time.Second, want: "<invalid>"},
		{d: -2*time.Second + 1, want: "0s"},
		{d: 0, want: "0s"},
		{d: time.Second - time.Millisecond, want: "0s"},
		{d: time.Second, want: "1s"},
		{d: 2*time.Second - time.Millisecond, want: "1s"},
		{d: time.Minute - time.Millisecond, want: "59s"},
		{d: time.Minute, want: "1m"},
		{d: 2*time.Minute - time.Millisecond, want: "1m"},
		{d: time.Hour - time.Millisecond, want: "59m"},
		{d: time.Hour, want: "1h"},
		{d: 2*time.Hour - time.Millisecond, want: "1h"},
		{d: 24*time.Hour - time.Millisecond, want: "23h"},
		{d: 24 * time.Hour, want: "1d"},
		{d: 2*24*time.Hour - time.Millisecond, want: "1d"},
		{d: 365*24*time.Hour - time.Millisecond, want: "364d"},
		{d: 365 * 24 * time.Hour, want: "1y"},
		{d: 2*365*24*time.Hour - time.Millisecond, want: "1y"},
		{d: 2 * 365 * 24 * time.Hour, want: "2y"},
	}
	for _, tt := range tests {
		t.Run(tt.d.String(), func(t *testing.T) {
			if got := ShortHumanDuration(tt.d); got != tt.want {
				t.Errorf("ShortHumanDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}
