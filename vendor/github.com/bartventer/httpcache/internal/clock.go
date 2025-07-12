// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"net/http"
	"time"
)

type Clock interface {
	Now() time.Time
	Since(t time.Time) time.Duration
}

type clock struct{}

func NewClock() *clock { return &clock{} }

func (c clock) Since(t time.Time) time.Duration { return time.Since(t) }
func (c clock) Now() time.Time                  { return time.Now() }

// FixDateHeader sets the "Date" header to the current time in UTC if it is
// missing or empty, as per RFC 9110 §6.6.1, and reports whether it was changed.
//
// NOTE: This cache forwards all requests to the client, so it MUST set the
// "Date" header to the current time for responses that do not have it set.
func FixDateHeader(h http.Header, receivedAt time.Time) bool {
	if date, valid := RawTime(h.Get("Date")).Value(); !valid || date.IsZero() {
		h.Set("Date", receivedAt.UTC().Format(http.TimeFormat))
		return true
	}
	return false
}
