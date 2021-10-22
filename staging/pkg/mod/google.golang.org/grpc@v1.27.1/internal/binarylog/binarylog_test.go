/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package binarylog

import (
	"testing"
)

// Test that get method logger returns the one with the most exact match.
func TestGetMethodLogger(t *testing.T) {
	testCases := []struct {
		in       string
		method   string
		hdr, msg uint64
	}{
		// Global.
		{
			in:     "*{h:12;m:23}",
			method: "/s/m",
			hdr:    12, msg: 23,
		},
		// service/*.
		{
			in:     "*,s/*{h:12;m:23}",
			method: "/s/m",
			hdr:    12, msg: 23,
		},
		// Service/method.
		{
			in:     "*{h;m},s/m{h:12;m:23}",
			method: "/s/m",
			hdr:    12, msg: 23,
		},
		{
			in:     "*{h;m},s/*{h:314;m},s/m{h:12;m:23}",
			method: "/s/m",
			hdr:    12, msg: 23,
		},
		{
			in:     "*{h;m},s/*{h:12;m:23},s/m",
			method: "/s/m",
			hdr:    maxUInt, msg: maxUInt,
		},

		// service/*.
		{
			in:     "*{h;m},s/*{h:12;m:23},s/m1",
			method: "/s/m",
			hdr:    12, msg: 23,
		},
		{
			in:     "*{h;m},s1/*,s/m{h:12;m:23}",
			method: "/s/m",
			hdr:    12, msg: 23,
		},

		// With black list.
		{
			in:     "*{h:12;m:23},-s/m1",
			method: "/s/m",
			hdr:    12, msg: 23,
		},
	}
	for _, tc := range testCases {
		l := NewLoggerFromConfigString(tc.in)
		if l == nil {
			t.Errorf("in: %q, failed to create logger from config string", tc.in)
			continue
		}
		ml := l.getMethodLogger(tc.method)
		if ml == nil {
			t.Errorf("in: %q, method logger is nil, want non-nil", tc.in)
			continue
		}

		if ml.headerMaxLen != tc.hdr || ml.messageMaxLen != tc.msg {
			t.Errorf("in: %q, want header: %v, message: %v, got header: %v, message: %v", tc.in, tc.hdr, tc.msg, ml.headerMaxLen, ml.messageMaxLen)
		}
	}
}

// expect method logger to be nil
func TestGetMethodLoggerOff(t *testing.T) {
	testCases := []struct {
		in     string
		method string
	}{
		// method not specified.
		{
			in:     "s1/m",
			method: "/s/m",
		},
		{
			in:     "s/m1",
			method: "/s/m",
		},
		{
			in:     "s1/*",
			method: "/s/m",
		},
		{
			in:     "s1/*,s/m1",
			method: "/s/m",
		},

		// blacklisted.
		{
			in:     "*,-s/m",
			method: "/s/m",
		},
		{
			in:     "s/*,-s/m",
			method: "/s/m",
		},
		{
			in:     "-s/m,s/*",
			method: "/s/m",
		},
	}
	for _, tc := range testCases {
		l := NewLoggerFromConfigString(tc.in)
		if l == nil {
			t.Errorf("in: %q, failed to create logger from config string", tc.in)
			continue
		}
		ml := l.getMethodLogger(tc.method)
		if ml != nil {
			t.Errorf("in: %q, method logger is non-nil, want nil", tc.in)
		}
	}
}
