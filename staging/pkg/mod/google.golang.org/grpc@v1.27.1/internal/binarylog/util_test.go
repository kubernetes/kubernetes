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

import "testing"

func TestParseMethodName(t *testing.T) {
	testCases := []struct {
		methodName      string
		service, method string
	}{
		{methodName: "/s/m", service: "s", method: "m"},
		{methodName: "/p.s/m", service: "p.s", method: "m"},
		{methodName: "/p/s/m", service: "p/s", method: "m"},
	}
	for _, tc := range testCases {
		s, m, err := parseMethodName(tc.methodName)
		if err != nil {
			t.Errorf("Parsing %q got error %v, want nil", tc.methodName, err)
			continue
		}
		if s != tc.service || m != tc.method {
			t.Errorf("Parseing %q got service %q, method %q, want service %q, method %q",
				tc.methodName, s, m, tc.service, tc.method,
			)
		}
	}
}

func TestParseMethodNameInvalid(t *testing.T) {
	testCases := []string{
		"/",
		"/sm",
		"",
		"sm",
	}
	for _, tc := range testCases {
		_, _, err := parseMethodName(tc)
		if err == nil {
			t.Errorf("Parsing %q got nil error, want non-nil error", tc)
		}
	}
}
