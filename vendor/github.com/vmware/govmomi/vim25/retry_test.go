/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package vim25

import (
	"context"
	"testing"

	"github.com/vmware/govmomi/vim25/soap"
)

type tempError struct{}

func (tempError) Error() string   { return "tempError" }
func (tempError) Timeout() bool   { return true }
func (tempError) Temporary() bool { return true }

type nonTempError struct{}

func (nonTempError) Error() string   { return "nonTempError" }
func (nonTempError) Timeout() bool   { return false }
func (nonTempError) Temporary() bool { return false }

type fakeRoundTripper struct {
	errs []error
}

func (f *fakeRoundTripper) RoundTrip(ctx context.Context, req, res soap.HasFault) error {
	err := f.errs[0]
	f.errs = f.errs[1:]
	return err
}

func TestRetry(t *testing.T) {
	var tcs = []struct {
		errs     []error
		expected error
	}{
		{
			errs:     []error{nil},
			expected: nil,
		},
		{
			errs:     []error{tempError{}, nil},
			expected: nil,
		},
		{
			errs:     []error{tempError{}, tempError{}},
			expected: tempError{},
		},
		{
			errs:     []error{nonTempError{}},
			expected: nonTempError{},
		},
		{
			errs:     []error{tempError{}, nonTempError{}},
			expected: nonTempError{},
		},
	}

	for _, tc := range tcs {
		var rt soap.RoundTripper

		rt = &fakeRoundTripper{errs: tc.errs}
		rt = Retry(rt, TemporaryNetworkError(2))

		err := rt.RoundTrip(nil, nil, nil)
		if err != tc.expected {
			t.Errorf("Expected: %s, got: %s", tc.expected, err)
		}
	}
}
