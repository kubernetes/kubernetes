/*
Copyright 2019 The Kubernetes Authors.

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

package helpers

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
)

func Test_RoundUpToGiB(t *testing.T) {
	testcases := []struct {
		name        string
		resource    resource.Quantity
		roundedVal  int64
		expectError bool
	}{
		{
			name:       "round Ki to GiB",
			resource:   resource.MustParse("1000Ki"),
			roundedVal: int64(1),
		},
		{
			name:       "round k to GiB",
			resource:   resource.MustParse("1000k"),
			roundedVal: int64(1),
		},
		{
			name:       "round Mi to GiB",
			resource:   resource.MustParse("1000Mi"),
			roundedVal: int64(1),
		},
		{
			name:       "round M to GiB",
			resource:   resource.MustParse("1000M"),
			roundedVal: int64(1),
		},
		{
			name:       "round G to GiB",
			resource:   resource.MustParse("1000G"),
			roundedVal: int64(932),
		},
		{
			name:       "round Gi to GiB",
			resource:   resource.MustParse("1000Gi"),
			roundedVal: int64(1000),
		},
		{
			name:        "round overflowed quantity to int64",
			resource:    resource.MustParse("73786976299133170k"),
			roundedVal:  int64(0),
			expectError: true,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			val, err := RoundUpToGiB(test.resource)
			if !test.expectError && err != nil {
				t.Errorf("expected no error got: %v", err)
			}

			if test.expectError && err == nil {
				t.Errorf("expected error but got nothing")
			}

			if val != test.roundedVal {
				t.Logf("actual rounded value: %d", val)
				t.Logf("expected rounded value: %d", test.roundedVal)
				t.Error("unexpected rounded value")
			}
		})
	}
}

func Test_RoundUpToMB(t *testing.T) {
	testcases := []struct {
		name        string
		resource    resource.Quantity
		roundedVal  int64
		expectError bool
	}{
		{
			name:       "round Ki to MB",
			resource:   resource.MustParse("1000Ki"),
			roundedVal: int64(2),
		},
		{
			name:       "round k to MB",
			resource:   resource.MustParse("1000k"),
			roundedVal: int64(1),
		},
		{
			name:       "round Mi to MB",
			resource:   resource.MustParse("1000Mi"),
			roundedVal: int64(1049),
		},
		{
			name:       "round M to MB",
			resource:   resource.MustParse("1000M"),
			roundedVal: int64(1000),
		},
		{
			name:       "round G to MB",
			resource:   resource.MustParse("1000G"),
			roundedVal: int64(1000000),
		},
		{
			name:       "round Gi to MB",
			resource:   resource.MustParse("1000Gi"),
			roundedVal: int64(1073742),
		},
		{
			name:        "round overflowed quantity to int64",
			resource:    resource.MustParse("73786976299133170k"),
			roundedVal:  int64(0),
			expectError: true,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			val, err := RoundUpToMB(test.resource)
			if !test.expectError && err != nil {
				t.Errorf("expected no error got: %v", err)
			}

			if test.expectError && err == nil {
				t.Errorf("expected error but got nothing")
			}

			if val != test.roundedVal {
				t.Logf("actual rounded value: %d", val)
				t.Logf("expected rounded value: %d", test.roundedVal)
				t.Error("unexpected rounded value")
			}
		})
	}
}

func Test_RoundUpToMiB(t *testing.T) {
	testcases := []struct {
		name        string
		resource    resource.Quantity
		roundedVal  int64
		expectError bool
	}{
		{
			name:       "round Ki to MiB",
			resource:   resource.MustParse("1000Ki"),
			roundedVal: int64(1),
		},
		{
			name:       "round k to MiB",
			resource:   resource.MustParse("1000k"),
			roundedVal: int64(1),
		},
		{
			name:       "round Mi to MiB",
			resource:   resource.MustParse("1000Mi"),
			roundedVal: int64(1000),
		},
		{
			name:       "round M to MiB",
			resource:   resource.MustParse("1000M"),
			roundedVal: int64(954),
		},
		{
			name:       "round G to MiB",
			resource:   resource.MustParse("1000G"),
			roundedVal: int64(953675),
		},
		{
			name:       "round Gi to MiB",
			resource:   resource.MustParse("1000Gi"),
			roundedVal: int64(1024000),
		},
		{
			name:        "round overflowed quantity to int64",
			resource:    resource.MustParse("73786976299133170k"),
			roundedVal:  int64(0),
			expectError: true,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			val, err := RoundUpToMiB(test.resource)
			if !test.expectError && err != nil {
				t.Errorf("expected no error got: %v", err)
			}

			if test.expectError && err == nil {
				t.Errorf("expected error but got nothing")
			}

			if val != test.roundedVal {
				t.Logf("actual rounded value: %d", val)
				t.Logf("expected rounded value: %d", test.roundedVal)
				t.Error("unexpected rounded value")
			}
		})
	}
}

func Test_RoundUpToKB(t *testing.T) {
	testcases := []struct {
		name        string
		resource    resource.Quantity
		roundedVal  int64
		expectError bool
	}{
		{
			name:       "round Ki to KB",
			resource:   resource.MustParse("1000Ki"),
			roundedVal: int64(1024),
		},
		{
			name:       "round k to KB",
			resource:   resource.MustParse("1000k"),
			roundedVal: int64(1000),
		},
		{
			name:       "round Mi to KB",
			resource:   resource.MustParse("1000Mi"),
			roundedVal: int64(1048576),
		},
		{
			name:       "round M to KB",
			resource:   resource.MustParse("1000M"),
			roundedVal: int64(1000000),
		},
		{
			name:       "round G to KB",
			resource:   resource.MustParse("1000G"),
			roundedVal: int64(1000000000),
		},
		{
			name:       "round Gi to KB",
			resource:   resource.MustParse("1000Gi"),
			roundedVal: int64(1073741824),
		},
		{
			name:        "round overflowed quantity to int64",
			resource:    resource.MustParse("73786976299133170k"),
			roundedVal:  int64(0),
			expectError: true,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			val, err := RoundUpToKB(test.resource)
			if !test.expectError && err != nil {
				t.Errorf("expected no error got: %v", err)
			}

			if test.expectError && err == nil {
				t.Errorf("expected error but got nothing")
			}

			if val != test.roundedVal {
				t.Logf("actual rounded value: %d", val)
				t.Logf("expected rounded value: %d", test.roundedVal)
				t.Error("unexpected rounded value")
			}
		})
	}
}

func Test_RoundUpToKiB(t *testing.T) {
	testcases := []struct {
		name        string
		resource    resource.Quantity
		roundedVal  int64
		expectError bool
	}{
		{
			name:       "round Ki to KiB",
			resource:   resource.MustParse("1000Ki"),
			roundedVal: int64(1000),
		},
		{
			name:       "round k to KiB",
			resource:   resource.MustParse("1000k"),
			roundedVal: int64(977),
		},
		{
			name:       "round Mi to KiB",
			resource:   resource.MustParse("1000Mi"),
			roundedVal: int64(1024000),
		},
		{
			name:       "round M to KiB",
			resource:   resource.MustParse("1000M"),
			roundedVal: int64(976563),
		},
		{
			name:       "round G to KiB",
			resource:   resource.MustParse("1000G"),
			roundedVal: int64(976562500),
		},
		{
			name:       "round Gi to KiB",
			resource:   resource.MustParse("1000Gi"),
			roundedVal: int64(1048576000),
		},
		{
			name:        "round overflowed quantity to int64",
			resource:    resource.MustParse("73786976299133170k"),
			roundedVal:  int64(0),
			expectError: true,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			val, err := RoundUpToKiB(test.resource)
			if !test.expectError && err != nil {
				t.Errorf("expected no error got: %v", err)
			}

			if test.expectError && err == nil {
				t.Errorf("expected error but got nothing")
			}

			if val != test.roundedVal {
				t.Logf("actual rounded value: %d", val)
				t.Logf("expected rounded value: %d", test.roundedVal)
				t.Error("unexpected rounded value")
			}
		})
	}
}

func Test_RoundUpToGiBInt32(t *testing.T) {
	testcases := []struct {
		name        string
		resource    resource.Quantity
		roundedVal  int32
		expectError bool
	}{
		{
			name:       "round Ki to GiB",
			resource:   resource.MustParse("1000Ki"),
			roundedVal: int32(1),
		},
		{
			name:       "round k to GiB",
			resource:   resource.MustParse("1000k"),
			roundedVal: int32(1),
		},
		{
			name:       "round Mi to GiB",
			resource:   resource.MustParse("1000Mi"),
			roundedVal: int32(1),
		},
		{
			name:       "round M to GiB",
			resource:   resource.MustParse("1000M"),
			roundedVal: int32(1),
		},
		{
			name:       "round G to GiB",
			resource:   resource.MustParse("1000G"),
			roundedVal: int32(932),
		},
		{
			name:       "round Gi to GiB",
			resource:   resource.MustParse("1000Gi"),
			roundedVal: int32(1000),
		},
		{
			name:        "round overflowed quantity to int32",
			resource:    resource.MustParse("73786976299133170k"),
			roundedVal:  int32(0),
			expectError: true,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			val, err := RoundUpToGiBInt32(test.resource)
			if !test.expectError && err != nil {
				t.Errorf("expected no error got: %v", err)
			}

			if test.expectError && err == nil {
				t.Errorf("expected error but got nothing")
			}

			if val != test.roundedVal {
				t.Logf("actual rounded value: %d", val)
				t.Logf("expected rounded value: %d", test.roundedVal)
				t.Error("unexpected rounded value")
			}
		})
	}
}
