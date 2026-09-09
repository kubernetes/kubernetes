/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidatePodLogOptions(t *testing.T) {

	var (
		positiveLine             = int64(8)
		negativeLine             = int64(-8)
		limitBytesGreaterThan1   = int64(12)
		limitBytesLessThan1      = int64(0)
		sinceSecondsGreaterThan1 = int64(10)
		sinceSecondsLessThan1    = int64(0)
		timestamp                = metav1.Now()
		stdoutStream             = v1.LogStreamStdout
		stderrStream             = v1.LogStreamStderr
		allStream                = v1.LogStreamAll
		invalidStream            = "invalid"
	)

	successCase := []struct {
		name          string
		podLogOptions v1.PodLogOptions
	}{{
		name:          "Empty PodLogOptions",
		podLogOptions: v1.PodLogOptions{},
	}, {
		name: "PodLogOptions with TailLines",
		podLogOptions: v1.PodLogOptions{
			TailLines: &positiveLine,
		},
	}, {
		name: "PodLogOptions with LimitBytes",
		podLogOptions: v1.PodLogOptions{
			LimitBytes: &limitBytesGreaterThan1,
		},
	}, {
		name: "PodLogOptions with only sinceSeconds",
		podLogOptions: v1.PodLogOptions{
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "PodLogOptions with LimitBytes with TailLines",
		podLogOptions: v1.PodLogOptions{
			LimitBytes: &limitBytesGreaterThan1,
			TailLines:  &positiveLine,
		},
	}, {
		name: "PodLogOptions with LimitBytes with TailLines with SinceSeconds",
		podLogOptions: v1.PodLogOptions{
			LimitBytes:   &limitBytesGreaterThan1,
			TailLines:    &positiveLine,
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "PodLogOptions with stdout Stream",
		podLogOptions: v1.PodLogOptions{
			Stream: &stdoutStream,
		},
	}, {
		name: "PodLogOptions with stderr Stream and Follow",
		podLogOptions: v1.PodLogOptions{
			Stream: &stderrStream,
			Follow: true,
		},
	}, {
		name: "PodLogOptions with All Stream, TailLines and LimitBytes",
		podLogOptions: v1.PodLogOptions{
			Stream:     &allStream,
			TailLines:  &positiveLine,
			LimitBytes: &limitBytesGreaterThan1,
		},
	}}
	for _, tc := range successCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) != 0 {
				t.Errorf("unexpected error: %v", errs)
			}
		})
	}

	errorCase := []struct {
		name          string
		podLogOptions v1.PodLogOptions
	}{{
		name: "Invalid podLogOptions with Negative TailLines",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &negativeLine,
			LimitBytes:   &limitBytesGreaterThan1,
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "Invalid podLogOptions with zero or negative LimitBytes",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &positiveLine,
			LimitBytes:   &limitBytesLessThan1,
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "Invalid podLogOptions with zero or negative SinceSeconds",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &negativeLine,
			LimitBytes:   &limitBytesGreaterThan1,
			SinceSeconds: &sinceSecondsLessThan1,
		},
	}, {
		name: "Invalid podLogOptions with both SinceSeconds and SinceTime set",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &negativeLine,
			LimitBytes:   &limitBytesGreaterThan1,
			SinceSeconds: &sinceSecondsGreaterThan1,
			SinceTime:    &timestamp,
		},
	}, {
		name: "Invalid podLogOptions with invalid Stream",
		podLogOptions: v1.PodLogOptions{
			Stream: &invalidStream,
		},
	}, {
		name: "Invalid podLogOptions with stdout Stream and TailLines set",
		podLogOptions: v1.PodLogOptions{
			Stream:    &stdoutStream,
			TailLines: &positiveLine,
		},
	}, {
		name: "Invalid podLogOptions with stderr Stream and TailLines set",
		podLogOptions: v1.PodLogOptions{
			Stream:    &stderrStream,
			TailLines: &positiveLine,
		},
	}}
	for _, tc := range errorCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) == 0 {
				t.Errorf("expected error")
			}
		})
	}
}
