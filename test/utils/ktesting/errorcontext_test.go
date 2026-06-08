/*
Copyright 2024 The Kubernetes Authors.

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

package ktesting

import (
	"errors"
	"testing"

	"github.com/onsi/gomega"
)

func TestWithError(t *testing.T) {
	t.Run("panic", func(t *testing.T) {
		tCtx := Init(t)
		tCtx.Expect(func() {
			tCtx := Init(t)
			var err error
			_, finalize := tCtx.WithError(&err)
			defer finalize()

			panic("pass me through")
		}).To(gomega.Panic())
	})

	normalErr := errors.New("normal error")

	for name, tc := range map[string]struct {
		cb           func(TContext)
		expectNoFail bool
		expectError  string
	}{
		"none": {
			cb:           func(tCtx TContext) {},
			expectNoFail: true,
			expectError:  normalErr.Error(),
		},
		"Error": {
			cb: func(tCtx TContext) {
				tCtx.Error("some error")
			},
			expectError: "some error",
		},
		"Errorf": {
			cb: func(tCtx TContext) {
				tCtx.Errorf("some %s", "error")
			},
			expectError: "some error",
		},
		"Fatal": {
			cb: func(tCtx TContext) {
				tCtx.Fatal("some error")
				tCtx.Error("another error")
			},
			expectError: "some error",
		},
		"Fatalf": {
			cb: func(tCtx TContext) {
				tCtx.Fatalf("some %s", "error")
				tCtx.Error("another error")
			},
			expectError: "some error",
		},
		"Fail": {
			cb: func(tCtx TContext) {
				tCtx.Fatalf("some %s", "error")
				tCtx.Error("another error")
			},
			expectError: "some error",
		},
		"FailNow": {
			cb: func(tCtx TContext) {
				tCtx.FailNow()
				tCtx.Error("another error")
			},
			expectError: errFailedWithNoExplanation.Error(),
		},
		"many": {
			cb: func(tCtx TContext) {
				tCtx.Error("first error")
				tCtx.Error("second error")
			},
			expectError: `first error
second error`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			tCtx := Init(t)
			err := normalErr
			tCtx, finalize := tCtx.WithError(&err)
			func() {
				defer finalize()
				tc.cb(tCtx)
			}()

			if tc.expectNoFail {
				tCtx.Assert(tCtx.Failed()).To(gomega.BeFalseBecause("should have failed"))
			} else {
				tCtx.Assert(tCtx.Failed()).To(gomega.BeTrueBecause("should not have failed"))
			}
			if tc.expectError == "" {
				tCtx.Assert(err).To(gomega.Succeed())
			} else {
				tCtx.Assert(err).To(gomega.MatchError(gomega.Equal(tc.expectError)))
			}
		})
	}
}
