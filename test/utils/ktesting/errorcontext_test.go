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

	"github.com/stretchr/testify/assert"
)

func TestWithError(t *testing.T) {
	t.Run("panic", func(t *testing.T) {
		assert.Panics(t, func() {
			tCtx := Init(t)
			var err error
			_, finalize := WithError(tCtx, &err)
			defer finalize()

			panic("pass me through")
		})
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
			tCtx, finalize := WithError(tCtx, &err)
			func() {
				defer finalize()
				tc.cb(tCtx)
			}()

			assert.Equal(t, !tc.expectNoFail, tCtx.Failed(), "Failed()")
			if tc.expectError == "" {
				assert.NoError(t, err)
			} else if assert.Error(t, err) {
				assert.Equal(t, tc.expectError, err.Error())
			}
		})
	}
}
