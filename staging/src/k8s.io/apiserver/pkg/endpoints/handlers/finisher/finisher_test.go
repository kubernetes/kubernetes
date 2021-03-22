/*
Copyright 2021 The Kubernetes Authors.

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

package finisher

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
)

func TestFinishRequest(t *testing.T) {
	exampleObj := &example.Pod{}
	exampleErr := fmt.Errorf("error")
	successStatusObj := &metav1.Status{Status: metav1.StatusSuccess, Message: "success message"}
	errorStatusObj := &metav1.Status{Status: metav1.StatusFailure, Message: "error message"}
	timeoutFunc := func() (context.Context, context.CancelFunc) {
		return context.WithTimeout(context.TODO(), time.Second)
	}

	testcases := []struct {
		name          string
		timeout       func() (context.Context, context.CancelFunc)
		fn            ResultFunc
		expectedObj   runtime.Object
		expectedErr   error
		expectedPanic string

		expectedPanicObj interface{}
	}{
		{
			name:    "Expected obj is returned",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				return exampleObj, nil
			},
			expectedObj: exampleObj,
			expectedErr: nil,
		},
		{
			name:    "Expected error is returned",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				return nil, exampleErr
			},
			expectedObj: nil,
			expectedErr: exampleErr,
		},
		{
			name:    "No expected error or object or panic",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				return nil, nil
			},
		},
		{
			name:    "Successful status object is returned as expected",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				return successStatusObj, nil
			},
			expectedObj: successStatusObj,
			expectedErr: nil,
		},
		{
			name:    "Error status object is converted to StatusError",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				return errorStatusObj, nil
			},
			expectedObj: nil,
			expectedErr: apierrors.FromObject(errorStatusObj),
		},
		{
			name:    "Panic is propagated up",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				panic("my panic")
			},
			expectedObj:   nil,
			expectedErr:   nil,
			expectedPanic: "my panic",
		},
		{
			name:    "Panic is propagated with stack",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				panic("my panic")
			},
			expectedObj:   nil,
			expectedErr:   nil,
			expectedPanic: "finisher_test.go",
		},
		{
			name:    "http.ErrAbortHandler panic is propagated without wrapping with stack",
			timeout: timeoutFunc,
			fn: func() (runtime.Object, error) {
				panic(http.ErrAbortHandler)
			},
			expectedObj:      nil,
			expectedErr:      nil,
			expectedPanic:    http.ErrAbortHandler.Error(),
			expectedPanicObj: http.ErrAbortHandler,
		},
	}
	for i, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := tc.timeout()
			defer func() {
				cancel()

				r := recover()
				switch {
				case r == nil && len(tc.expectedPanic) > 0:
					t.Errorf("expected panic containing '%s', got none", tc.expectedPanic)
				case r != nil && len(tc.expectedPanic) == 0:
					t.Errorf("unexpected panic: %v", r)
				case r != nil && len(tc.expectedPanic) > 0 && !strings.Contains(fmt.Sprintf("%v", r), tc.expectedPanic):
					t.Errorf("expected panic containing '%s', got '%v'", tc.expectedPanic, r)
				}

				if tc.expectedPanicObj != nil && !reflect.DeepEqual(tc.expectedPanicObj, r) {
					t.Errorf("expected panic obj %#v, got %#v", tc.expectedPanicObj, r)
				}
			}()
			obj, err := FinishRequest(ctx, tc.fn)
			if (err == nil && tc.expectedErr != nil) || (err != nil && tc.expectedErr == nil) || (err != nil && tc.expectedErr != nil && err.Error() != tc.expectedErr.Error()) {
				t.Errorf("%d: unexpected err. expected: %v, got: %v", i, tc.expectedErr, err)
			}
			if !apiequality.Semantic.DeepEqual(obj, tc.expectedObj) {
				t.Errorf("%d: unexpected obj. expected %#v, got %#v", i, tc.expectedObj, obj)
			}
		})
	}
}
