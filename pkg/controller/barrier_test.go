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

package controller

import (
	"errors"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func TestTaskBarrier(t *testing.T) {
	noOp := func() error {
		return nil
	}
	noOpJitter := func() error {
		time.Sleep(wait.Jitter(10*time.Millisecond, 0.5))
		return nil
	}
	noOpErr := func() error {
		return errors.New("err")
	}

	tests := []struct {
		prepare  func(*TaskBarrier)
		complete int
		errCount int
	}{
		{
			prepare: func(s *TaskBarrier) {
				s.Go(noOp)
			},
			complete: 1,
		},
		{
			prepare: func(s *TaskBarrier) {
				s.Go(noOp)
				s.Go(noOpJitter)
				s.Go(noOpJitter)
				s.Go(noOpJitter)
			},
			complete: 4,
		},
		{
			prepare: func(s *TaskBarrier) {
				s.Go(noOpErr)
				s.Go(noOp)
				s.Go(noOp)
				s.Go(noOp)
				s.Go(noOp)
			},
			complete: 1,
			errCount: 1,
		},
		{
			prepare: func(s *TaskBarrier) {
				s.Go(noOp)
				s.Go(noOp)
				s.Go(noOp)
				s.Go(noOp)
				s.Go(noOpErr)
			},
			complete: 5,
			errCount: 1,
		},
		{
			prepare: func(s *TaskBarrier) {
				for i := 0; i < 1000; i++ {
					s.Go(noOpJitter)
					s.Go(noOpJitter)
					s.Go(noOpJitter)
				}
				s.Go(noOpErr)
			},
			complete: 3001,
			errCount: 1,
		},
	}
	for _, test := range tests {
		t.Run("", func(t *testing.T) {
			var (
				errs     []error
				complete int
			)
			s := NewTaskBarrier()
			s.Limiter = NewSlowStarter()
			s.Policy = func(err error) bool {
				complete++
				if err != nil {
					errs = append(errs, err)
					return false
				}
				return true
			}

			test.prepare(s)
			s.Wait()

			if s.holds != 0 {
				t.Errorf("unexpected holds on the barrier: %v", s.holds)
			}
			if got, want := len(errs), test.errCount; got != want {
				t.Errorf("unexpected error count: got=%v want=%v", got, want)
			}
			if got, want := complete, test.complete; got != want {
				t.Errorf("unexpected completions: got=%v want=%v", got, want)
			}
		})
	}
}
