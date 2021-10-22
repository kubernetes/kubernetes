/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cloud

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestPollOperation(t *testing.T) {
	testErr := errors.New("test error")
	tests := []struct {
		name                  string
		op                    *fakeOperation
		cancel                bool
		wantErr               error
		wantRemainingAttempts int
	}{
		{
			name: "Retry",
			op:   &fakeOperation{attemptsRemaining: 10},
		},
		{
			name: "OperationFailed",
			op: &fakeOperation{
				attemptsRemaining: 2,
				err:               testErr,
			},
			wantErr: testErr,
		},
		{
			name: "DoneFailed",
			op: &fakeOperation{
				attemptsRemaining: 2,
				doneErr:           testErr,
			},
			wantErr: testErr,
		},
		{
			name:                  "Cancel",
			op:                    &fakeOperation{attemptsRemaining: 1},
			cancel:                true,
			wantErr:               context.Canceled,
			wantRemainingAttempts: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := Service{RateLimiter: &NopRateLimiter{}}
			ctx, cfn := context.WithTimeout(context.Background(), 3*time.Second)
			defer cfn()
			if test.cancel {
				cfn()
			}
			if gotErr := s.pollOperation(ctx, test.op); gotErr != test.wantErr {
				t.Errorf("pollOperation: got %v, want %v", gotErr, test.wantErr)
			}
			if test.op.attemptsRemaining != test.wantRemainingAttempts {
        t.Errorf("%d attempts remaining, want %d", test.op.attemptsRemaining, test.wantRemainingAttempts)
			}
		})
	}
}

type fakeOperation struct {
	attemptsRemaining int
	doneErr           error
	err               error
}

func (f *fakeOperation) isDone(ctx context.Context) (bool, error) {
	f.attemptsRemaining--
	if f.attemptsRemaining <= 0 {
		return f.doneErr == nil, f.doneErr
	}
	return false, nil
}

func (f *fakeOperation) error() error {
	return f.err
}

func (f *fakeOperation) rateLimitKey() *RateLimitKey {
	return nil
}
