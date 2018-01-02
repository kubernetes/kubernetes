// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pod

import (
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestRetryFailure(t *testing.T) {
	i := 0
	f := func() bool { i++; return false } // never succeed

	if err := retry(newTimeout(100*time.Millisecond), f, 10*time.Millisecond); err == nil {
		t.Error("expected failure, but got none")
		return
	}

	if i == 0 {
		t.Error("expected some retries, but got none")
		return
	}
}

func TestRetrySuccess(t *testing.T) {
	f := func() bool { return true } // always succeed

	if err := retry(newTimeout(5*time.Second), f, 10*time.Millisecond); err != nil {
		t.Errorf("expected success, but got %v", err)
	}

	i := 0
	f = func() bool { i++; return i > 5 } // succeed after 5 times

	if err := retry(newTimeout(5*time.Second), f, 10*time.Millisecond); err != nil {
		t.Errorf("expected success, but got %v", err)
	}
}

func newTimeout(t time.Duration) context.Context {
	ctx, _ := context.WithTimeout(context.Background(), t) // should never be reached
	return ctx
}
