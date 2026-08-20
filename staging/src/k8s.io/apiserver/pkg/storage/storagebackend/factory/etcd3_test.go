/*
Copyright 2022 The Kubernetes Authors.

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

package factory

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
)

func Test_atomicLastError(t *testing.T) {
	aError := &atomicLastError{err: fmt.Errorf("initial error")}
	// no timestamp is always updated
	aError.Store(errors.New("updated error"), time.Time{})
	err := aError.Load()
	if err.Error() != "updated error" {
		t.Fatalf("Expected: \"updated error\" got: %s", err.Error())
	}
	// update to current time
	now := time.Now()
	aError.Store(errors.New("now error"), now)
	err = aError.Load()
	if err.Error() != "now error" {
		t.Fatalf("Expected: \"now error\" got: %s", err.Error())
	}
	// no update to past time
	past := now.Add(-5 * time.Second)
	aError.Store(errors.New("past error"), past)
	err = aError.Load()
	if err.Error() != "now error" {
		t.Fatalf("Expected: \"now error\" got: %s", err.Error())
	}
}

func Test_isEtcdMaintenancePermissionDenied(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "etcd >= 3.6 maintenance permission denied",
			err:  grpcstatus.Error(codes.PermissionDenied, "etcdserver: permission denied"),
			want: true,
		},
		{
			name: "other grpc error",
			err:  grpcstatus.Error(codes.Unavailable, "etcdserver: unavailable"),
			want: false,
		},
		{
			name: "plain error",
			err:  errors.New("some error"),
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isEtcdMaintenancePermissionDenied(tt.err); got != tt.want {
				t.Fatalf("isEtcdMaintenancePermissionDenied(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}
