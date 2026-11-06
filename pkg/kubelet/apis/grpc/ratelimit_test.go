/*
Copyright The Kubernetes Authors.

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

package grpc

import (
	"errors"
	"testing"

	"google.golang.org/grpc"
)

type fakeLimiter bool

func (f fakeLimiter) Allow() bool { return bool(f) }

func TestLimiterStreamServerInterceptor(t *testing.T) {
	for _, allow := range []bool{true, false} {
		called := false
		err := LimiterStreamServerInterceptor(fakeLimiter(allow))(nil, nil, &grpc.StreamServerInfo{}, func(interface{}, grpc.ServerStream) error {
			called = true
			return nil
		})
		if allow && (err != nil || !called) {
			t.Errorf("allowed call: err=%v, handler called=%v", err, called)
		}
		if !allow && (!errors.Is(err, ErrorLimitExceeded) || called) {
			t.Errorf("rejected call: err=%v, handler called=%v", err, called)
		}
	}
}
