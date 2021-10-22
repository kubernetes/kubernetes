/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package service

import (
	"testing"

	grpc "google.golang.org/grpc"
)

const (
	// The address is irrelevant in this test.
	testAddress = "some_address"
)

func TestDial(t *testing.T) {
	defer func() func() {
		temp := hsDialer
		hsDialer = func(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
			return &grpc.ClientConn{}, nil
		}
		return func() {
			hsDialer = temp
		}
	}()

	// Ensure that hsConn is nil at first.
	hsConn = nil

	// First call to Dial, it should create set hsConn.
	conn1, err := Dial(testAddress)
	if err != nil {
		t.Fatalf("first call to Dial failed: %v", err)
	}
	if conn1 == nil {
		t.Fatal("first call to Dial(_)=(nil, _), want not nil")
	}
	if got, want := hsConn, conn1; got != want {
		t.Fatalf("hsConn=%v, want %v", got, want)
	}

	// Second call to Dial should return conn1 above.
	conn2, err := Dial(testAddress)
	if err != nil {
		t.Fatalf("second call to Dial(_) failed: %v", err)
	}
	if got, want := conn2, conn1; got != want {
		t.Fatalf("second call to Dial(_)=(%v, _), want (%v,. _)", got, want)
	}
	if got, want := hsConn, conn1; got != want {
		t.Fatalf("hsConn=%v, want %v", got, want)
	}
}
