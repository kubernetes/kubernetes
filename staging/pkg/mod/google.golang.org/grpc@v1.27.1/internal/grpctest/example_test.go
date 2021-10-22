/*
 *
 * Copyright 2019 gRPC authors.
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

package grpctest_test

import (
	"testing"

	"google.golang.org/grpc/internal/grpctest"
)

type s struct {
	i int
}

func (s *s) Setup(t *testing.T) {
	t.Log("Per-test setup code")
	s.i = 5
}

func (s *s) TestSomething(t *testing.T) {
	t.Log("TestSomething")
	if s.i != 5 {
		t.Errorf("s.i = %v; want 5", s.i)
	}
	s.i = 3
}

func (s *s) TestSomethingElse(t *testing.T) {
	t.Log("TestSomethingElse")
	if got, want := s.i%4, 1; got != want {
		t.Errorf("s.i %% 4 = %v; want %v", got, want)
	}
	s.i = 3
}

func (s *s) Teardown(t *testing.T) {
	t.Log("Per-test teardown code")
	if s.i != 3 {
		t.Fatalf("s.i = %v; want 3", s.i)
	}
}

func TestExample(t *testing.T) {
	grpctest.RunSubTests(t, &s{})
}
