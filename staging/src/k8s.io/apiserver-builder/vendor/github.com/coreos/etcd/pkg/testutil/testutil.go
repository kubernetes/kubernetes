// Copyright 2015 The etcd Authors
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

// Package testutil provides test utility functions.
package testutil

import (
	"net/url"
	"runtime"
	"testing"
	"time"
)

// WaitSchedule briefly sleeps in order to invoke the go scheduler.
// TODO: improve this when we are able to know the schedule or status of target go-routine.
func WaitSchedule() {
	time.Sleep(10 * time.Millisecond)
}

func MustNewURLs(t *testing.T, urls []string) []url.URL {
	if urls == nil {
		return nil
	}
	var us []url.URL
	for _, url := range urls {
		u := MustNewURL(t, url)
		us = append(us, *u)
	}
	return us
}

func MustNewURL(t *testing.T, s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		t.Fatalf("parse %v error: %v", s, err)
	}
	return u
}

// FatalStack helps to fatal the test and print out the stacks of all running goroutines.
func FatalStack(t *testing.T, s string) {
	stackTrace := make([]byte, 8*1024)
	n := runtime.Stack(stackTrace, true)
	t.Error(string(stackTrace[:n]))
	t.Fatalf(s)
}
