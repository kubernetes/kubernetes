// Go support for leveled logs, analogous to https://code.google.com/p/google-glog/
//
// Copyright 2013 Google Inc. All Rights Reserved.
// Copyright 2022 The Kubernetes Authors.
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

package klog

import (
	"fmt"
	"os"
	"time"
)

var (

	// ExitFlushTimeout is the timeout that klog has traditionally used during
	// calls like Fatal or Exit when flushing log data right before exiting.
	// Applications that replace those calls and do not have some specific
	// requirements like "exit immediately" can use this value as parameter
	// for FlushAndExit.
	//
	// Can be set for testing purpose or to change the application's
	// default.
	ExitFlushTimeout = 10 * time.Second

	// OsExit is the function called by FlushAndExit to terminate the program.
	//
	// Can be set for testing purpose or to change the application's
	// default behavior. Note that the function should not simply return
	// because callers of functions like Fatal will not expect that.
	OsExit = os.Exit
)

// FlushAndExit flushes log data for a certain amount of time and then calls
// os.Exit. Combined with some logging call it provides a replacement for
// traditional calls like Fatal or Exit.
func FlushAndExit(flushTimeout time.Duration, exitCode int) {
	timeoutFlush(flushTimeout)
	OsExit(exitCode)
}

// timeoutFlush calls Flush and returns when it completes or after timeout
// elapses, whichever happens first.  This is needed because the hooks invoked
// by Flush may deadlock when klog.Fatal is called from a hook that holds
// a lock. Flushing also might take too long.
func timeoutFlush(timeout time.Duration) {
	done := make(chan bool, 1)
	go func() {
		Flush() // calls logging.lockAndFlushAll()
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(timeout):
		fmt.Fprintln(os.Stderr, "klog: Flush took longer than", timeout)
	}
}
