// +build windows
// +build !unit_test

/*
Copyright 2015 The Kubernetes Authors.

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

package service

import (
	"os"
	"syscall"
)

func makeFailoverSigChan() <-chan os.Signal {
	/* TODO(jdef)
		from go's windows compatibility test, it looks like we need to provide a filtered
		signal channel here

	        c := make(chan os.Signal, 10)
	        signal.Notify(c)
	        select {
	        case s := <-c:
	                if s != os.Interrupt {
	                        log.Fatalf("Wrong signal received: got %q, want %q\n", s, os.Interrupt)
	                }
	        case <-time.After(3 * time.Second):
	                log.Fatalf("Timeout waiting for Ctrl+Break\n")
	        }
	*/
	return nil
}

func makeDisownedProcAttr() *syscall.SysProcAttr {
	//TODO(jdef) test this somehow?!?!
	return &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP | syscall.CREATE_UNICODE_ENVIRONMENT,
	}
}
