// +build darwin dragonfly freebsd linux netbsd openbsd
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
	"os/signal"
	"syscall"
)

func makeFailoverSigChan() <-chan os.Signal {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGUSR1)
	return ch
}

func makeDisownedProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		Setpgid: true, // disown the spawned scheduler
	}
}
