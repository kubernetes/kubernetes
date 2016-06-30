// +build !linux

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

package tasks

import (
	"syscall"
)

func sysProcAttr() *syscall.SysProcAttr {
	// TODO(jdef)
	// Consequence of not having Pdeathdig is that on non-Linux systems,
	// if SIGTERM doesn't stop child procs then they may "leak" and be
	// reparented 'up the chain' somewhere when the minion process
	// terminates. For example, such child procs end up living indefinitely
	// as children of the mesos slave process (I think the slave could handle
	// this case, but currently doesn't do it very well). Pdeathsig on Linux
	// was a fallback/failsafe mechanism implemented to guard against this. I
	// don't know if OS X has any syscalls that do something similar.
	return &syscall.SysProcAttr{
		Setpgid: true,
	}
}
