/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package minion

import (
	"syscall"

	log "github.com/golang/glog"
)

func enterPrivateMountNamespace() {
	// enter a new mount NS, useful for isolating changes to the mount table
	// that are made by the kubelet for storage volumes.
	err := syscall.Unshare(syscall.CLONE_NEWNS)
	if err != nil {
		log.Fatalf("failed to enter private mount NS: %v", err)
	}

	// make the rootfs / rslave to the parent mount NS so that we
	// pick up on any changes made there
	err = syscall.Mount("", "/", "dontcare", syscall.MS_REC|syscall.MS_SLAVE, "")
	if err != nil {
		log.Fatalf("failed to mark / rslave: %v", err)
	}
}
