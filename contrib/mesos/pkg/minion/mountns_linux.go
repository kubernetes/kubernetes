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

package minion

import (
	"syscall"

	log "github.com/golang/glog"
)

// enterPrivateMountNamespace does just that: the current mount ns is unshared (isolated)
// and then made a slave to the root mount / of the parent mount ns (mount events from /
// or its children that happen in the parent NS propagate to us).
//
// this is not yet compatible with volume plugins as implemented by the kubelet, which
// depends on using host-volume args to 'docker run' to attach plugin volumes to CT's
// at runtime. as such, docker needs to be able to see the volumes mounted by k8s plugins,
// which is impossible if k8s volume plugins are running in an isolated mount ns.
//
// an alternative approach would be to always run the kubelet in the host's mount-ns and
// rely upon mesos to forcibly umount bindings in the task sandbox before rmdir'ing it:
// https://issues.apache.org/jira/browse/MESOS-349.
//
// use at your own risk.
func enterPrivateMountNamespace() {
	log.Warningln("EXPERIMENTAL FEATURE: entering private mount ns")

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
