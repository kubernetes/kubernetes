//go:build linux
// +build linux

/*
Copyright 2025 The Kubernetes Authors.

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

package util

import (
	"errors"

	"golang.org/x/sys/unix"

	"k8s.io/klog/v2"
)

const (
	// Arbitrary limit on max attempts at netlink calls if they are repeatedly interrupted.
	MaxAttemptsEINTR = 5
)

func RetryOnIntr(f func() error) {
	for attempt := 0; attempt < MaxAttemptsEINTR; attempt += 1 {
		if err := f(); !errors.Is(err, unix.EINTR) {
			return
		}
	}
	klog.V(2).InfoS("netlink call interrupted", "attempts", MaxAttemptsEINTR)
}
