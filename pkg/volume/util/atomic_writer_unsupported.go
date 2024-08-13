//go:build !linux
// +build !linux

/*
Copyright 2024 The Kubernetes Authors.

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
	"runtime"

	"k8s.io/klog/v2"
)

// chown changes the numeric uid and gid of the named file.
// This is a no-op on unsupported platforms.
func (w *AtomicWriter) chown(name string, uid, _ /* gid */ int) error {
	klog.Warningf("%s: skipping change of Linux owner %v for file %s; unsupported on %s", w.logContext, uid, name, runtime.GOOS)
	return nil
}
