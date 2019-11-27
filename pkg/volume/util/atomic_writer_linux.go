// +build linux

/*
Copyright 2019 The Kubernetes Authors.

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
	"os"
	"syscall"

	"k8s.io/klog"
)

// chown changes the group that owns the given file.
func (w *AtomicWriter) chown(file string) error {
	stat, err := os.Stat(file)
	if err != nil {
		return err
	}

	sysStat, ok := stat.Sys().(*syscall.Stat_t)
	if !ok {
		return nil
	}

	user := int(sysStat.Uid)
	group := int(sysStat.Gid)
	if w.fsGroup != nil {
		group = int(*w.fsGroup)
	}

	err = os.Chown(file, user, group)
	if err != nil {
		// Replicate the behaviour of `volume.SetVolumeOwnership` which ignores any failure to chown a file.
		klog.Errorf("%s: error trying to chown %s to %d:%d: %v", w.logContext, file, user, group, err)
		return nil
	}

	return nil
}
