//go:build linux
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

package phases

import (
	"os"
	"strings"
	"syscall"

	"github.com/pkg/errors"

	"k8s.io/klog/v2"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// unmountKubeletDirectory unmounts all paths that contain KubeletRunDirectory
func unmountKubeletDirectory(kubeletRunDirectory string) error {
	raw, err := os.ReadFile("/proc/mounts")
	if err != nil {
		return err
	}

	if !strings.HasSuffix(kubeletRunDirectory, "/") {
		// trailing "/" is needed to ensure that possibly mounted /var/lib/kubelet is skipped
		kubeletRunDirectory += "/"
	}

	var errList []error
	mounts := strings.Split(string(raw), "\n")
	for _, mount := range mounts {
		m := strings.Split(mount, " ")
		if len(m) < 2 || !strings.HasPrefix(m[1], kubeletRunDirectory) {
			continue
		}
		klog.V(5).Infof("[reset] Unmounting %q", m[1])
		if err := syscall.Unmount(m[1], 0); err != nil {
			errList = append(errList, errors.WithMessagef(err, "failed to unmount %q", m[1]))
		}
	}
	return errors.Wrapf(utilerrors.NewAggregate(errList),
		"encountered the following errors while unmounting directories in %q", kubeletRunDirectory)
}
