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
	"golang.org/x/sys/unix"

	"k8s.io/klog/v2"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

var flagMap = map[string]int{
	kubeadmapi.UnmountFlagMNTForce:       unix.MNT_FORCE,
	kubeadmapi.UnmountFlagMNTDetach:      unix.MNT_DETACH,
	kubeadmapi.UnmountFlagMNTExpire:      unix.MNT_EXPIRE,
	kubeadmapi.UnmountFlagUmountNoFollow: unix.UMOUNT_NOFOLLOW,
}

func flagsToInt(flags []string) int {
	res := 0
	for _, f := range flags {
		res |= flagMap[f]
	}
	return res
}

// unmountKubeletDirectory unmounts all paths that contain KubeletRunDirectory
func unmountKubeletDirectory(kubeletRunDirectory string, flags []string) error {
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
	flagsInt := flagsToInt(flags)
	for _, mount := range mounts {
		m := strings.Split(mount, " ")
		if len(m) < 2 || !strings.HasPrefix(m[1], kubeletRunDirectory) {
			continue
		}
		klog.V(5).Infof("[reset] Unmounting %q", m[1])
		if err := syscall.Unmount(m[1], flagsInt); err != nil {
			errList = append(errList, errors.WithMessagef(err, "failed to unmount %q", m[1]))
		}
	}
	return errors.Wrapf(utilerrors.NewAggregate(errList),
		"encountered the following errors while unmounting directories in %q", kubeletRunDirectory)
}
