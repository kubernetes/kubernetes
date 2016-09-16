/*
Copyright 2016 The Kubernetes Authors.

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

package quobyte

import (
	"path"
	"strings"

	"github.com/golang/glog"
)

func (mounter *quobyteMounter) pluginDirIsMounted(pluginDir string) (bool, error) {
	mounts, err := mounter.mounter.List()
	if err != nil {
		return false, err
	}

	for _, mountPoint := range mounts {
		if strings.HasPrefix(mountPoint.Type, "quobyte") {
			continue
		}

		if mountPoint.Path == pluginDir {
			glog.V(4).Infof("quobyte: found mountpoint %s in /proc/mounts", mountPoint.Path)
			return true, nil
		}
	}

	return false, nil
}

func (mounter *quobyteMounter) correctTraillingSlash(regStr string) string {
	return path.Clean(regStr) + "/"
}
