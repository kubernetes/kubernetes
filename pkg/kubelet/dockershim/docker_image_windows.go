//go:build windows && !dockerless
// +build windows,!dockerless

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

package dockershim

import (
	"context"
	"time"

	"k8s.io/klog/v2"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

// ImageFsInfo returns information of the filesystem that is used to store images.
func (ds *dockerService) ImageFsInfo(_ context.Context, _ *runtimeapi.ImageFsInfoRequest) (*runtimeapi.ImageFsInfoResponse, error) {
	statsClient := &winstats.StatsClient{}
	fsinfo, err := statsClient.GetDirFsInfo(ds.dockerRootDir)
	if err != nil {
		klog.ErrorS(err, "Failed to get fsInfo for dockerRootDir", "path", ds.dockerRootDir)
		return nil, err
	}

	filesystems := []*runtimeapi.FilesystemUsage{
		{
			Timestamp: time.Now().UnixNano(),
			UsedBytes: &runtimeapi.UInt64Value{Value: fsinfo.Usage},
			FsId: &runtimeapi.FilesystemIdentifier{
				Mountpoint: ds.dockerRootDir,
			},
		},
	}

	return &runtimeapi.ImageFsInfoResponse{ImageFilesystems: filesystems}, nil
}
