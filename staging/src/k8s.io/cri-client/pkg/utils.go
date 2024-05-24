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

package cri

import (
	"fmt"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
)

// maxMsgSize use 16MB as the default message size limit.
// grpc library default is 4MB
const maxMsgSize = 1024 * 1024 * 16

// verifySandboxStatus verified whether all required fields are set in PodSandboxStatus.
func verifySandboxStatus(status *runtimeapi.PodSandboxStatus) error {
	if status.Id == "" {
		return fmt.Errorf("status.Id is not set")
	}

	if status.Metadata == nil {
		return fmt.Errorf("status.Metadata is not set")
	}

	metadata := status.Metadata
	if metadata.Name == "" || metadata.Namespace == "" || metadata.Uid == "" {
		return fmt.Errorf("metadata.Name, metadata.Namespace or metadata.Uid is not in metadata %q", metadata)
	}

	if status.CreatedAt == 0 {
		return fmt.Errorf("status.CreatedAt is not set")
	}

	return nil
}

// verifyContainerStatus verified whether all required fields are set in ContainerStatus.
func verifyContainerStatus(status *runtimeapi.ContainerStatus) error {
	if status.Id == "" {
		return fmt.Errorf("status.Id is not set")
	}

	if status.Metadata == nil {
		return fmt.Errorf("status.Metadata is not set")
	}

	metadata := status.Metadata
	if metadata.Name == "" {
		return fmt.Errorf("metadata.Name is not in metadata %q", metadata)
	}

	if status.CreatedAt == 0 {
		return fmt.Errorf("status.CreatedAt is not set")
	}

	if status.Image == nil || status.Image.Image == "" {
		return fmt.Errorf("status.Image is not set")
	}

	if status.ImageRef == "" {
		return fmt.Errorf("status.ImageRef is not set")
	}

	return nil
}

func log(logger *klog.Logger, level int, msg string, keyAndValues ...any) {
	if logger == nil {
		return
	}
	logger.V(level).Info(msg, keyAndValues...)
}

func logErr(logger *klog.Logger, err error, msg string, keyAndValues ...any) {
	if logger == nil {
		return
	}
	logger.Error(err, msg, keyAndValues...)
}
