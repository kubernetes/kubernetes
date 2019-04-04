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

package remote

import (
	"context"
	"fmt"
	"time"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

// maxMsgSize use 16MB as the default message size limit.
// grpc library default is 4MB
const maxMsgSize = 1024 * 1024 * 16

// getContextWithTimeout returns a context with timeout.
func getContextWithTimeout(timeout time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), timeout)
}

// getContextWithCancel returns a context with cancel.
func getContextWithCancel() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

// verifySandboxStatus verified whether all required fields are set in PodSandboxStatus.
func verifySandboxStatus(status *runtimeapi.PodSandboxStatus) error {
	if status.Id == "" {
		return fmt.Errorf("Id is not set")
	}

	if status.Metadata == nil {
		return fmt.Errorf("Metadata is not set")
	}

	metadata := status.Metadata
	if metadata.Name == "" || metadata.Namespace == "" || metadata.Uid == "" {
		return fmt.Errorf("Name, Namespace or Uid is not in metadata %q", metadata)
	}

	if status.CreatedAt == 0 {
		return fmt.Errorf("CreatedAt is not set")
	}

	return nil
}

// verifyContainerStatus verified whether all required fields are set in ContainerStatus.
func verifyContainerStatus(status *runtimeapi.ContainerStatus) error {
	if status.Id == "" {
		return fmt.Errorf("Id is not set")
	}

	if status.Metadata == nil {
		return fmt.Errorf("Metadata is not set")
	}

	metadata := status.Metadata
	if metadata.Name == "" {
		return fmt.Errorf("Name is not in metadata %q", metadata)
	}

	if status.CreatedAt == 0 {
		return fmt.Errorf("CreatedAt is not set")
	}

	if status.Image == nil || status.Image.Image == "" {
		return fmt.Errorf("Image is not set")
	}

	if status.ImageRef == "" {
		return fmt.Errorf("ImageRef is not set")
	}

	return nil
}
