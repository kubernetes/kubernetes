/*
Copyright 2020 The Kubernetes Authors.

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

package legacy

import (
	"context"
	"io"

	"k8s.io/api/core/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime"
)

// DockerLegacyService interface is used throughout `pkg/kubelet`.
// It used to live in the `pkg/kubelet/dockershim` package. While we
// would eventually like to remove it entirely, we need to give users some form
// of warning.
type DockerLegacyService interface {
	// GetContainerLogs gets logs for a specific container.
	GetContainerLogs(context.Context, *v1.Pod, kubecontainer.ContainerID, *v1.PodLogOptions, io.Writer, io.Writer) error

	// IsCRISupportedLogDriver checks whether the logging driver used by docker is
	// supported by native CRI integration.
	// TODO(resouer): remove this when deprecating unsupported log driver
	IsCRISupportedLogDriver() (bool, error)

	kuberuntime.LegacyLogProvider
}
