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
//
// By including the interface in
// `pkg/kubelet/legacy/logs.go`, we ensure the interface is
// available to `pkg/kubelet`, even when we are building with the `dockerless`
// tag (i.e. not compiling the dockershim).
// While the interface always exists, there will be no implementations of the
// interface when building with the `dockerless` tag. The lack of
// implementations should not be an issue, as we only expect `pkg/kubelet` code
// to need an implementation of the `DockerLegacyService` when we are using
// docker. If we are using docker, but building with the `dockerless` tag, than
// this will be just one of many things that breaks.
type DockerLegacyService interface {
	// GetContainerLogs gets logs for a specific container.
	GetContainerLogs(context.Context, *v1.Pod, kubecontainer.ContainerID, *v1.PodLogOptions, io.Writer, io.Writer) error

	// IsCRISupportedLogDriver checks whether the logging driver used by docker is
	// supported by native CRI integration.
	// TODO(resouer): remove this when deprecating unsupported log driver
	IsCRISupportedLogDriver() (bool, error)

	kuberuntime.LegacyLogProvider
}
