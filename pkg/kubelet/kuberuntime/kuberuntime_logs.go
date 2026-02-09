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

package kuberuntime

import (
	"context"
	"io"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/cri-client/pkg/logs"
	"k8s.io/klog/v2"
)

// ReadLogs read the container log and redirect into stdout and stderr.
// Note that containerID is only needed when following the log, or else
// just pass in empty string "".
func (m *kubeGenericRuntimeManager) ReadLogs(ctx context.Context, path, containerID string, apiOpts *v1.PodLogOptions, stdout, stderr io.Writer) error {
	// Convert v1.PodLogOptions into internal log options.
	opts := logs.NewLogOptions(apiOpts, time.Now())
	logger := klog.FromContext(ctx)
	return logs.ReadLogs(ctx, &logger, path, containerID, opts, m.runtimeService, stdout, stderr)
}
