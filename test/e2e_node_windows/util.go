//go:build windows

/*
Copyright The Kubernetes Authors.

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

package e2enodewindows

import (
	"context"
	"flag"
	"time"

	"go.opentelemetry.io/otel/trace/noop"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	internalapi "k8s.io/cri-api/pkg/apis"
	remote "k8s.io/cri-client/pkg"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	startServices = flag.Bool("start-services", true, "If true, start local node services")
	stopServices  = flag.Bool("stop-services", true, "If true, stop local node services after running tests")
	busyboxImage  = imageutils.GetE2EImage(imageutils.BusyBox)

	// kubeletCfg is the kubelet configuration the test is running against.
	kubeletCfg *kubeletconfig.KubeletConfiguration
)

// getCRIClient connects CRI and returns CRI runtime service clients and image service client.
func getCRIClient(ctx context.Context) (internalapi.RuntimeService, internalapi.ImageManagerService, error) {
	// connection timeout for CRI service connection
	const connectionTimeout = 2 * time.Minute
	runtimeEndpoint := framework.TestContext.ContainerRuntimeEndpoint
	useStreaming := utilfeature.DefaultFeatureGate.Enabled(features.CRIListStreaming)
	r, err := remote.NewRemoteRuntimeService(ctx, runtimeEndpoint, connectionTimeout, noop.NewTracerProvider(), useStreaming)
	if err != nil {
		return nil, nil, err
	}
	imageManagerEndpoint := runtimeEndpoint
	if framework.TestContext.ImageServiceEndpoint != "" {
		// ImageServiceEndpoint is the same as ContainerRuntimeEndpoint if not
		// explicitly specified.
		imageManagerEndpoint = framework.TestContext.ImageServiceEndpoint
	}
	i, err := remote.NewRemoteImageService(ctx, imageManagerEndpoint, connectionTimeout, noop.NewTracerProvider(), useStreaming)
	if err != nil {
		return nil, nil, err
	}
	return r, i, nil
}
