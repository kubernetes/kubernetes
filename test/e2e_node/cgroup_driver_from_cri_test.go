//go:build linux

/*
Copyright 2025 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Cgroup Driver From CRI", feature.CriProxy, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cgroup-driver-from-cri")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
		})

		ginkgo.It("should only report a metric if CRI is outdated", func(ctx context.Context) {
			expectedErr := status.Error(codes.Unimplemented, "unimplemented")
			err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if apiName == criproxy.RuntimeConfig {
					return expectedErr
				}
				return nil
			})
			framework.ExpectNoError(err)
			restartKubelet(context.Background(), true)
			time.Sleep(time.Second * 1)

			m, err := e2emetrics.GrabKubeletMetricsWithoutProxy(context.Background(), nodeNameOrIP()+":10255", "/metrics")
			framework.ExpectNoError(err)
			samples := m[kubeletmetrics.KubeletSubsystem+"_"+kubeletmetrics.CRILosingSupportKey]

			gomega.Expect(samples).NotTo(gomega.BeEmpty())
			gomega.Expect(samples[0].Metric["version"]).To(gomega.BeEquivalentTo("1.36.0"))
		})
		ginkgo.It("should not emit metric if CRI is new enough", func() {
			restartKubelet(context.Background(), true)
			time.Sleep(time.Second * 1)

			m, err := e2emetrics.GrabKubeletMetricsWithoutProxy(context.Background(), nodeNameOrIP()+":10255", "/metrics")
			framework.ExpectNoError(err)
			samples := m[kubeletmetrics.KubeletSubsystem+"_"+kubeletmetrics.CRILosingSupportKey]

			gomega.Expect(samples).To(gomega.BeEmpty())
		})
	})
})
