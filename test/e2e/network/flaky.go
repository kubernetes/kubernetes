/*
Copyright 2023 The Kubernetes Authors.

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

package network

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("Flaky hostnetwork test", func() {

	fr := framework.NewDefaultFramework("hostnetwork")
	fr.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var (
		ns        string
		podClient *e2epod.PodClient
	)
	ginkgo.BeforeEach(func(ctx context.Context) {
		ns = fr.Namespace.Name
		podClient = e2epod.NewPodClient(fr)
	})

	ginkgo.It("should be able to create multiple hostNetwork pods", func(ctx context.Context) {
		for i := 0; i < 5; i++ {
			name := fmt.Sprintf("pod%d", i)
			ginkgo.By("Creating hostNetwork pod")
			pod := e2epod.NewAgnhostPod(ns, name, nil, nil, nil, "netexec", "--http-port=80")
			pod.Spec.HostNetwork = true
			podClient.CreateSync(ctx, pod)
		}

	})

})
