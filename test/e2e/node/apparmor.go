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

package node

import (
	"context"

	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2esecurity "k8s.io/kubernetes/test/e2e/framework/security"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("AppArmor", func() {
	f := framework.NewDefaultFramework("apparmor")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("load AppArmor profiles", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			e2eskipper.SkipIfAppArmorNotSupported()
			e2esecurity.LoadAppArmorProfiles(ctx, f.Namespace.Name, f.ClientSet)
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			if !ginkgo.CurrentSpecReport().Failed() {
				return
			}
			e2ekubectl.LogFailedContainers(ctx, f.ClientSet, f.Namespace.Name, framework.Logf)
		})

		ginkgo.It("should enforce an AppArmor profile", func(ctx context.Context) {
			e2esecurity.CreateAppArmorTestPod(ctx, f.Namespace.Name, f.ClientSet, e2epod.NewPodClient(f), false, true)
		})

		ginkgo.It("can disable an AppArmor profile, using unconfined", func(ctx context.Context) {
			e2esecurity.CreateAppArmorTestPod(ctx, f.Namespace.Name, f.ClientSet, e2epod.NewPodClient(f), true, true)
		})
	})
})
