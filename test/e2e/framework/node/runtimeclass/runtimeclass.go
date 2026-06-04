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

package runtimeclass

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/ptr"
)

const (
	// PreconfiguredRuntimeClassHandler is the name of the runtime handler
	// that is expected to be preconfigured in the test environment.
	PreconfiguredRuntimeClassHandler = "test-handler"
)

// NewRuntimeClassPod returns a test pod with the given runtimeClassName
func NewRuntimeClassPod(runtimeClassName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("test-runtimeclass-%s-", runtimeClassName),
		},
		Spec: v1.PodSpec{
			RuntimeClassName: &runtimeClassName,
			Containers: []v1.Container{{
				Name:    "test",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"true"},
			}},
			RestartPolicy:                v1.RestartPolicyNever,
			AutomountServiceAccountToken: ptr.To(false),
		},
	}
}

// NodeSupportsPreconfiguredRuntimeClassHandler checks if test-handler is configured by reading the configuration from container runtime config.
// If no error is returned, the container runtime is assumed to support the test-handler, otherwise an error will be returned.
func NodeSupportsPreconfiguredRuntimeClassHandler(ctx context.Context, f *framework.Framework) error {
	node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
	framework.ExpectNoError(err)
	hostExec := storageutils.NewHostExec(f)
	ginkgo.DeferCleanup(hostExec.Cleanup)

	// This is a hacky check that greps the container runtime config to determine if the test-handler is the underlying container runtime config.
	// For containerd, this is configured in kube-up for GCE clusters here: https://github.com/kubernetes/kubernetes/blob/eb729620c522753bc7ae61fc2c7b7ea19d4aad2f/cluster/gce/gci/configure-helper.sh#L3069-L3076
	// For cri-o, see configuration here: https://github.com/cri-o/cri-o/blob/5dc6a035c940f5dde3a727b2e2d8d4b7e371ad55/contrib/test/ci/e2e-base.yml#L2-L13
	// If the `runtimes.test-handler` substring is found in the runtime config, it is assumed that the handler is configured.
	cmd := fmt.Sprintf(`if [ -e '/etc/containerd/config.toml' ]; then
grep -q 'runtimes.%s' /etc/containerd/config.toml
  exit
fi

if [ -e '/etc/crio/crio.conf' ]; then
  grep -q 'runtimes.%s' /etc/crio/crio.conf
  exit
fi

exit 1
`, PreconfiguredRuntimeClassHandler, PreconfiguredRuntimeClassHandler)

	_, err = hostExec.IssueCommandWithResult(ctx, cmd, node)
	return err
}
