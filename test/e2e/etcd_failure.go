/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Etcd failure [Disruptive]", func() {

	f := framework.NewDefaultFramework("etcd-failure")

	BeforeEach(func() {
		// This test requires:
		// - SSH
		// - master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		framework.SkipUnlessProviderIs("gce")

		Expect(framework.RunRC(framework.RCConfig{
			Client:    f.Client,
			Name:      "baz",
			Namespace: f.Namespace.Name,
			Image:     "gcr.io/google_containers/pause-amd64:3.0",
			Replicas:  1,
		})).NotTo(HaveOccurred())
	})

	It("should recover from network partition with master", func() {
		etcdFailTest(
			f,
			"sudo iptables -A INPUT -p tcp --destination-port 4001 -j DROP",
			"sudo iptables -D INPUT -p tcp --destination-port 4001 -j DROP",
		)
	})

	It("should recover from SIGKILL", func() {
		etcdFailTest(
			f,
			"pgrep etcd | xargs -I {} sudo kill -9 {}",
			"echo 'do nothing. monit should restart etcd.'",
		)
	})
})

func etcdFailTest(f *framework.Framework, failCommand, fixCommand string) {
	doEtcdFailure(failCommand, fixCommand)

	checkExistingRCRecovers(f)

	ServeImageOrFail(f, "basic", "gcr.io/google_containers/serve_hostname:v1.4")
}

// For this duration, etcd will be failed by executing a failCommand on the master.
// If repeat is true, the failCommand will be called at a rate of once per second for
// the failure duration. If repeat is false, failCommand will only be called once at the
// beginning of the failure duration. After this duration, we execute a fixCommand on the
// master and go on to assert that etcd and kubernetes components recover.
const etcdFailureDuration = 20 * time.Second

func doEtcdFailure(failCommand, fixCommand string) {
	By("failing etcd")

	masterExec(failCommand)
	time.Sleep(etcdFailureDuration)
	masterExec(fixCommand)
}

func masterExec(cmd string) {
	result, err := framework.SSH(cmd, framework.GetMasterHost()+":22", framework.TestContext.Provider)
	Expect(err).NotTo(HaveOccurred())
	if result.Code != 0 {
		framework.LogSSHResult(result)
		framework.Failf("master exec command returned non-zero")
	}
}

func checkExistingRCRecovers(f *framework.Framework) {
	By("assert that the pre-existing replication controller recovers")
	podClient := f.Client.Pods(f.Namespace.Name)
	rcSelector := labels.Set{"name": "baz"}.AsSelector()

	By("deleting pods from existing replication controller")
	framework.ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*60, func() (bool, error) {
		options := api.ListOptions{LabelSelector: rcSelector}
		pods, err := podClient.List(options)
		if err != nil {
			framework.Logf("apiserver returned error, as expected before recovery: %v", err)
			return false, nil
		}
		if len(pods.Items) == 0 {
			return false, nil
		}
		for _, pod := range pods.Items {
			err = podClient.Delete(pod.Name, api.NewDeleteOptions(0))
			Expect(err).NotTo(HaveOccurred())
		}
		framework.Logf("apiserver has recovered")
		return true, nil
	}))

	By("waiting for replication controller to recover")
	framework.ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*60, func() (bool, error) {
		options := api.ListOptions{LabelSelector: rcSelector}
		pods, err := podClient.List(options)
		Expect(err).NotTo(HaveOccurred())
		for _, pod := range pods.Items {
			if pod.DeletionTimestamp == nil && api.IsPodReady(&pod) {
				return true, nil
			}
		}
		return false, nil
	}))
}
