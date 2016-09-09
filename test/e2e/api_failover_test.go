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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Nodes api failover [Slow]", func() {
	f := framework.NewDefaultFramework("nodes-failover")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vagrant")
	})

	// this test relies on kubelet being configured with multiple api servers with 443 and 6444 port
	It("should work if 443 is blocked, because client will try to connect on 6444 port", func() {
		originalPort := 443
		backupPort := 6444
		markName := "111"
		framework.MasterExec(fmt.Sprintf("sudo iptables -t nat -A PREROUTING -p tcp -m tcp --dport %v -j REDIRECT --to-ports %v", originalPort, backupPort))
		framework.MasterExec(fmt.Sprintf("sudo iptables -t mangle -A PREROUTING -p tcp --dport %v -j MARK --set-mark %v", originalPort, markName))
		framework.MasterExec(fmt.Sprintf("sudo iptables -A INPUT -m mark --mark %v -j DROP", markName))
		Eventually(func() error {
			nodes, err := f.Client.Nodes().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			if len(nodes.Items) == 0 {
				return fmt.Errorf("empty node list: %+v", nodes)
			}
			for _, node := range nodes.Items {
				if !api.IsNodeReady(&node) {
					return fmt.Errorf("Node %v is not in ready condition", node.Name)
				}
			}
			return nil
		}, 2*time.Minute, 5*time.Second).Should(BeNil())
		framework.MasterExec(fmt.Sprintf("sudo iptables -t nat -D PREROUTING -p tcp -m tcp --dport %v -j REDIRECT --to-ports %v", originalPort, backupPort))
		framework.MasterExec(fmt.Sprintf("sudo iptables -t mangle -D PREROUTING -p tcp --dport %v -j MARK --set-mark %v", originalPort, markName))
		framework.MasterExec(fmt.Sprintf("sudo iptables -D INPUT -m mark --mark %v -j DROP", markName))
	})
})
