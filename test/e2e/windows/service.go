/*
Copyright 2019 The Kubernetes Authors.

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

package windows

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")

	var cs clientset.Interface

	ginkgo.BeforeEach(func() {
		//Only for Windows containers
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		cs = f.ClientSet
	})
	ginkgo.It("should be able to create a functioning NodePort service for Windows", func() {
		serviceName := "nodeport-test"
		ns := f.Namespace.Name

		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		nodeIP, err := e2enode.PickIP(jig.Client)
		framework.ExpectNoError(err)

		ginkgo.By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)

		nodePort := int(svc.Spec.Ports[0].NodePort)

		ginkgo.By("creating Pod to be part of service " + serviceName)
		_, err = jig.Run(nil)
		framework.ExpectNoError(err)

		//using hybrid_network methods
		ginkgo.By("creating Windows testing Pod")
		windowsPod := createTestPod(f, windowsBusyBoximage, windowsOS)

		ginkgo.By(fmt.Sprintf("checking connectivity Pod to curl http://%s:%d", nodeIP, nodePort))
		assertConsistentConnectivity(f, windowsPod.ObjectMeta.Name, windowsOS, windowsCheck(fmt.Sprintf("http://%s:%d", nodeIP, nodePort)))

	})

})
