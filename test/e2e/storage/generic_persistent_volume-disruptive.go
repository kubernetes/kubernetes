/*
Copyright 2018 The Kubernetes Authors.

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

package storage

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("GenericPersistentVolume[Disruptive]", func() {
	f := framework.NewDefaultFramework("generic-disruptive-pv")
	var (
		c  clientset.Interface
		ns string
	)

	BeforeEach(func() {
		// Skip tests unless number of nodes is 2
		framework.SkipUnlessNodeCountIsAtLeast(2)
		framework.SkipIfProviderIs("local")
		c = f.ClientSet
		ns = f.Namespace.Name
	})
	disruptiveTestTable := []disruptiveTest{
		{
			testItStmt: "Should test that a file written to the mount before kubelet restart is readable after restart.",
			runTest:    utils.TestKubeletRestartsAndRestoresMount,
		},
		{
			testItStmt: "Should test that a volume mounted to a pod that is deleted while the kubelet is down unmounts when the kubelet returns.",
			runTest:    utils.TestVolumeUnmountsFromDeletedPod,
		},
		{
			testItStmt: "Should test that a volume mounted to a pod that is force deleted while the kubelet is down unmounts when the kubelet returns.",
			runTest:    utils.TestVolumeUnmountsFromForceDeletedPod,
		},
	}
	Context("When kubelet restarts", func() {
		// Test table housing the It() title string and test spec.  runTest is type testBody, defined at
		// the start of this file.  To add tests, define a function mirroring the testBody signature and assign
		// to runTest.
		var (
			clientPod *v1.Pod
			pvc       *v1.PersistentVolumeClaim
			pv        *v1.PersistentVolume
		)
		BeforeEach(func() {
			framework.Logf("Initializing pod and pvcs for test")
			clientPod, pvc, pv = createPodPVCFromSC(f, c, ns)
		})
		for _, test := range disruptiveTestTable {
			func(t disruptiveTest) {
				It(t.testItStmt, func() {
					By("Executing Spec")
					t.runTest(c, f, clientPod)
				})
			}(test)
		}
		AfterEach(func() {
			framework.Logf("Tearing down test spec")
			tearDownTestCase(c, f, ns, clientPod, pvc, pv, false)
			pvc, clientPod = nil, nil
		})
	})
})

func createPodPVCFromSC(f *framework.Framework, c clientset.Interface, ns string) (*v1.Pod, *v1.PersistentVolumeClaim, *v1.PersistentVolume) {
	var err error
	test := storageClassTest{
		name:      "default",
		claimSize: "2Gi",
	}
	pvc := newClaim(test, ns, "default")
	pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
	Expect(err).NotTo(HaveOccurred(), "Error creating pvc")
	pvcClaims := []*v1.PersistentVolumeClaim{pvc}
	pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
	Expect(len(pvs)).To(Equal(1))

	By("Creating a pod with dynamically provisioned volume")
	pod, err := framework.CreateNginxPod(c, ns, nil, pvcClaims)
	Expect(err).NotTo(HaveOccurred(), "While creating pods for kubelet restart test")
	return pod, pvc, pvs[0]
}
