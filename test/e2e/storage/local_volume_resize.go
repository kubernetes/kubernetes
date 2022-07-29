/*
Copyright 2021 The Kubernetes Authors.

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
	"context"
	"fmt"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("PersistentVolumes-expansion ", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-expansion")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.Context("loopback local block volume", func() {
		var (
			config *localTestConfig
			scName string
		)

		testVolType := BlockFsWithFormatLocalVolumeType
		var testVol *localTestVolume
		testMode := immediateMode
		ginkgo.BeforeEach(func() {
			nodes, err := e2enode.GetBoundedReadySchedulableNodes(f.ClientSet, maxNodes)
			framework.ExpectNoError(err)

			scName = fmt.Sprintf("%v-%v", testSCPrefix, f.Namespace.Name)
			// Choose a random node
			randomNode := &nodes.Items[rand.Intn(len(nodes.Items))]

			hostExec := utils.NewHostExec(f)
			ltrMgr := utils.NewLocalResourceManager("local-volume-test", hostExec, hostBase)
			config = &localTestConfig{
				ns:           f.Namespace.Name,
				client:       f.ClientSet,
				timeouts:     f.Timeouts,
				nodes:        nodes.Items,
				randomNode:   randomNode,
				scName:       scName,
				discoveryDir: filepath.Join(hostBase, f.Namespace.Name),
				hostExec:     hostExec,
				ltrMgr:       ltrMgr,
			}

			setupExpandableLocalStorageClass(config, &testMode)
			testVols := setupLocalVolumesPVCsPVs(config, testVolType, config.randomNode, 1, testMode)
			testVol = testVols[0]
		})
		ginkgo.AfterEach(func() {
			cleanupLocalVolumes(config, []*localTestVolume{testVol})
			cleanupStorageClass(config)
		})

		ginkgo.It("should support online expansion on node", func() {
			var (
				pod1    *v1.Pod
				pod1Err error
			)
			ginkgo.By("Creating pod1")
			pod1, pod1Err = createLocalPod(config, testVol, nil)
			framework.ExpectNoError(pod1Err)
			verifyLocalPod(config, testVol, pod1, config.randomNode.Name)

			// We expand the PVC while l.pod is using it for online expansion.
			ginkgo.By("Expanding current pvc")
			currentPvcSize := testVol.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("10Mi"))
			framework.Logf("currentPvcSize %s, newSize %s", currentPvcSize.String(), newSize.String())
			newPVC, err := testsuites.ExpandPVCSize(testVol.pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			testVol.pvc = newPVC
			gomega.Expect(testVol.pvc).NotTo(gomega.BeNil())

			pvcSize := testVol.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc size %q", testVol.pvc.Name)
			}

			// Now update the underlying volume manually
			err = config.ltrMgr.ExpandBlockDevice(testVol.ltr, 10 /*number of 1M blocks to add*/)
			framework.ExpectNoError(err, "while expanding loopback device")

			// now update PV to matching size
			pv, err := UpdatePVSize(testVol.pv, newSize, f.ClientSet)
			framework.ExpectNoError(err, "while updating pv to more size")
			gomega.Expect(pv).NotTo(gomega.BeNil())
			testVol.pv = pv

			ginkgo.By("Waiting for file system resize to finish")
			testVol.pvc, err = testsuites.WaitForFSResize(testVol.pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := testVol.pvc.Status.Conditions
			framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
		})

	})

})

func UpdatePVSize(pv *v1.PersistentVolume, size resource.Quantity, c clientset.Interface) (*v1.PersistentVolume, error) {
	pvName := pv.Name
	pvToUpdate := pv.DeepCopy()

	var lastError error
	waitErr := wait.PollImmediate(5*time.Second, csiResizeWaitPeriod, func() (bool, error) {
		var err error
		pvToUpdate, err = c.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching pv %s: %v", pvName, err)
		}
		pvToUpdate.Spec.Capacity[v1.ResourceStorage] = size
		pvToUpdate, err = c.CoreV1().PersistentVolumes().Update(context.TODO(), pvToUpdate, metav1.UpdateOptions{})
		if err != nil {
			framework.Logf("error updating PV %s: %v", pvName, err)
			lastError = err
			return false, nil
		}
		return true, nil
	})
	if waitErr == wait.ErrWaitTimeout {
		return nil, fmt.Errorf("timed out attempting to update PV size. last update error: %v", lastError)
	}
	if waitErr != nil {
		return nil, fmt.Errorf("failed to expand PV size: %v", waitErr)
	}
	return pvToUpdate, nil
}

func setupExpandableLocalStorageClass(config *localTestConfig, mode *storagev1.VolumeBindingMode) {
	enableExpansion := true
	sc := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.scName,
		},
		Provisioner:          "kubernetes.io/no-provisioner",
		VolumeBindingMode:    mode,
		AllowVolumeExpansion: &enableExpansion,
	}

	_, err := config.client.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{})
	framework.ExpectNoError(err)
}
