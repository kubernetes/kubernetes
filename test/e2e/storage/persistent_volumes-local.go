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
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type localTestConfig struct {
	ns           string
	nodes        []v1.Node
	node0        *v1.Node
	client       clientset.Interface
	scName       string
	ssTester     *framework.StatefulSetTester
	discoveryDir string
	hostExec     utils.HostExec
	ltrMgr       utils.LocalTestResourceManager
}

type localVolumeType string

const (
	// default local volume type, aka a directory
	DirectoryLocalVolumeType localVolumeType = "dir"
	// like DirectoryLocalVolumeType but it's a symbolic link to directory
	DirectoryLinkLocalVolumeType localVolumeType = "dir-link"
	// like DirectoryLocalVolumeType but bind mounted
	DirectoryBindMountedLocalVolumeType localVolumeType = "dir-bindmounted"
	// like DirectoryLocalVolumeType but it's a symbolic link to self bind mounted directory
	// Note that bind mounting at symbolic link actually mounts at directory it
	// links to.
	DirectoryLinkBindMountedLocalVolumeType localVolumeType = "dir-link-bindmounted"
	// creates a tmpfs and mounts it
	TmpfsLocalVolumeType localVolumeType = "tmpfs"
	// tests based on local ssd at /mnt/disks/by-uuid/
	GCELocalSSDVolumeType localVolumeType = "gce-localssd-scsi-fs"
	// Creates a local file, formats it, and maps it as a block device.
	BlockLocalVolumeType localVolumeType = "block"
	// Creates a local file serving as the backing for block device., formats it,
	// and mounts it to use as FS mode local volume.
	BlockFsWithFormatLocalVolumeType localVolumeType = "blockfswithformat"
	// Creates a local file serving as the backing for block device. do not format it manually,
	// and mounts it to use as FS mode local volume.
	BlockFsWithoutFormatLocalVolumeType localVolumeType = "blockfswithoutformat"
)

// map to local test resource type
var setupLocalVolumeMap = map[localVolumeType]utils.LocalVolumeType{
	GCELocalSSDVolumeType:                   utils.LocalVolumeGCELocalSSD,
	TmpfsLocalVolumeType:                    utils.LocalVolumeTmpfs,
	DirectoryLocalVolumeType:                utils.LocalVolumeDirectory,
	DirectoryLinkLocalVolumeType:            utils.LocalVolumeDirectoryLink,
	DirectoryBindMountedLocalVolumeType:     utils.LocalVolumeDirectoryBindMounted,
	DirectoryLinkBindMountedLocalVolumeType: utils.LocalVolumeDirectoryLinkBindMounted,
	BlockLocalVolumeType:                    utils.LocalVolumeBlock, // block device in Block mode
	BlockFsWithFormatLocalVolumeType:        utils.LocalVolumeBlockFS,
	BlockFsWithoutFormatLocalVolumeType:     utils.LocalVolumeBlock, // block device in Filesystem mode (default in this test suite)
}

type localTestVolume struct {
	// Local test resource
	ltr *utils.LocalTestResource
	// PVC for this volume
	pvc *v1.PersistentVolumeClaim
	// PV for this volume
	pv *v1.PersistentVolume
	// Type of local volume
	localVolumeType localVolumeType
}

const (
	// TODO: This may not be available/writable on all images.
	hostBase = "/tmp"
	// Path to the first volume in the test containers
	// created via createLocalPod or makeLocalPod
	// leveraging pv_util.MakePod
	volumeDir = "/mnt/volume1"
	// testFile created in setupLocalVolume
	testFile = "test-file"
	// testFileContent written into testFile
	testFileContent = "test-file-content"
	testSCPrefix    = "local-volume-test-storageclass"

	// A sample request size
	testRequestSize = "10Mi"

	// Max number of nodes to use for testing
	maxNodes = 5
)

var (
	// storage class volume binding modes
	waitMode      = storagev1.VolumeBindingWaitForFirstConsumer
	immediateMode = storagev1.VolumeBindingImmediate

	// Common selinux labels
	selinuxLabel = &v1.SELinuxOptions{
		Level: "s0:c0,c1"}
)

var _ = utils.SIGDescribe("PersistentVolumes-local ", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-test")

	var (
		config *localTestConfig
		scName string
	)

	BeforeEach(func() {
		// Get all the schedulable nodes
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).NotTo(BeZero(), "No available nodes for scheduling")

		// Cap max number of nodes
		maxLen := len(nodes.Items)
		if maxLen > maxNodes {
			maxLen = maxNodes
		}

		scName = fmt.Sprintf("%v-%v", testSCPrefix, f.Namespace.Name)
		// Choose the first node
		node0 := &nodes.Items[0]

		ssTester := framework.NewStatefulSetTester(f.ClientSet)
		hostExec := utils.NewHostExec(f)
		ltrMgr := utils.NewLocalResourceManager("local-volume-test", hostExec, hostBase)
		config = &localTestConfig{
			ns:           f.Namespace.Name,
			client:       f.ClientSet,
			nodes:        nodes.Items[:maxLen],
			node0:        node0,
			scName:       scName,
			ssTester:     ssTester,
			discoveryDir: filepath.Join(hostBase, f.Namespace.Name),
			hostExec:     hostExec,
			ltrMgr:       ltrMgr,
		}
	})

	for tempTestVolType := range setupLocalVolumeMap {

		// New variable required for gingko test closures
		testVolType := tempTestVolType
		serialStr := ""
		if testVolType == GCELocalSSDVolumeType {
			serialStr = " [Serial]"
		}
		ctxString := fmt.Sprintf("[Volume type: %s]%v", testVolType, serialStr)
		testMode := immediateMode

		Context(ctxString, func() {
			var testVol *localTestVolume

			BeforeEach(func() {
				if testVolType == GCELocalSSDVolumeType {
					SkipUnlessLocalSSDExists(config, "scsi", "fs", config.node0)
				}
				setupStorageClass(config, &testMode)
				testVols := setupLocalVolumesPVCsPVs(config, testVolType, config.node0, 1, testMode)
				testVol = testVols[0]
			})

			AfterEach(func() {
				cleanupLocalVolumes(config, []*localTestVolume{testVol})
				cleanupStorageClass(config)
			})

			Context("One pod requesting one prebound PVC", func() {
				var (
					pod1    *v1.Pod
					pod1Err error
				)

				BeforeEach(func() {
					By("Creating pod1")
					pod1, pod1Err = createLocalPod(config, testVol, nil)
					Expect(pod1Err).NotTo(HaveOccurred())
					verifyLocalPod(config, testVol, pod1, config.node0.Name)

					writeCmd := createWriteCmd(volumeDir, testFile, testFileContent, testVol.localVolumeType)

					By("Writing in pod1")
					podRWCmdExec(pod1, writeCmd)
				})

				AfterEach(func() {
					By("Deleting pod1")
					framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
				})

				It("should be able to mount volume and read from pod1", func() {
					By("Reading in pod1")
					// testFileContent was written in BeforeEach
					testReadFileContent(volumeDir, testFile, testFileContent, pod1, testVolType)
				})

				It("should be able to mount volume and write from pod1", func() {
					// testFileContent was written in BeforeEach
					testReadFileContent(volumeDir, testFile, testFileContent, pod1, testVolType)

					By("Writing in pod1")
					writeCmd := createWriteCmd(volumeDir, testFile, testVol.ltr.Path /*writeTestFileContent*/, testVolType)
					podRWCmdExec(pod1, writeCmd)
				})
			})

			Context("Two pods mounting a local volume at the same time", func() {
				It("should be able to write from pod1 and read from pod2", func() {
					twoPodsReadWriteTest(config, testVol)
				})
			})

			Context("Two pods mounting a local volume one after the other", func() {
				It("should be able to write from pod1 and read from pod2", func() {
					twoPodsReadWriteSerialTest(config, testVol)
				})
			})

			Context("Set fsGroup for local volume", func() {
				BeforeEach(func() {
					if testVolType == BlockLocalVolumeType {
						framework.Skipf("We don't set fsGroup on block device, skipped.")
					}
				})

				It("should set fsGroup for one pod", func() {
					By("Checking fsGroup is set")
					pod := createPodWithFsGroupTest(config, testVol, 1234, 1234)
					By("Deleting pod")
					framework.DeletePodOrFail(config.client, config.ns, pod.Name)
				})

				It("should set same fsGroup for two pods simultaneously", func() {
					fsGroup := int64(1234)
					By("Create first pod and check fsGroup is set")
					pod1 := createPodWithFsGroupTest(config, testVol, fsGroup, fsGroup)
					By("Create second pod with same fsGroup and check fsGroup is correct")
					pod2 := createPodWithFsGroupTest(config, testVol, fsGroup, fsGroup)
					By("Deleting first pod")
					framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
					By("Deleting second pod")
					framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
				})

				It("should set different fsGroup for second pod if first pod is deleted", func() {
					framework.Skipf("Disabled temporarily, reopen after #73168 is fixed")
					fsGroup1, fsGroup2 := int64(1234), int64(4321)
					By("Create first pod and check fsGroup is set")
					pod1 := createPodWithFsGroupTest(config, testVol, fsGroup1, fsGroup1)
					By("Deleting first pod")
					err := framework.DeletePodWithWait(f, config.client, pod1)
					Expect(err).NotTo(HaveOccurred(), "while deleting first pod")
					By("Create second pod and check fsGroup is the new one")
					pod2 := createPodWithFsGroupTest(config, testVol, fsGroup2, fsGroup2)
					By("Deleting second pod")
					framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
				})
			})

		})
	}

	Context("Local volume that cannot be mounted [Slow]", func() {
		// TODO:
		// - check for these errors in unit tests instead
		It("should fail due to non-existent path", func() {
			testVol := &localTestVolume{
				ltr: &utils.LocalTestResource{
					Node: config.node0,
					Path: "/non-existent/location/nowhere",
				},
				localVolumeType: DirectoryLocalVolumeType,
			}
			By("Creating local PVC and PV")
			createLocalPVCsPVs(config, []*localTestVolume{testVol}, immediateMode)
			pod, err := createLocalPod(config, testVol, nil)
			Expect(err).To(HaveOccurred())
			err = framework.WaitTimeoutForPodRunningInNamespace(config.client, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
			Expect(err).To(HaveOccurred())
			cleanupLocalPVCsPVs(config, []*localTestVolume{testVol})
		})

		It("should fail due to wrong node", func() {
			if len(config.nodes) < 2 {
				framework.Skipf("Runs only when number of nodes >= 2")
			}

			testVols := setupLocalVolumesPVCsPVs(config, DirectoryLocalVolumeType, config.node0, 1, immediateMode)
			testVol := testVols[0]

			pod := makeLocalPodWithNodeName(config, testVol, config.nodes[1].Name)
			pod, err := config.client.CoreV1().Pods(config.ns).Create(pod)
			Expect(err).NotTo(HaveOccurred())

			err = framework.WaitTimeoutForPodRunningInNamespace(config.client, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
			Expect(err).To(HaveOccurred())

			cleanupLocalVolumes(config, []*localTestVolume{testVol})
		})
	})

	Context("Pod with node different from PV's NodeAffinity", func() {
		var (
			testVol    *localTestVolume
			volumeType localVolumeType
		)

		BeforeEach(func() {
			if len(config.nodes) < 2 {
				framework.Skipf("Runs only when number of nodes >= 2")
			}

			volumeType = DirectoryLocalVolumeType
			setupStorageClass(config, &immediateMode)
			testVols := setupLocalVolumesPVCsPVs(config, volumeType, config.node0, 1, immediateMode)
			testVol = testVols[0]
		})

		AfterEach(func() {
			cleanupLocalVolumes(config, []*localTestVolume{testVol})
			cleanupStorageClass(config)
		})

		It("should fail scheduling due to different NodeAffinity", func() {
			testPodWithNodeConflict(config, volumeType, config.nodes[1].Name, makeLocalPodWithNodeAffinity, immediateMode)
		})

		It("should fail scheduling due to different NodeSelector", func() {
			testPodWithNodeConflict(config, volumeType, config.nodes[1].Name, makeLocalPodWithNodeSelector, immediateMode)
		})
	})

	Context("StatefulSet with pod affinity [Slow]", func() {
		var testVols map[string][]*localTestVolume
		const (
			ssReplicas  = 3
			volsPerNode = 6
		)

		BeforeEach(func() {
			setupStorageClass(config, &waitMode)

			testVols = map[string][]*localTestVolume{}
			for i, node := range config.nodes {
				// The PVCs created here won't be used
				By(fmt.Sprintf("Setting up local volumes on node %q", node.Name))
				vols := setupLocalVolumesPVCsPVs(config, DirectoryLocalVolumeType, &config.nodes[i], volsPerNode, waitMode)
				testVols[node.Name] = vols
			}
		})

		AfterEach(func() {
			for _, vols := range testVols {
				cleanupLocalVolumes(config, vols)
			}
			cleanupStorageClass(config)
		})

		It("should use volumes spread across nodes when pod has anti-affinity", func() {
			if len(config.nodes) < ssReplicas {
				framework.Skipf("Runs only when number of nodes >= %v", ssReplicas)
			}
			By("Creating a StatefulSet with pod anti-affinity on nodes")
			ss := createStatefulSet(config, ssReplicas, volsPerNode, true, false)
			validateStatefulSet(config, ss, true)
		})

		It("should use volumes on one node when pod has affinity", func() {
			By("Creating a StatefulSet with pod affinity on nodes")
			ss := createStatefulSet(config, ssReplicas, volsPerNode/ssReplicas, false, false)
			validateStatefulSet(config, ss, false)
		})

		It("should use volumes spread across nodes when pod management is parallel and pod has anti-affinity", func() {
			if len(config.nodes) < ssReplicas {
				framework.Skipf("Runs only when number of nodes >= %v", ssReplicas)
			}
			By("Creating a StatefulSet with pod anti-affinity on nodes")
			ss := createStatefulSet(config, ssReplicas, 1, true, true)
			validateStatefulSet(config, ss, true)
		})

		It("should use volumes on one node when pod management is parallel and pod has affinity", func() {
			By("Creating a StatefulSet with pod affinity on nodes")
			ss := createStatefulSet(config, ssReplicas, 1, false, true)
			validateStatefulSet(config, ss, false)
		})
	})

	Context("Stress with local volumes [Serial]", func() {
		var (
			allLocalVolumes = make(map[string][]*localTestVolume)
			volType         = TmpfsLocalVolumeType
			stopCh          = make(chan struct{})
			wg              sync.WaitGroup
		)

		const (
			volsPerNode = 10 // Make this non-divisable by volsPerPod to increase changes of partial binding failure
			volsPerPod  = 3
			podsFactor  = 4
		)

		BeforeEach(func() {
			setupStorageClass(config, &waitMode)
			for i, node := range config.nodes {
				By(fmt.Sprintf("Setting up %d local volumes on node %q", volsPerNode, node.Name))
				allLocalVolumes[node.Name] = setupLocalVolumes(config, volType, &config.nodes[i], volsPerNode)
			}
			By(fmt.Sprintf("Create %d PVs", volsPerNode*len(config.nodes)))
			var err error
			for _, localVolumes := range allLocalVolumes {
				for _, localVolume := range localVolumes {
					pvConfig := makeLocalPVConfig(config, localVolume)
					localVolume.pv, err = framework.CreatePV(config.client, framework.MakePersistentVolume(pvConfig))
					framework.ExpectNoError(err)
				}
			}
			By("Start a goroutine to recycle unbound PVs")
			wg.Add(1)
			go func() {
				defer wg.Done()
				w, err := config.client.CoreV1().PersistentVolumes().Watch(metav1.ListOptions{})
				framework.ExpectNoError(err)
				if w == nil {
					return
				}
				defer w.Stop()
				for {
					select {
					case event := <-w.ResultChan():
						if event.Type != watch.Modified {
							continue
						}
						pv, ok := event.Object.(*v1.PersistentVolume)
						if !ok {
							continue
						}
						if pv.Status.Phase == v1.VolumeBound || pv.Status.Phase == v1.VolumeAvailable {
							continue
						}
						pv, err = config.client.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
						if apierrors.IsNotFound(err) {
							continue
						}
						// Delete and create a new PV for same local volume storage
						By(fmt.Sprintf("Delete %q and create a new PV for same local volume storage", pv.Name))
						for _, localVolumes := range allLocalVolumes {
							for _, localVolume := range localVolumes {
								if localVolume.pv.Name != pv.Name {
									continue
								}
								err = config.client.CoreV1().PersistentVolumes().Delete(pv.Name, &metav1.DeleteOptions{})
								framework.ExpectNoError(err)
								pvConfig := makeLocalPVConfig(config, localVolume)
								localVolume.pv, err = framework.CreatePV(config.client, framework.MakePersistentVolume(pvConfig))
								framework.ExpectNoError(err)
							}
						}
					case <-stopCh:
						return
					}
				}
			}()
		})

		AfterEach(func() {
			By("Stop and wait for recycle goroutine to finish")
			close(stopCh)
			wg.Wait()
			By("Clean all PVs")
			for nodeName, localVolumes := range allLocalVolumes {
				By(fmt.Sprintf("Cleaning up %d local volumes on node %q", len(localVolumes), nodeName))
				cleanupLocalVolumes(config, localVolumes)
			}
			cleanupStorageClass(config)
		})

		It("should be able to process many pods and reuse local volumes", func() {
			var (
				podsLock sync.Mutex
				// Have one extra pod pending
				numConcurrentPods = volsPerNode/volsPerPod*len(config.nodes) + 1
				totalPods         = numConcurrentPods * podsFactor
				numCreated        = 0
				numFinished       = 0
				pods              = map[string]*v1.Pod{}
			)

			// Create pods gradually instead of all at once because scheduler has
			// exponential backoff
			By(fmt.Sprintf("Creating %v pods periodically", numConcurrentPods))
			stop := make(chan struct{})
			go wait.Until(func() {
				podsLock.Lock()
				defer podsLock.Unlock()

				if numCreated >= totalPods {
					// Created all the pods for the test
					return
				}

				if len(pods) > numConcurrentPods/2 {
					// Too many outstanding pods
					return
				}

				for i := 0; i < numConcurrentPods; i++ {
					pvcs := []*v1.PersistentVolumeClaim{}
					for j := 0; j < volsPerPod; j++ {
						pvc := framework.MakePersistentVolumeClaim(makeLocalPVCConfig(config, volType), config.ns)
						pvc, err := framework.CreatePVC(config.client, config.ns, pvc)
						framework.ExpectNoError(err)
						pvcs = append(pvcs, pvc)
					}

					pod := framework.MakeSecPod(config.ns, pvcs, false, "sleep 1", false, false, selinuxLabel, nil)
					pod, err := config.client.CoreV1().Pods(config.ns).Create(pod)
					Expect(err).NotTo(HaveOccurred())
					pods[pod.Name] = pod
					numCreated++
				}
			}, 2*time.Second, stop)

			defer func() {
				close(stop)
				podsLock.Lock()
				defer podsLock.Unlock()

				for _, pod := range pods {
					if err := deletePodAndPVCs(config, pod); err != nil {
						framework.Logf("Deleting pod %v failed: %v", pod.Name, err)
					}
				}
			}()

			By("Waiting for all pods to complete successfully")
			err := wait.PollImmediate(time.Second, 5*time.Minute, func() (done bool, err error) {
				podsList, err := config.client.CoreV1().Pods(config.ns).List(metav1.ListOptions{})
				if err != nil {
					return false, err
				}

				podsLock.Lock()
				defer podsLock.Unlock()

				for _, pod := range podsList.Items {
					switch pod.Status.Phase {
					case v1.PodSucceeded:
						// Delete pod and its PVCs
						if err := deletePodAndPVCs(config, &pod); err != nil {
							return false, err
						}
						delete(pods, pod.Name)
						numFinished++
						framework.Logf("%v/%v pods finished", numFinished, totalPods)
					case v1.PodFailed:
					case v1.PodUnknown:
						return false, fmt.Errorf("pod %v is in %v phase", pod.Name, pod.Status.Phase)
					}
				}

				return numFinished == totalPods, nil
			})
			Expect(err).ToNot(HaveOccurred())
		})
	})

	Context("Pods sharing a single local PV [Serial]", func() {
		var (
			pv *v1.PersistentVolume
		)

		BeforeEach(func() {
			localVolume := &localTestVolume{
				ltr: &utils.LocalTestResource{
					Node: config.node0,
					Path: "/tmp",
				},
				localVolumeType: DirectoryLocalVolumeType,
			}
			pvConfig := makeLocalPVConfig(config, localVolume)
			var err error
			pv, err = framework.CreatePV(config.client, framework.MakePersistentVolume(pvConfig))
			framework.ExpectNoError(err)
		})

		AfterEach(func() {
			if pv == nil {
				return
			}
			By(fmt.Sprintf("Clean PV %s", pv.Name))
			err := config.client.CoreV1().PersistentVolumes().Delete(pv.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		It("all pods should be running", func() {
			var (
				pvc   *v1.PersistentVolumeClaim
				pods  = map[string]*v1.Pod{}
				count = 50
				err   error
			)
			pvc = framework.MakePersistentVolumeClaim(makeLocalPVCConfig(config, DirectoryLocalVolumeType), config.ns)
			By(fmt.Sprintf("Create a PVC %s", pvc.Name))
			pvc, err = framework.CreatePVC(config.client, config.ns, pvc)
			framework.ExpectNoError(err)
			By(fmt.Sprintf("Create %d pods to use this PVC", count))
			for i := 0; i < count; i++ {
				pod := framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{pvc}, false, "", false, false, selinuxLabel, nil)
				pod, err := config.client.CoreV1().Pods(config.ns).Create(pod)
				Expect(err).NotTo(HaveOccurred())
				pods[pod.Name] = pod
			}
			By("Wait for all pods are running")
			err = wait.PollImmediate(time.Second, 5*time.Minute, func() (done bool, err error) {
				podsList, err := config.client.CoreV1().Pods(config.ns).List(metav1.ListOptions{})
				if err != nil {
					return false, err
				}
				runningPods := 0
				for _, pod := range podsList.Items {
					switch pod.Status.Phase {
					case v1.PodRunning:
						runningPods++
					}
				}
				return runningPods == count, nil
			})
			Expect(err).ToNot(HaveOccurred())
		})
	})
})

func deletePodAndPVCs(config *localTestConfig, pod *v1.Pod) error {
	framework.Logf("Deleting pod %v", pod.Name)
	if err := config.client.CoreV1().Pods(config.ns).Delete(pod.Name, nil); err != nil {
		return err
	}

	// Delete PVCs
	for _, vol := range pod.Spec.Volumes {
		pvcSource := vol.VolumeSource.PersistentVolumeClaim
		if pvcSource != nil {
			if err := framework.DeletePersistentVolumeClaim(config.client, pvcSource.ClaimName, config.ns); err != nil {
				return err
			}
		}
	}
	return nil
}

type makeLocalPodWith func(config *localTestConfig, volume *localTestVolume, nodeName string) *v1.Pod

func testPodWithNodeConflict(config *localTestConfig, testVolType localVolumeType, nodeName string, makeLocalPodFunc makeLocalPodWith, bindingMode storagev1.VolumeBindingMode) {
	By(fmt.Sprintf("local-volume-type: %s", testVolType))
	testVols := setupLocalVolumesPVCsPVs(config, testVolType, config.node0, 1, bindingMode)
	testVol := testVols[0]

	pod := makeLocalPodFunc(config, testVol, nodeName)
	pod, err := config.client.CoreV1().Pods(config.ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForPodNameUnschedulableInNamespace(config.client, pod.Name, pod.Namespace)
	Expect(err).NotTo(HaveOccurred())
}

// The tests below are run against multiple mount point types

// Test two pods at the same time, write from pod1, and read from pod2
func twoPodsReadWriteTest(config *localTestConfig, testVol *localTestVolume) {
	By("Creating pod1 to write to the PV")
	pod1, pod1Err := createLocalPod(config, testVol, nil)
	Expect(pod1Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod1, config.node0.Name)

	writeCmd := createWriteCmd(volumeDir, testFile, testFileContent, testVol.localVolumeType)

	By("Writing in pod1")
	podRWCmdExec(pod1, writeCmd)

	// testFileContent was written after creating pod1
	testReadFileContent(volumeDir, testFile, testFileContent, pod1, testVol.localVolumeType)

	By("Creating pod2 to read from the PV")
	pod2, pod2Err := createLocalPod(config, testVol, nil)
	Expect(pod2Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod2, config.node0.Name)

	// testFileContent was written after creating pod1
	testReadFileContent(volumeDir, testFile, testFileContent, pod2, testVol.localVolumeType)

	writeCmd = createWriteCmd(volumeDir, testFile, testVol.ltr.Path /*writeTestFileContent*/, testVol.localVolumeType)

	By("Writing in pod2")
	podRWCmdExec(pod2, writeCmd)

	By("Reading in pod1")
	testReadFileContent(volumeDir, testFile, testVol.ltr.Path, pod1, testVol.localVolumeType)

	By("Deleting pod1")
	framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
	By("Deleting pod2")
	framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
}

// Test two pods one after other, write from pod1, and read from pod2
func twoPodsReadWriteSerialTest(config *localTestConfig, testVol *localTestVolume) {
	By("Creating pod1")
	pod1, pod1Err := createLocalPod(config, testVol, nil)
	Expect(pod1Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod1, config.node0.Name)

	writeCmd := createWriteCmd(volumeDir, testFile, testFileContent, testVol.localVolumeType)

	By("Writing in pod1")
	podRWCmdExec(pod1, writeCmd)

	// testFileContent was written after creating pod1
	testReadFileContent(volumeDir, testFile, testFileContent, pod1, testVol.localVolumeType)

	By("Deleting pod1")
	framework.DeletePodOrFail(config.client, config.ns, pod1.Name)

	By("Creating pod2")
	pod2, pod2Err := createLocalPod(config, testVol, nil)
	Expect(pod2Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod2, config.node0.Name)

	By("Reading in pod2")
	testReadFileContent(volumeDir, testFile, testFileContent, pod2, testVol.localVolumeType)

	By("Deleting pod2")
	framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
}

// Test creating pod with fsGroup, and check fsGroup is expected fsGroup.
func createPodWithFsGroupTest(config *localTestConfig, testVol *localTestVolume, fsGroup int64, expectedFsGroup int64) *v1.Pod {
	pod, err := createLocalPod(config, testVol, &fsGroup)
	framework.ExpectNoError(err)
	_, err = framework.LookForStringInPodExec(config.ns, pod.Name, []string{"stat", "-c", "%g", volumeDir}, strconv.FormatInt(expectedFsGroup, 10), time.Second*3)
	Expect(err).NotTo(HaveOccurred(), "failed to get expected fsGroup %d on directory %s in pod %s", fsGroup, volumeDir, pod.Name)
	return pod
}

func setupStorageClass(config *localTestConfig, mode *storagev1.VolumeBindingMode) {
	sc := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.scName,
		},
		Provisioner:       "kubernetes.io/no-provisioner",
		VolumeBindingMode: mode,
	}

	_, err := config.client.StorageV1().StorageClasses().Create(sc)
	Expect(err).NotTo(HaveOccurred())
}

func cleanupStorageClass(config *localTestConfig) {
	framework.ExpectNoError(config.client.StorageV1().StorageClasses().Delete(config.scName, nil))
}

// podNode wraps RunKubectl to get node where pod is running
func podNodeName(config *localTestConfig, pod *v1.Pod) (string, error) {
	runtimePod, runtimePodErr := config.client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	return runtimePod.Spec.NodeName, runtimePodErr
}

// setupLocalVolumes sets up directories to use for local PV
func setupLocalVolumes(config *localTestConfig, localVolumeType localVolumeType, node *v1.Node, count int) []*localTestVolume {
	vols := []*localTestVolume{}
	for i := 0; i < count; i++ {
		ltrType, ok := setupLocalVolumeMap[localVolumeType]
		Expect(ok).To(BeTrue())
		ltr := config.ltrMgr.Create(node, ltrType, nil)
		vols = append(vols, &localTestVolume{
			ltr:             ltr,
			localVolumeType: localVolumeType,
		})
	}
	return vols
}

func cleanupLocalPVCsPVs(config *localTestConfig, volumes []*localTestVolume) {
	for _, volume := range volumes {
		By("Cleaning up PVC and PV")
		errs := framework.PVPVCCleanup(config.client, config.ns, volume.pv, volume.pvc)
		if len(errs) > 0 {
			framework.Failf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
		}
	}
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolumes(config *localTestConfig, volumes []*localTestVolume) {
	cleanupLocalPVCsPVs(config, volumes)

	for _, volume := range volumes {
		config.ltrMgr.Remove(volume.ltr)
	}
}

func verifyLocalVolume(config *localTestConfig, volume *localTestVolume) {
	framework.ExpectNoError(framework.WaitOnPVandPVC(config.client, config.ns, volume.pv, volume.pvc))
}

func verifyLocalPod(config *localTestConfig, volume *localTestVolume, pod *v1.Pod, expectedNodeName string) {
	podNodeName, err := podNodeName(config, pod)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("pod %q created on Node %q", pod.Name, podNodeName)
	Expect(podNodeName).To(Equal(expectedNodeName))
}

func makeLocalPVCConfig(config *localTestConfig, volumeType localVolumeType) framework.PersistentVolumeClaimConfig {
	pvcConfig := framework.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &config.scName,
	}
	if volumeType == BlockLocalVolumeType {
		pvcVolumeMode := v1.PersistentVolumeBlock
		pvcConfig.VolumeMode = &pvcVolumeMode
	}
	return pvcConfig
}

func makeLocalPVConfig(config *localTestConfig, volume *localTestVolume) framework.PersistentVolumeConfig {
	// TODO: hostname may not be the best option
	nodeKey := "kubernetes.io/hostname"
	if volume.ltr.Node.Labels == nil {
		framework.Failf("Node does not have labels")
	}
	nodeValue, found := volume.ltr.Node.Labels[nodeKey]
	if !found {
		framework.Failf("Node does not have required label %q", nodeKey)
	}

	pvConfig := framework.PersistentVolumeConfig{
		PVSource: v1.PersistentVolumeSource{
			Local: &v1.LocalVolumeSource{
				Path: volume.ltr.Path,
			},
		},
		NamePrefix:       "local-pv",
		StorageClassName: config.scName,
		NodeAffinity: &v1.VolumeNodeAffinity{
			Required: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      nodeKey,
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{nodeValue},
							},
						},
					},
				},
			},
		},
	}

	if volume.localVolumeType == BlockLocalVolumeType {
		pvVolumeMode := v1.PersistentVolumeBlock
		pvConfig.VolumeMode = &pvVolumeMode
	}
	return pvConfig
}

// Creates a PVC and PV with prebinding
func createLocalPVCsPVs(config *localTestConfig, volumes []*localTestVolume, mode storagev1.VolumeBindingMode) {
	var err error

	for _, volume := range volumes {
		pvcConfig := makeLocalPVCConfig(config, volume.localVolumeType)
		pvConfig := makeLocalPVConfig(config, volume)

		volume.pv, volume.pvc, err = framework.CreatePVPVC(config.client, pvConfig, pvcConfig, config.ns, false)
		framework.ExpectNoError(err)
	}

	if mode == storagev1.VolumeBindingImmediate {
		for _, volume := range volumes {
			verifyLocalVolume(config, volume)
		}
	} else {
		// Verify PVCs are not bound
		// There isn't really a great way to verify this without making the test be slow...
		err = wait.PollImmediate(time.Second, 10*time.Second, func() (done bool, err error) {
			for _, volume := range volumes {
				pvc, err := config.client.CoreV1().PersistentVolumeClaims(volume.pvc.Namespace).Get(volume.pvc.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				Expect(pvc.Status.Phase).To(Equal(v1.ClaimPending))
			}
			return false, nil
		})
		Expect(err).To(HaveOccurred())
	}
}

func makeLocalPodWithNodeAffinity(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	pod = framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel, nil)
	if pod == nil {
		return
	}
	affinity := &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "kubernetes.io/hostname",
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{nodeName},
							},
						},
					},
				},
			},
		},
	}
	pod.Spec.Affinity = affinity
	return
}

func makeLocalPodWithNodeSelector(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	pod = framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel, nil)
	if pod == nil {
		return
	}
	ns := map[string]string{
		"kubernetes.io/hostname": nodeName,
	}
	pod.Spec.NodeSelector = ns
	return
}

func makeLocalPodWithNodeName(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	pod = framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel, nil)
	if pod == nil {
		return
	}
	pod.Spec.NodeName = nodeName
	return
}

func createLocalPod(config *localTestConfig, volume *localTestVolume, fsGroup *int64) (*v1.Pod, error) {
	By("Creating a pod")
	return framework.CreateSecPod(config.client, config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel, fsGroup, framework.PodStartShortTimeout)
}

func createWriteCmd(testDir string, testFile string, writeTestFileContent string, volumeType localVolumeType) string {
	if volumeType == BlockLocalVolumeType {
		// testDir is the block device.
		testFileDir := filepath.Join("/tmp", testDir)
		testFilePath := filepath.Join(testFileDir, testFile)
		// Create a file containing the testFileContent.
		writeTestFileCmd := fmt.Sprintf("mkdir -p %s; echo %s > %s", testFileDir, writeTestFileContent, testFilePath)
		// sudo is needed when using ssh exec to node.
		// sudo is not needed and does not exist in some containers (e.g. busybox), when using pod exec.
		sudoCmd := fmt.Sprintf("SUDO_CMD=$(which sudo); echo ${SUDO_CMD}")
		// Write the testFileContent into the block device.
		writeBlockCmd := fmt.Sprintf("${SUDO_CMD} dd if=%s of=%s bs=512 count=100", testFilePath, testDir)
		// Cleanup the file containing testFileContent.
		deleteTestFileCmd := fmt.Sprintf("rm %s", testFilePath)
		return fmt.Sprintf("%s && %s && %s && %s", writeTestFileCmd, sudoCmd, writeBlockCmd, deleteTestFileCmd)
	} else {
		testFilePath := filepath.Join(testDir, testFile)
		return fmt.Sprintf("mkdir -p %s; echo %s > %s", testDir, writeTestFileContent, testFilePath)
	}
}

func createReadCmd(testFileDir string, testFile string, volumeType localVolumeType) string {
	if volumeType == BlockLocalVolumeType {
		// Create the command to read the beginning of the block device and print it in ascii.
		return fmt.Sprintf("hexdump -n 100 -e '100 \"%%_p\"' %s | head -1", testFileDir)
	} else {
		// Create the command to read (aka cat) a file.
		testFilePath := filepath.Join(testFileDir, testFile)
		return fmt.Sprintf("cat %s", testFilePath)
	}
}

// Read testFile and evaluate whether it contains the testFileContent
func testReadFileContent(testFileDir string, testFile string, testFileContent string, pod *v1.Pod, volumeType localVolumeType) {
	readCmd := createReadCmd(testFileDir, testFile, volumeType)
	readOut := podRWCmdExec(pod, readCmd)
	Expect(readOut).To(ContainSubstring(testFileContent))
}

// Execute a read or write command in a pod.
// Fail on error
func podRWCmdExec(pod *v1.Pod, cmd string) string {
	out, err := utils.PodExec(pod, cmd)
	framework.Logf("podRWCmdExec out: %q err: %v", out, err)
	Expect(err).NotTo(HaveOccurred())
	return out
}

// Initialize test volume on node
// and create local PVC and PV
func setupLocalVolumesPVCsPVs(
	config *localTestConfig,
	localVolumeType localVolumeType,
	node *v1.Node,
	count int,
	mode storagev1.VolumeBindingMode) []*localTestVolume {

	By("Initializing test volumes")
	testVols := setupLocalVolumes(config, localVolumeType, node, count)

	By("Creating local PVCs and PVs")
	createLocalPVCsPVs(config, testVols, mode)

	return testVols
}

// newLocalClaim creates a new persistent volume claim.
func newLocalClaimWithName(config *localTestConfig, name string) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: config.ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: &config.scName,
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(testRequestSize),
				},
			},
		},
	}

	return &claim
}

func createStatefulSet(config *localTestConfig, ssReplicas int32, volumeCount int, anti, parallel bool) *appsv1.StatefulSet {
	mounts := []v1.VolumeMount{}
	claims := []v1.PersistentVolumeClaim{}
	for i := 0; i < volumeCount; i++ {
		name := fmt.Sprintf("vol%v", i+1)
		pvc := newLocalClaimWithName(config, name)
		mounts = append(mounts, v1.VolumeMount{Name: name, MountPath: "/" + name})
		claims = append(claims, *pvc)
	}

	podAffinityTerms := []v1.PodAffinityTerm{
		{
			LabelSelector: &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      "app",
						Operator: metav1.LabelSelectorOpIn,
						Values:   []string{"local-volume-test"},
					},
				},
			},
			TopologyKey: "kubernetes.io/hostname",
		},
	}

	affinity := v1.Affinity{}
	if anti {
		affinity.PodAntiAffinity = &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: podAffinityTerms,
		}
	} else {
		affinity.PodAffinity = &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: podAffinityTerms,
		}
	}

	labels := map[string]string{"app": "local-volume-test"}
	spec := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "local-volume-statefulset",
			Namespace: config.ns,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "local-volume-test"},
			},
			Replicas: &ssReplicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:         "nginx",
							Image:        imageutils.GetE2EImage(imageutils.Nginx),
							VolumeMounts: mounts,
						},
					},
					Affinity: &affinity,
				},
			},
			VolumeClaimTemplates: claims,
			ServiceName:          "test-service",
		},
	}

	if parallel {
		spec.Spec.PodManagementPolicy = appsv1.ParallelPodManagement
	}

	ss, err := config.client.AppsV1().StatefulSets(config.ns).Create(spec)
	Expect(err).NotTo(HaveOccurred())

	config.ssTester.WaitForRunningAndReady(ssReplicas, ss)
	return ss
}

func validateStatefulSet(config *localTestConfig, ss *appsv1.StatefulSet, anti bool) {
	pods := config.ssTester.GetPodList(ss)

	nodes := sets.NewString()
	for _, pod := range pods.Items {
		nodes.Insert(pod.Spec.NodeName)
	}

	if anti {
		// Verify that each pod is on a different node
		Expect(nodes.Len()).To(Equal(len(pods.Items)))
	} else {
		// Verify that all pods are on same node.
		Expect(nodes.Len()).To(Equal(1))
	}

	// Validate all PVCs are bound
	for _, pod := range pods.Items {
		for _, volume := range pod.Spec.Volumes {
			pvcSource := volume.VolumeSource.PersistentVolumeClaim
			if pvcSource != nil {
				err := framework.WaitForPersistentVolumeClaimPhase(
					v1.ClaimBound, config.client, config.ns, pvcSource.ClaimName, framework.Poll, time.Second)
				Expect(err).NotTo(HaveOccurred())
			}
		}
	}
}

// SkipUnlessLocalSSDExists takes in an ssdInterface (scsi/nvme) and a filesystemType (fs/block)
// and skips if a disk of that type does not exist on the node
func SkipUnlessLocalSSDExists(config *localTestConfig, ssdInterface, filesystemType string, node *v1.Node) {
	ssdCmd := fmt.Sprintf("ls -1 /mnt/disks/by-uuid/google-local-ssds-%s-%s/ | wc -l", ssdInterface, filesystemType)
	res, err := config.hostExec.IssueCommandWithResult(ssdCmd, node)
	Expect(err).NotTo(HaveOccurred())
	num, err := strconv.Atoi(strings.TrimSpace(res))
	Expect(err).NotTo(HaveOccurred())
	if num < 1 {
		framework.Skipf("Requires at least 1 %s %s localSSD ", ssdInterface, filesystemType)
	}
}
