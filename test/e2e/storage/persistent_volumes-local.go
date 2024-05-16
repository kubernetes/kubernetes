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
	"context"
	"errors"
	"fmt"
	"math/rand"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
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
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

type localTestConfig struct {
	ns           string
	nodes        []v1.Node
	randomNode   *v1.Node
	client       clientset.Interface
	timeouts     *framework.TimeoutContext
	scName       string
	discoveryDir string
	hostExec     utils.HostExec
	ltrMgr       utils.LocalTestResourceManager
}

type localVolumeType string

const (
	// DirectoryLocalVolumeType is the default local volume type, aka a directory
	DirectoryLocalVolumeType localVolumeType = "dir"
	// DirectoryLinkLocalVolumeType is like DirectoryLocalVolumeType,
	// but it's a symbolic link to directory
	DirectoryLinkLocalVolumeType localVolumeType = "dir-link"
	// DirectoryBindMountedLocalVolumeType is like DirectoryLocalVolumeType
	// but bind mounted
	DirectoryBindMountedLocalVolumeType localVolumeType = "dir-bindmounted"
	// DirectoryLinkBindMountedLocalVolumeType is like DirectoryLocalVolumeType,
	// but it's a symbolic link to self bind mounted directory
	// Note that bind mounting at symbolic link actually mounts at directory it
	// links to.
	DirectoryLinkBindMountedLocalVolumeType localVolumeType = "dir-link-bindmounted"
	// TmpfsLocalVolumeType creates a tmpfs and mounts it
	TmpfsLocalVolumeType localVolumeType = "tmpfs"
	// GCELocalSSDVolumeType tests based on local ssd at /mnt/disks/by-uuid/
	GCELocalSSDVolumeType localVolumeType = "gce-localssd-scsi-fs"
	// BlockLocalVolumeType creates a local file, formats it, and maps it as a block device.
	BlockLocalVolumeType localVolumeType = "block"
	// BlockFsWithFormatLocalVolumeType creates a local file serving as the backing for block device,
	// formats it, and mounts it to use as FS mode local volume.
	BlockFsWithFormatLocalVolumeType localVolumeType = "blockfswithformat"
	// BlockFsWithoutFormatLocalVolumeType creates a local file serving as the backing for block device,
	// does not format it manually, and mounts it to use as FS mode local volume.
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

var _ = utils.SIGDescribe("PersistentVolumes-local", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		config *localTestConfig
		scName string
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, f.ClientSet, maxNodes)
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
	})

	for tempTestVolType := range setupLocalVolumeMap {

		// New variable required for ginkgo test closures
		testVolType := tempTestVolType
		args := []interface{}{fmt.Sprintf("[Volume type: %s]", testVolType)}
		if testVolType == GCELocalSSDVolumeType {
			args = append(args, framework.WithSerial())
		}
		testMode := immediateMode

		args = append(args, func() {
			var testVol *localTestVolume

			ginkgo.BeforeEach(func(ctx context.Context) {
				if testVolType == GCELocalSSDVolumeType {
					SkipUnlessLocalSSDExists(ctx, config, "scsi", "fs", config.randomNode)
				}
				setupStorageClass(ctx, config, &testMode)
				testVols := setupLocalVolumesPVCsPVs(ctx, config, testVolType, config.randomNode, 1, testMode)
				if len(testVols) > 0 {
					testVol = testVols[0]
				} else {
					framework.Failf("Failed to get a test volume")
				}
			})

			ginkgo.AfterEach(func(ctx context.Context) {
				if testVol != nil {
					cleanupLocalVolumes(ctx, config, []*localTestVolume{testVol})
					cleanupStorageClass(ctx, config)
				} else {
					framework.Failf("no test volume to cleanup")
				}
			})

			ginkgo.Context("One pod requesting one prebound PVC", func() {
				var (
					pod1    *v1.Pod
					pod1Err error
				)

				ginkgo.BeforeEach(func(ctx context.Context) {
					ginkgo.By("Creating pod1")
					pod1, pod1Err = createLocalPod(ctx, config, testVol, nil)
					framework.ExpectNoError(pod1Err)
					verifyLocalPod(ctx, config, testVol, pod1, config.randomNode.Name)

					writeCmd := createWriteCmd(volumeDir, testFile, testFileContent, testVol.localVolumeType)

					ginkgo.By("Writing in pod1")
					podRWCmdExec(f, pod1, writeCmd)
				})

				ginkgo.AfterEach(func(ctx context.Context) {
					ginkgo.By("Deleting pod1")
					e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod1.Name)
				})

				ginkgo.It("should be able to mount volume and read from pod1", func(ctx context.Context) {
					ginkgo.By("Reading in pod1")
					// testFileContent was written in BeforeEach
					testReadFileContent(f, volumeDir, testFile, testFileContent, pod1, testVolType)
				})

				ginkgo.It("should be able to mount volume and write from pod1", func(ctx context.Context) {
					// testFileContent was written in BeforeEach
					testReadFileContent(f, volumeDir, testFile, testFileContent, pod1, testVolType)

					ginkgo.By("Writing in pod1")
					writeCmd := createWriteCmd(volumeDir, testFile, testVol.ltr.Path /*writeTestFileContent*/, testVolType)
					podRWCmdExec(f, pod1, writeCmd)
				})
			})

			ginkgo.Context("Two pods mounting a local volume at the same time", func() {
				ginkgo.It("should be able to write from pod1 and read from pod2", func(ctx context.Context) {
					twoPodsReadWriteTest(ctx, f, config, testVol)
				})
			})

			ginkgo.Context("Two pods mounting a local volume one after the other", func() {
				ginkgo.It("should be able to write from pod1 and read from pod2", func(ctx context.Context) {
					twoPodsReadWriteSerialTest(ctx, f, config, testVol)
				})
			})

			ginkgo.Context("Set fsGroup for local volume", func() {
				ginkgo.BeforeEach(func() {
					if testVolType == BlockLocalVolumeType {
						e2eskipper.Skipf("We don't set fsGroup on block device, skipped.")
					}
				})

				f.It("should set fsGroup for one pod", f.WithSlow(), func(ctx context.Context) {
					ginkgo.By("Checking fsGroup is set")
					pod := createPodWithFsGroupTest(ctx, config, testVol, 1234, 1234)
					ginkgo.By("Deleting pod")
					e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod.Name)
				})

				f.It("should set same fsGroup for two pods simultaneously", f.WithSlow(), func(ctx context.Context) {
					fsGroup := int64(1234)
					ginkgo.By("Create first pod and check fsGroup is set")
					pod1 := createPodWithFsGroupTest(ctx, config, testVol, fsGroup, fsGroup)
					ginkgo.By("Create second pod with same fsGroup and check fsGroup is correct")
					pod2 := createPodWithFsGroupTest(ctx, config, testVol, fsGroup, fsGroup)
					ginkgo.By("Deleting first pod")
					e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod1.Name)
					ginkgo.By("Deleting second pod")
					e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod2.Name)
				})

				f.It("should set different fsGroup for second pod if first pod is deleted", f.WithFlaky(), func(ctx context.Context) {
					// TODO: Disabled temporarily, remove [Flaky] tag after #73168 is fixed.
					fsGroup1, fsGroup2 := int64(1234), int64(4321)
					ginkgo.By("Create first pod and check fsGroup is set")
					pod1 := createPodWithFsGroupTest(ctx, config, testVol, fsGroup1, fsGroup1)
					ginkgo.By("Deleting first pod")
					err := e2epod.DeletePodWithWait(ctx, config.client, pod1)
					framework.ExpectNoError(err, "while deleting first pod")
					ginkgo.By("Create second pod and check fsGroup is the new one")
					pod2 := createPodWithFsGroupTest(ctx, config, testVol, fsGroup2, fsGroup2)
					ginkgo.By("Deleting second pod")
					e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod2.Name)
				})
			})
		})
		f.Context(args...)
	}

	f.Context("Local volume that cannot be mounted", f.WithSlow(), func() {
		// TODO:
		// - check for these errors in unit tests instead
		ginkgo.It("should fail due to non-existent path", func(ctx context.Context) {
			testVol := &localTestVolume{
				ltr: &utils.LocalTestResource{
					Node: config.randomNode,
					Path: "/non-existent/location/nowhere",
				},
				localVolumeType: DirectoryLocalVolumeType,
			}
			ginkgo.By("Creating local PVC and PV")
			createLocalPVCsPVs(ctx, config, []*localTestVolume{testVol}, immediateMode)
			// createLocalPod will create a pod and wait for it to be running. In this case,
			// It's expected that the Pod fails to start.
			_, err := createLocalPod(ctx, config, testVol, nil)
			gomega.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("is not Running")))
			cleanupLocalPVCsPVs(ctx, config, []*localTestVolume{testVol})
		})

		ginkgo.It("should fail due to wrong node", func(ctx context.Context) {
			if len(config.nodes) < 2 {
				e2eskipper.Skipf("Runs only when number of nodes >= 2")
			}

			testVols := setupLocalVolumesPVCsPVs(ctx, config, DirectoryLocalVolumeType, config.randomNode, 1, immediateMode)
			testVol := testVols[0]

			conflictNodeName := config.nodes[0].Name
			if conflictNodeName == config.randomNode.Name {
				conflictNodeName = config.nodes[1].Name
			}
			pod := makeLocalPodWithNodeName(config, testVol, conflictNodeName)
			pod, err := config.client.CoreV1().Pods(config.ns).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			getPod := e2epod.Get(f.ClientSet, pod)
			gomega.Consistently(ctx, getPod, f.Timeouts.PodStart, 2*time.Second).ShouldNot(e2epod.BeInPhase(v1.PodRunning))

			cleanupLocalVolumes(ctx, config, []*localTestVolume{testVol})
		})
	})

	ginkgo.Context("Pod with node different from PV's NodeAffinity", func() {
		var (
			testVol          *localTestVolume
			volumeType       localVolumeType
			conflictNodeName string
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			if len(config.nodes) < 2 {
				e2eskipper.Skipf("Runs only when number of nodes >= 2")
			}

			volumeType = DirectoryLocalVolumeType
			setupStorageClass(ctx, config, &immediateMode)
			testVols := setupLocalVolumesPVCsPVs(ctx, config, volumeType, config.randomNode, 1, immediateMode)
			conflictNodeName = config.nodes[0].Name
			if conflictNodeName == config.randomNode.Name {
				conflictNodeName = config.nodes[1].Name
			}

			testVol = testVols[0]
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			cleanupLocalVolumes(ctx, config, []*localTestVolume{testVol})
			cleanupStorageClass(ctx, config)
		})

		ginkgo.It("should fail scheduling due to different NodeAffinity", func(ctx context.Context) {
			testPodWithNodeConflict(ctx, config, testVol, conflictNodeName, makeLocalPodWithNodeAffinity)
		})

		ginkgo.It("should fail scheduling due to different NodeSelector", func(ctx context.Context) {
			testPodWithNodeConflict(ctx, config, testVol, conflictNodeName, makeLocalPodWithNodeSelector)
		})
	})

	f.Context("StatefulSet with pod affinity", f.WithSlow(), func() {
		var testVols map[string][]*localTestVolume
		const (
			ssReplicas  = 3
			volsPerNode = 6
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			setupStorageClass(ctx, config, &waitMode)

			testVols = map[string][]*localTestVolume{}
			for i, node := range config.nodes {
				// The PVCs created here won't be used
				ginkgo.By(fmt.Sprintf("Setting up local volumes on node %q", node.Name))
				vols := setupLocalVolumesPVCsPVs(ctx, config, DirectoryLocalVolumeType, &config.nodes[i], volsPerNode, waitMode)
				testVols[node.Name] = vols
			}
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			for _, vols := range testVols {
				cleanupLocalVolumes(ctx, config, vols)
			}
			cleanupStorageClass(ctx, config)
		})

		ginkgo.It("should use volumes spread across nodes when pod has anti-affinity", func(ctx context.Context) {
			if len(config.nodes) < ssReplicas {
				e2eskipper.Skipf("Runs only when number of nodes >= %v", ssReplicas)
			}
			ginkgo.By("Creating a StatefulSet with pod anti-affinity on nodes")
			ss := createStatefulSet(ctx, config, ssReplicas, volsPerNode, true, false)
			validateStatefulSet(ctx, config, ss, true)
		})

		ginkgo.It("should use volumes on one node when pod has affinity", func(ctx context.Context) {
			ginkgo.By("Creating a StatefulSet with pod affinity on nodes")
			ss := createStatefulSet(ctx, config, ssReplicas, volsPerNode/ssReplicas, false, false)
			validateStatefulSet(ctx, config, ss, false)
		})

		ginkgo.It("should use volumes spread across nodes when pod management is parallel and pod has anti-affinity", func(ctx context.Context) {
			if len(config.nodes) < ssReplicas {
				e2eskipper.Skipf("Runs only when number of nodes >= %v", ssReplicas)
			}
			ginkgo.By("Creating a StatefulSet with pod anti-affinity on nodes")
			ss := createStatefulSet(ctx, config, ssReplicas, 1, true, true)
			validateStatefulSet(ctx, config, ss, true)
		})

		ginkgo.It("should use volumes on one node when pod management is parallel and pod has affinity", func(ctx context.Context) {
			ginkgo.By("Creating a StatefulSet with pod affinity on nodes")
			ss := createStatefulSet(ctx, config, ssReplicas, 1, false, true)
			validateStatefulSet(ctx, config, ss, false)
		})
	})

	f.Context("Stress with local volumes", f.WithSerial(), func() {
		var (
			allLocalVolumes = make(map[string][]*localTestVolume)
			volType         = TmpfsLocalVolumeType
		)

		const (
			volsPerNode = 10 // Make this non-divisable by volsPerPod to increase changes of partial binding failure
			volsPerPod  = 3
			podsFactor  = 4
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			setupStorageClass(ctx, config, &waitMode)
			ginkgo.DeferCleanup(cleanupStorageClass, config)

			for i, node := range config.nodes {
				ginkgo.By(fmt.Sprintf("Setting up %d local volumes on node %q", volsPerNode, node.Name))
				allLocalVolumes[node.Name] = setupLocalVolumes(ctx, config, volType, &config.nodes[i], volsPerNode)
			}
			ginkgo.By(fmt.Sprintf("Create %d PVs", volsPerNode*len(config.nodes)))
			var err error
			for _, localVolumes := range allLocalVolumes {
				for _, localVolume := range localVolumes {
					pvConfig := makeLocalPVConfig(config, localVolume)
					localVolume.pv, err = e2epv.CreatePV(ctx, config.client, f.Timeouts, e2epv.MakePersistentVolume(pvConfig))
					framework.ExpectNoError(err)
				}
			}
			ginkgo.DeferCleanup(func(ctx context.Context) {
				ginkgo.By("Clean all PVs")
				for nodeName, localVolumes := range allLocalVolumes {
					ginkgo.By(fmt.Sprintf("Cleaning up %d local volumes on node %q", len(localVolumes), nodeName))
					cleanupLocalVolumes(ctx, config, localVolumes)
				}
			})
			ginkgo.By("Start a goroutine to recycle unbound PVs")
			backgroundCtx, cancel := context.WithCancel(context.Background())
			var wg sync.WaitGroup
			wg.Add(1)
			ginkgo.DeferCleanup(func() {
				ginkgo.By("Stop and wait for recycle goroutine to finish")
				cancel()
				wg.Wait()
			})
			go func() {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				w, err := config.client.CoreV1().PersistentVolumes().Watch(backgroundCtx, metav1.ListOptions{})
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
						pv, err = config.client.CoreV1().PersistentVolumes().Get(backgroundCtx, pv.Name, metav1.GetOptions{})
						if apierrors.IsNotFound(err) || errors.Is(err, context.Canceled) {
							continue
						}
						// Delete and create a new PV for same local volume storage
						ginkgo.By(fmt.Sprintf("Delete %q and create a new PV for same local volume storage", pv.Name))
						for _, localVolumes := range allLocalVolumes {
							for _, localVolume := range localVolumes {
								if localVolume.pv.Name != pv.Name {
									continue
								}
								err = config.client.CoreV1().PersistentVolumes().Delete(backgroundCtx, pv.Name, metav1.DeleteOptions{})
								if apierrors.IsNotFound(err) || errors.Is(err, context.Canceled) {
									continue
								}
								framework.ExpectNoError(err)
								pvConfig := makeLocalPVConfig(config, localVolume)
								localVolume.pv, err = e2epv.CreatePV(backgroundCtx, config.client, f.Timeouts, e2epv.MakePersistentVolume(pvConfig))
								if errors.Is(err, context.Canceled) {
									continue
								}
								framework.ExpectNoError(err)
							}
						}
					case <-backgroundCtx.Done():
						return
					}
				}
			}()
		})

		ginkgo.It("should be able to process many pods and reuse local volumes", func(ctx context.Context) {
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
			ginkgo.By(fmt.Sprintf("Creating %v pods periodically", numConcurrentPods))
			stop := make(chan struct{})
			go wait.Until(func() {
				defer ginkgo.GinkgoRecover()
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
						pvc := e2epv.MakePersistentVolumeClaim(makeLocalPVCConfig(config, volType), config.ns)
						pvc, err := e2epv.CreatePVC(ctx, config.client, config.ns, pvc)
						framework.ExpectNoError(err)
						pvcs = append(pvcs, pvc)
					}
					podConfig := e2epod.Config{
						NS:           config.ns,
						PVCs:         pvcs,
						Command:      "sleep 1",
						SeLinuxLabel: selinuxLabel,
					}
					pod, err := e2epod.MakeSecPod(&podConfig)
					framework.ExpectNoError(err)
					pod, err = config.client.CoreV1().Pods(config.ns).Create(ctx, pod, metav1.CreateOptions{})
					framework.ExpectNoError(err)
					pods[pod.Name] = pod
					numCreated++
				}
			}, 2*time.Second, stop)

			defer func() {
				close(stop)
				podsLock.Lock()
				defer podsLock.Unlock()

				for _, pod := range pods {
					if err := deletePodAndPVCs(ctx, config, pod); err != nil {
						framework.Logf("Deleting pod %v failed: %v", pod.Name, err)
					}
				}
			}()

			ginkgo.By("Waiting for all pods to complete successfully")
			const completeTimeout = 5 * time.Minute
			waitErr := wait.PollUntilContextTimeout(ctx, time.Second, completeTimeout, true, func(ctx context.Context) (done bool, err error) {
				podsList, err := config.client.CoreV1().Pods(config.ns).List(ctx, metav1.ListOptions{})
				if err != nil {
					return false, err
				}

				podsLock.Lock()
				defer podsLock.Unlock()

				for _, pod := range podsList.Items {
					if pod.Status.Phase == v1.PodSucceeded {
						// Delete pod and its PVCs
						if err := deletePodAndPVCs(ctx, config, &pod); err != nil {
							return false, err
						}
						delete(pods, pod.Name)
						numFinished++
						framework.Logf("%v/%v pods finished", numFinished, totalPods)
					}
				}

				return numFinished == totalPods, nil
			})
			framework.ExpectNoError(waitErr, "some pods failed to complete within %v", completeTimeout)
		})
	})
})

func deletePodAndPVCs(ctx context.Context, config *localTestConfig, pod *v1.Pod) error {
	framework.Logf("Deleting pod %v", pod.Name)
	if err := config.client.CoreV1().Pods(config.ns).Delete(ctx, pod.Name, metav1.DeleteOptions{}); err != nil {
		return err
	}

	// Delete PVCs
	for _, vol := range pod.Spec.Volumes {
		pvcSource := vol.VolumeSource.PersistentVolumeClaim
		if pvcSource != nil {
			if err := e2epv.DeletePersistentVolumeClaim(ctx, config.client, pvcSource.ClaimName, config.ns); err != nil {
				return err
			}
		}
	}
	return nil
}

type makeLocalPodWith func(config *localTestConfig, volume *localTestVolume, nodeName string) *v1.Pod

func testPodWithNodeConflict(ctx context.Context, config *localTestConfig, testVol *localTestVolume, nodeName string, makeLocalPodFunc makeLocalPodWith) {
	ginkgo.By(fmt.Sprintf("local-volume-type: %s", testVol.localVolumeType))

	pod := makeLocalPodFunc(config, testVol, nodeName)
	pod, err := config.client.CoreV1().Pods(config.ns).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, config.client, pod.Name, pod.Namespace)
	framework.ExpectNoError(err)
}

// The tests below are run against multiple mount point types

// Test two pods at the same time, write from pod1, and read from pod2
func twoPodsReadWriteTest(ctx context.Context, f *framework.Framework, config *localTestConfig, testVol *localTestVolume) {
	ginkgo.By("Creating pod1 to write to the PV")
	pod1, pod1Err := createLocalPod(ctx, config, testVol, nil)
	framework.ExpectNoError(pod1Err)
	verifyLocalPod(ctx, config, testVol, pod1, config.randomNode.Name)

	writeCmd := createWriteCmd(volumeDir, testFile, testFileContent, testVol.localVolumeType)

	ginkgo.By("Writing in pod1")
	podRWCmdExec(f, pod1, writeCmd)

	// testFileContent was written after creating pod1
	testReadFileContent(f, volumeDir, testFile, testFileContent, pod1, testVol.localVolumeType)

	ginkgo.By("Creating pod2 to read from the PV")
	pod2, pod2Err := createLocalPod(ctx, config, testVol, nil)
	framework.ExpectNoError(pod2Err)
	verifyLocalPod(ctx, config, testVol, pod2, config.randomNode.Name)

	// testFileContent was written after creating pod1
	testReadFileContent(f, volumeDir, testFile, testFileContent, pod2, testVol.localVolumeType)

	writeCmd = createWriteCmd(volumeDir, testFile, testVol.ltr.Path /*writeTestFileContent*/, testVol.localVolumeType)

	ginkgo.By("Writing in pod2")
	podRWCmdExec(f, pod2, writeCmd)

	ginkgo.By("Reading in pod1")
	testReadFileContent(f, volumeDir, testFile, testVol.ltr.Path, pod1, testVol.localVolumeType)

	ginkgo.By("Deleting pod1")
	e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod1.Name)
	ginkgo.By("Deleting pod2")
	e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod2.Name)
}

// Test two pods one after other, write from pod1, and read from pod2
func twoPodsReadWriteSerialTest(ctx context.Context, f *framework.Framework, config *localTestConfig, testVol *localTestVolume) {
	ginkgo.By("Creating pod1")
	pod1, pod1Err := createLocalPod(ctx, config, testVol, nil)
	framework.ExpectNoError(pod1Err)
	verifyLocalPod(ctx, config, testVol, pod1, config.randomNode.Name)

	writeCmd := createWriteCmd(volumeDir, testFile, testFileContent, testVol.localVolumeType)

	ginkgo.By("Writing in pod1")
	podRWCmdExec(f, pod1, writeCmd)

	// testFileContent was written after creating pod1
	testReadFileContent(f, volumeDir, testFile, testFileContent, pod1, testVol.localVolumeType)

	ginkgo.By("Deleting pod1")
	e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod1.Name)

	ginkgo.By("Creating pod2")
	pod2, pod2Err := createLocalPod(ctx, config, testVol, nil)
	framework.ExpectNoError(pod2Err)
	verifyLocalPod(ctx, config, testVol, pod2, config.randomNode.Name)

	ginkgo.By("Reading in pod2")
	testReadFileContent(f, volumeDir, testFile, testFileContent, pod2, testVol.localVolumeType)

	ginkgo.By("Deleting pod2")
	e2epod.DeletePodOrFail(ctx, config.client, config.ns, pod2.Name)
}

// Test creating pod with fsGroup, and check fsGroup is expected fsGroup.
func createPodWithFsGroupTest(ctx context.Context, config *localTestConfig, testVol *localTestVolume, fsGroup int64, expectedFsGroup int64) *v1.Pod {
	pod, err := createLocalPod(ctx, config, testVol, &fsGroup)
	framework.ExpectNoError(err)
	_, err = e2eoutput.LookForStringInPodExec(config.ns, pod.Name, []string{"stat", "-c", "%g", volumeDir}, strconv.FormatInt(expectedFsGroup, 10), time.Second*3)
	framework.ExpectNoError(err, "failed to get expected fsGroup %d on directory %s in pod %s", fsGroup, volumeDir, pod.Name)
	return pod
}

func setupStorageClass(ctx context.Context, config *localTestConfig, mode *storagev1.VolumeBindingMode) {
	sc := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.scName,
		},
		Provisioner:       "kubernetes.io/no-provisioner",
		VolumeBindingMode: mode,
	}

	_, err := config.client.StorageV1().StorageClasses().Create(ctx, sc, metav1.CreateOptions{})
	framework.ExpectNoError(err)
}

func cleanupStorageClass(ctx context.Context, config *localTestConfig) {
	framework.ExpectNoError(config.client.StorageV1().StorageClasses().Delete(ctx, config.scName, metav1.DeleteOptions{}))
}

// podNode wraps RunKubectl to get node where pod is running
func podNodeName(ctx context.Context, config *localTestConfig, pod *v1.Pod) (string, error) {
	runtimePod, runtimePodErr := config.client.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
	return runtimePod.Spec.NodeName, runtimePodErr
}

// setupLocalVolumes sets up directories to use for local PV
func setupLocalVolumes(ctx context.Context, config *localTestConfig, localVolumeType localVolumeType, node *v1.Node, count int) []*localTestVolume {
	vols := []*localTestVolume{}
	for i := 0; i < count; i++ {
		ltrType, ok := setupLocalVolumeMap[localVolumeType]
		if !ok {
			framework.Failf("Invalid localVolumeType: %v", localVolumeType)
		}
		ltr := config.ltrMgr.Create(ctx, node, ltrType, nil)
		vols = append(vols, &localTestVolume{
			ltr:             ltr,
			localVolumeType: localVolumeType,
		})
	}
	return vols
}

func cleanupLocalPVCsPVs(ctx context.Context, config *localTestConfig, volumes []*localTestVolume) {
	for _, volume := range volumes {
		ginkgo.By("Cleaning up PVC and PV")
		errs := e2epv.PVPVCCleanup(ctx, config.client, config.ns, volume.pv, volume.pvc)
		if len(errs) > 0 {
			framework.Failf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
		}
	}
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolumes(ctx context.Context, config *localTestConfig, volumes []*localTestVolume) {
	cleanupLocalPVCsPVs(ctx, config, volumes)

	for _, volume := range volumes {
		config.ltrMgr.Remove(ctx, volume.ltr)
	}
}

func verifyLocalVolume(ctx context.Context, config *localTestConfig, volume *localTestVolume) {
	framework.ExpectNoError(e2epv.WaitOnPVandPVC(ctx, config.client, config.timeouts, config.ns, volume.pv, volume.pvc))
}

func verifyLocalPod(ctx context.Context, config *localTestConfig, volume *localTestVolume, pod *v1.Pod, expectedNodeName string) {
	podNodeName, err := podNodeName(ctx, config, pod)
	framework.ExpectNoError(err)
	framework.Logf("pod %q created on Node %q", pod.Name, podNodeName)
	gomega.Expect(podNodeName).To(gomega.Equal(expectedNodeName))
}

func makeLocalPVCConfig(config *localTestConfig, volumeType localVolumeType) e2epv.PersistentVolumeClaimConfig {
	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &config.scName,
	}
	if volumeType == BlockLocalVolumeType {
		pvcVolumeMode := v1.PersistentVolumeBlock
		pvcConfig.VolumeMode = &pvcVolumeMode
	}
	return pvcConfig
}

func makeLocalPVConfig(config *localTestConfig, volume *localTestVolume) e2epv.PersistentVolumeConfig {
	// TODO: hostname may not be the best option
	nodeKey := "kubernetes.io/hostname"
	if volume.ltr.Node.Labels == nil {
		framework.Failf("Node does not have labels")
	}
	nodeValue, found := volume.ltr.Node.Labels[nodeKey]
	if !found {
		framework.Failf("Node does not have required label %q", nodeKey)
	}

	pvConfig := e2epv.PersistentVolumeConfig{
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
func createLocalPVCsPVs(ctx context.Context, config *localTestConfig, volumes []*localTestVolume, mode storagev1.VolumeBindingMode) {
	var err error

	for _, volume := range volumes {
		pvcConfig := makeLocalPVCConfig(config, volume.localVolumeType)
		pvConfig := makeLocalPVConfig(config, volume)

		volume.pv, volume.pvc, err = e2epv.CreatePVPVC(ctx, config.client, config.timeouts, pvConfig, pvcConfig, config.ns, false)
		framework.ExpectNoError(err)
	}

	if mode == storagev1.VolumeBindingImmediate {
		for _, volume := range volumes {
			verifyLocalVolume(ctx, config, volume)
		}
	} else {
		// Verify PVCs are not bound by waiting for phase==bound with a timeout and asserting that we hit the timeout.
		// There isn't really a great way to verify this without making the test be slow...
		const bindTimeout = 10 * time.Second
		waitErr := wait.PollImmediate(time.Second, bindTimeout, func() (done bool, err error) {
			for _, volume := range volumes {
				pvc, err := config.client.CoreV1().PersistentVolumeClaims(volume.pvc.Namespace).Get(ctx, volume.pvc.Name, metav1.GetOptions{})
				if err != nil {
					return false, fmt.Errorf("failed to get PVC %s/%s: %w", volume.pvc.Namespace, volume.pvc.Name, err)
				}
				if pvc.Status.Phase != v1.ClaimPending {
					return true, nil
				}
			}
			return false, nil
		})
		if wait.Interrupted(waitErr) {
			framework.Logf("PVCs were not bound within %v (that's good)", bindTimeout)
			waitErr = nil
		}
		framework.ExpectNoError(waitErr, "Error making sure PVCs are not bound")
	}
}

func makeLocalPodWithNodeAffinity(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
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
	podConfig := e2epod.Config{
		NS:            config.ns,
		PVCs:          []*v1.PersistentVolumeClaim{volume.pvc},
		SeLinuxLabel:  selinuxLabel,
		NodeSelection: e2epod.NodeSelection{Affinity: affinity},
	}
	pod, err := e2epod.MakeSecPod(&podConfig)
	if pod == nil || err != nil {
		return
	}
	pod.Spec.Affinity = affinity
	return
}

func makeLocalPodWithNodeSelector(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	ns := map[string]string{
		"kubernetes.io/hostname": nodeName,
	}
	podConfig := e2epod.Config{
		NS:            config.ns,
		PVCs:          []*v1.PersistentVolumeClaim{volume.pvc},
		SeLinuxLabel:  selinuxLabel,
		NodeSelection: e2epod.NodeSelection{Selector: ns},
	}
	pod, err := e2epod.MakeSecPod(&podConfig)
	if pod == nil || err != nil {
		return
	}
	return
}

func makeLocalPodWithNodeName(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	podConfig := e2epod.Config{
		NS:           config.ns,
		PVCs:         []*v1.PersistentVolumeClaim{volume.pvc},
		SeLinuxLabel: selinuxLabel,
	}
	pod, err := e2epod.MakeSecPod(&podConfig)
	if pod == nil || err != nil {
		return
	}

	e2epod.SetNodeAffinity(&pod.Spec, nodeName)
	return
}

func createLocalPod(ctx context.Context, config *localTestConfig, volume *localTestVolume, fsGroup *int64) (*v1.Pod, error) {
	ginkgo.By("Creating a pod")
	podConfig := e2epod.Config{
		NS:           config.ns,
		PVCs:         []*v1.PersistentVolumeClaim{volume.pvc},
		SeLinuxLabel: selinuxLabel,
		FsGroup:      fsGroup,
	}
	return e2epod.CreateSecPod(ctx, config.client, &podConfig, config.timeouts.PodStart)
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
	}
	testFilePath := filepath.Join(testDir, testFile)
	return fmt.Sprintf("mkdir -p %s; echo %s > %s", testDir, writeTestFileContent, testFilePath)
}

func createReadCmd(testFileDir string, testFile string, volumeType localVolumeType) string {
	if volumeType == BlockLocalVolumeType {
		// Create the command to read the beginning of the block device and print it in ascii.
		return fmt.Sprintf("hexdump -n 100 -e '100 \"%%_p\"' %s | head -1", testFileDir)
	}
	// Create the command to read (aka cat) a file.
	testFilePath := filepath.Join(testFileDir, testFile)
	return fmt.Sprintf("cat %s", testFilePath)
}

// Read testFile and evaluate whether it contains the testFileContent
func testReadFileContent(f *framework.Framework, testFileDir string, testFile string, testFileContent string, pod *v1.Pod, volumeType localVolumeType) {
	readCmd := createReadCmd(testFileDir, testFile, volumeType)
	readOut := podRWCmdExec(f, pod, readCmd)
	gomega.Expect(readOut).To(gomega.ContainSubstring(testFileContent))
}

// Execute a read or write command in a pod.
// Fail on error
func podRWCmdExec(f *framework.Framework, pod *v1.Pod, cmd string) string {
	stdout, stderr, err := e2evolume.PodExec(f, pod, cmd)
	framework.Logf("podRWCmdExec cmd: %q, out: %q, stderr: %q, err: %v", cmd, stdout, stderr, err)
	framework.ExpectNoError(err)
	return stdout
}

// Initialize test volume on node
// and create local PVC and PV
func setupLocalVolumesPVCsPVs(
	ctx context.Context,
	config *localTestConfig,
	localVolumeType localVolumeType,
	node *v1.Node,
	count int,
	mode storagev1.VolumeBindingMode) []*localTestVolume {

	ginkgo.By("Initializing test volumes")
	testVols := setupLocalVolumes(ctx, config, localVolumeType, node, count)

	ginkgo.By("Creating local PVCs and PVs")
	createLocalPVCsPVs(ctx, config, testVols, mode)

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
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(testRequestSize),
				},
			},
		},
	}

	return &claim
}

func createStatefulSet(ctx context.Context, config *localTestConfig, ssReplicas int32, volumeCount int, anti, parallel bool) *appsv1.StatefulSet {
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

	ss, err := config.client.AppsV1().StatefulSets(config.ns).Create(ctx, spec, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	e2estatefulset.WaitForRunningAndReady(ctx, config.client, ssReplicas, ss)
	return ss
}

func validateStatefulSet(ctx context.Context, config *localTestConfig, ss *appsv1.StatefulSet, anti bool) {
	pods := e2estatefulset.GetPodList(ctx, config.client, ss)

	nodes := sets.NewString()
	for _, pod := range pods.Items {
		nodes.Insert(pod.Spec.NodeName)
	}

	if anti {
		// Verify that each pod is on a different node
		gomega.Expect(pods.Items).To(gomega.HaveLen(nodes.Len()))
	} else {
		// Verify that all pods are on same node.
		gomega.Expect(nodes.Len()).To(gomega.Equal(1))
	}

	// Validate all PVCs are bound
	for _, pod := range pods.Items {
		for _, volume := range pod.Spec.Volumes {
			pvcSource := volume.VolumeSource.PersistentVolumeClaim
			if pvcSource != nil {
				err := e2epv.WaitForPersistentVolumeClaimPhase(ctx,
					v1.ClaimBound, config.client, config.ns, pvcSource.ClaimName, framework.Poll, time.Second)
				framework.ExpectNoError(err)
			}
		}
	}
}

// SkipUnlessLocalSSDExists takes in an ssdInterface (scsi/nvme) and a filesystemType (fs/block)
// and skips if a disk of that type does not exist on the node
func SkipUnlessLocalSSDExists(ctx context.Context, config *localTestConfig, ssdInterface, filesystemType string, node *v1.Node) {
	ssdCmd := fmt.Sprintf("ls -1 /mnt/disks/by-uuid/google-local-ssds-%s-%s/ | wc -l", ssdInterface, filesystemType)
	res, err := config.hostExec.Execute(ctx, ssdCmd, node)
	utils.LogResult(res)
	framework.ExpectNoError(err)
	num, err := strconv.Atoi(strings.TrimSpace(res.Stdout))
	framework.ExpectNoError(err)
	if num < 1 {
		e2eskipper.Skipf("Requires at least 1 %s %s localSSD ", ssdInterface, filesystemType)
	}
}
