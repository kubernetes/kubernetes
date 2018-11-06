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
	"fmt"
	"math/rand"
	"regexp"
	"time"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	csiclient "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/podlogs"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"crypto/sha256"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type csiTestDriver interface {
	createCSIDriver()
	cleanupCSIDriver()
	createStorageClassTest() testsuites.StorageClassTest
}

var csiTestDrivers = map[string]func(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver{
	"hostPath": initCSIHostpath,
	"gcePD":    initCSIgcePD,
	// TODO(#70258): this is temporary until we can figure out how to make e2e tests a library
	"[Feature: gcePD-external]": initCSIgcePDExternal,
}

var _ = utils.SIGDescribe("CSI Volumes", func() {
	f := framework.NewDefaultFramework("csi-volumes")

	var (
		cancel    context.CancelFunc
		cs        clientset.Interface
		crdclient apiextensionsclient.Interface
		csics     csiclient.Interface
		ns        *v1.Namespace
		node      v1.Node
		config    framework.VolumeTestConfig
	)

	BeforeEach(func() {
		ctx, c := context.WithCancel(context.Background())
		cancel = c

		cs = f.ClientSet
		crdclient = f.APIExtensionsClientSet
		csics = f.CSIClientSet
		ns = f.Namespace

		// Debugging of the following tests heavily depends on the log output
		// of the different containers. Therefore include all of that in log
		// files (when using --report-dir, as in the CI) or the output stream
		// (otherwise).
		to := podlogs.LogOutput{
			StatusWriter: GinkgoWriter,
		}
		if framework.TestContext.ReportDir == "" {
			to.LogWriter = GinkgoWriter
		} else {
			test := CurrentGinkgoTestDescription()
			reg := regexp.MustCompile("[^a-zA-Z0-9_-]+")
			// We end the prefix with a slash to ensure that all logs
			// end up in a directory named after the current test.
			to.LogPathPrefix = framework.TestContext.ReportDir + "/" +
				reg.ReplaceAllString(test.FullTestText, "_") + "/"
		}
		podlogs.CopyAllLogs(ctx, cs, ns.Name, to)

		// pod events are something that the framework already collects itself
		// after a failed test. Logging them live is only useful for interactive
		// debugging, not when we collect reports.
		if framework.TestContext.ReportDir == "" {
			podlogs.WatchPods(ctx, cs, ns.Name, GinkgoWriter)
		}

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		node = nodes.Items[rand.Intn(len(nodes.Items))]
		config = framework.VolumeTestConfig{
			Namespace: ns.Name,
			Prefix:    "csi",
			// TODO(#70259): this needs to be parameterized so only hostpath sets node name
			ClientNodeName:    node.Name,
			ServerNodeName:    node.Name,
			WaitForCompletion: true,
		}
		createCSICRDs(crdclient)
	})

	AfterEach(func() {
		cancel()
	})

	for driverName, initCSIDriver := range csiTestDrivers {
		curDriverName := driverName
		curInitCSIDriver := initCSIDriver

		Context(fmt.Sprintf("CSI plugin test using CSI driver: %s", curDriverName), func() {
			var (
				driver csiTestDriver
			)

			BeforeEach(func() {
				driver = curInitCSIDriver(f, config)
				driver.createCSIDriver()
			})

			AfterEach(func() {
				driver.cleanupCSIDriver()
			})

			It("should provision storage", func() {
				t := driver.createStorageClassTest()
				claim := newClaim(t, ns.GetName(), "")
				var class *storagev1.StorageClass
				if t.StorageClassName == "" {
					class = newStorageClass(t, ns.GetName(), "")
					claim.Spec.StorageClassName = &class.ObjectMeta.Name
				} else {
					scName := t.StorageClassName
					claim.Spec.StorageClassName = &scName
				}
				testsuites.TestDynamicProvisioning(t, cs, claim, class)
			})
		})
	}

	// The CSIDriverRegistry feature gate is needed for this test in Kubernetes 1.12.
	Context("CSI attach test using HostPath driver [Feature:CSISkipAttach]", func() {
		var (
			driver csiTestDriver
		)
		BeforeEach(func() {
			driver = initCSIHostpath(f, config)
			driver.createCSIDriver()
		})

		AfterEach(func() {
			driver.cleanupCSIDriver()
		})

		tests := []struct {
			name                   string
			driverAttachable       bool
			driverExists           bool
			expectVolumeAttachment bool
		}{
			{
				name:                   "non-attachable volume does not need VolumeAttachment",
				driverAttachable:       false,
				driverExists:           true,
				expectVolumeAttachment: false,
			},
			{
				name:                   "attachable volume needs VolumeAttachment",
				driverAttachable:       true,
				driverExists:           true,
				expectVolumeAttachment: true,
			},
			{
				name:                   "volume with no CSI driver needs VolumeAttachment",
				driverExists:           false,
				expectVolumeAttachment: true,
			},
		}

		for _, t := range tests {
			test := t
			It(test.name, func() {
				if test.driverExists {
					driver := createCSIDriver(csics, "csi-hostpath-"+f.UniqueName, test.driverAttachable)
					if driver != nil {
						defer csics.CsiV1alpha1().CSIDrivers().Delete(driver.Name, nil)
					}
				}

				By("Creating pod")
				t := driver.createStorageClassTest()
				class, claim, pod := startPausePod(cs, t, ns.Name)
				if class != nil {
					defer cs.StorageV1().StorageClasses().Delete(class.Name, nil)
				}
				if claim != nil {
					defer cs.CoreV1().PersistentVolumeClaims(ns.Name).Delete(claim.Name, nil)
				}
				if pod != nil {
					// Fully delete (=unmount) the pod before deleting CSI driver
					defer framework.DeletePodWithWait(f, cs, pod)
				}
				if pod == nil {
					return
				}

				err := framework.WaitForPodNameRunningInNamespace(cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				By("Checking if VolumeAttachment was created for the pod")
				// Check that VolumeAttachment does not exist
				handle := getVolumeHandle(cs, claim)
				attachmentHash := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", handle, t.Provisioner, node.Name)))
				attachmentName := fmt.Sprintf("csi-%x", attachmentHash)
				_, err = cs.StorageV1beta1().VolumeAttachments().Get(attachmentName, metav1.GetOptions{})
				if err != nil {
					if errors.IsNotFound(err) {
						if test.expectVolumeAttachment {
							framework.ExpectNoError(err, "Expected VolumeAttachment but none was found")
						}
					} else {
						framework.ExpectNoError(err, "Failed to find VolumeAttachment")
					}
				}
				if !test.expectVolumeAttachment {
					Expect(err).To(HaveOccurred(), "Unexpected VolumeAttachment found")
				}
			})
		}
	})
})

func createCSIDriver(csics csiclient.Interface, name string, attachable bool) *csiv1alpha1.CSIDriver {
	By("Creating CSIDriver instance")
	driver := &csiv1alpha1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: csiv1alpha1.CSIDriverSpec{
			AttachRequired: &attachable,
		},
	}
	driver, err := csics.CsiV1alpha1().CSIDrivers().Create(driver)
	framework.ExpectNoError(err, "Failed to create CSIDriver: %v", err)
	return driver
}

func getVolumeHandle(cs clientset.Interface, claim *v1.PersistentVolumeClaim) string {
	// re-get the claim to the latest state with bound volume
	claim, err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PVC")
		return ""
	}
	pvName := claim.Spec.VolumeName
	pv, err := cs.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PV")
		return ""
	}
	if pv.Spec.CSI == nil {
		Expect(pv.Spec.CSI).NotTo(BeNil())
		return ""
	}
	return pv.Spec.CSI.VolumeHandle
}

func startPausePod(cs clientset.Interface, t testsuites.StorageClassTest, ns string) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
	class := newStorageClass(t, ns, "")
	class, err := cs.StorageV1().StorageClasses().Create(class)
	framework.ExpectNoError(err, "Failed to create class : %v", err)
	claim := newClaim(t, ns, "")
	claim.Spec.StorageClassName = &class.Name
	claim, err = cs.CoreV1().PersistentVolumeClaims(ns).Create(claim)
	framework.ExpectNoError(err, "Failed to create claim: %v", err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "volume-tester",
					Image: imageutils.GetE2EImage(imageutils.Pause),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: claim.Name,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}

	if len(t.NodeName) != 0 {
		pod.Spec.NodeName = t.NodeName
	}
	pod, err = cs.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return class, claim, pod
}

type hostpathCSIDriver struct {
	f       *framework.Framework
	config  framework.VolumeTestConfig
	cleanup func()
}

func initCSIHostpath(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver {
	return &hostpathCSIDriver{
		f:      f,
		config: config,
	}
}

func (h *hostpathCSIDriver) createStorageClassTest() testsuites.StorageClassTest {
	return testsuites.StorageClassTest{
		Name:         "csi-hostpath",
		Parameters:   map[string]string{},
		ClaimSize:    "1Gi",
		ExpectedSize: "1Gi",

		// The hostpath driver only works when everything runs on a single node.
		NodeName: h.config.ServerNodeName,

		// Provisioner and storage class name must match what's used in
		// csi-storageclass.yaml, plus the test-specific suffix.
		Provisioner:      "csi-hostpath-" + h.f.UniqueName,
		StorageClassName: "csi-hostpath-sc-" + h.f.UniqueName,
	}
}

func (h *hostpathCSIDriver) createCSIDriver() {
	By("deploying csi hostpath driver")
	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:            "csi-hostpath",
		NewDriverName:            "csi-hostpath-" + h.f.UniqueName,
		DriverContainerName:      "hostpath",
		ProvisionerContainerName: "csi-provisioner",
		NodeName:                 h.config.ServerNodeName,
	}
	cleanup, err := h.f.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(h.f, o, item)
	},
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-attacher.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-provisioner.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpathplugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/e2e-test-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/usage/csi-storageclass.yaml",
	)
	h.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying csi hostpath driver: %v", err)
	}
}

func (h *hostpathCSIDriver) cleanupCSIDriver() {
	if h.cleanup != nil {
		By("uninstalling csi hostpath driver")
		h.cleanup()
	}
}

type gcePDCSIDriver struct {
	f       *framework.Framework
	config  framework.VolumeTestConfig
	cleanup func()
}

func initCSIgcePD(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver {
	cs := f.ClientSet
	framework.SkipUnlessProviderIs("gce", "gke")
	framework.SkipIfMultizone(cs)

	// TODO(#62561): Use credentials through external pod identity when that goes GA instead of downloading keys.
	createGCESecrets(cs, config)

	framework.SkipUnlessSecretExistsAfterWait(cs, "cloud-sa", config.Namespace, 3*time.Minute)

	return &gcePDCSIDriver{
		f:      f,
		config: config,
	}
}

func (g *gcePDCSIDriver) createStorageClassTest() testsuites.StorageClassTest {
	return testsuites.StorageClassTest{
		Name: "com.google.csi.gcepd",
		// *Not* renaming the driver, see below.
		Provisioner:  "com.google.csi.gcepd",
		Parameters:   map[string]string{"type": "pd-standard"},
		ClaimSize:    "5Gi",
		ExpectedSize: "5Gi",
	}
}

func (g *gcePDCSIDriver) createCSIDriver() {
	By("deploying gce-pd driver")
	// It would be safer to rename the gcePD driver, but that
	// hasn't been done before either and attempts to do so now led to
	// errors during driver registration, therefore it is disabled
	// by passing a nil function below.
	//
	// These are the options which would have to be used:
	// o := utils.PatchCSIOptions{
	// 	OldDriverName:            "com.google.csi.gcepd",
	// 	NewDriverName:            "com.google.csi.gcepd-" + g.f.UniqueName,
	// 	DriverContainerName:      "gce-driver",
	// 	ProvisionerContainerName: "csi-external-provisioner",
	// }
	cleanup, err := g.f.CreateFromManifests(nil,
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/csi-controller-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/node_ds.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss.yaml",
	)
	g.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying csi hostpath driver: %v", err)
	}
}

func (g *gcePDCSIDriver) cleanupCSIDriver() {
	By("uninstalling gce-pd driver")
	if g.cleanup != nil {
		g.cleanup()
	}
}

type gcePDCSIDriverExternal struct {
}

func initCSIgcePDExternal(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver {
	cs := f.ClientSet
	framework.SkipUnlessProviderIs("gce", "gke")
	framework.SkipIfMultizone(cs)

	return &gcePDCSIDriverExternal{}
}

func (g *gcePDCSIDriverExternal) createStorageClassTest() testsuites.StorageClassTest {
	return testsuites.StorageClassTest{
		Name:         "com.google.csi.gcepd",
		Provisioner:  "com.google.csi.gcepd",
		Parameters:   map[string]string{"type": "pd-standard"},
		ClaimSize:    "5Gi",
		ExpectedSize: "5Gi",
	}
}

func (g *gcePDCSIDriverExternal) createCSIDriver() {
}

func (g *gcePDCSIDriverExternal) cleanupCSIDriver() {
}
