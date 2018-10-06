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
	"math/rand"
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
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"crypto/sha256"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	csiExternalProvisionerClusterRoleName string = "system:csi-external-provisioner"
	csiExternalAttacherClusterRoleName    string = "system:csi-external-attacher"
	csiDriverRegistrarClusterRoleName     string = "csi-driver-registrar"
)

type csiTestDriver interface {
	createCSIDriver()
	cleanupCSIDriver()
	createStorageClassTest(node v1.Node) storageClassTest
}

var csiTestDrivers = map[string]func(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver{
	"hostPath": initCSIHostpath,
	"gcePD":    initCSIgcePD,
}

var _ = utils.SIGDescribe("[Serial] CSI Volumes", func() {
	f := framework.NewDefaultFramework("csi-mock-plugin")

	var (
		cs        clientset.Interface
		crdclient apiextensionsclient.Interface
		csics     csiclient.Interface
		ns        *v1.Namespace
		node      v1.Node
		config    framework.VolumeTestConfig
	)

	BeforeEach(func() {
		cs = f.ClientSet
		crdclient = f.APIExtensionsClientSet
		csics = f.CSIClientSet
		ns = f.Namespace
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		node = nodes.Items[rand.Intn(len(nodes.Items))]
		config = framework.VolumeTestConfig{
			Namespace:         ns.Name,
			Prefix:            "csi",
			ClientNodeName:    node.Name,
			ServerNodeName:    node.Name,
			WaitForCompletion: true,
		}
		csiDriverRegistrarClusterRole(config)
		createCSICRDs(crdclient)
	})

	AfterEach(func() {
		deleteCSICRDs(crdclient)
	})

	for driverName, initCSIDriver := range csiTestDrivers {
		curDriverName := driverName
		curInitCSIDriver := initCSIDriver

		Context(fmt.Sprintf("CSI plugin test using CSI driver: %s [Serial]", curDriverName), func() {
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
				t := driver.createStorageClassTest(node)
				claim := newClaim(t, ns.GetName(), "")
				class := newStorageClass(t, ns.GetName(), "")
				claim.Spec.StorageClassName = &class.ObjectMeta.Name
				testDynamicProvisioning(t, cs, claim, class)
			})
		})
	}

	// Use [Serial], because there can be only one CSIDriver for csi-hostpath driver.
	Context("CSI attach test using HostPath driver [Serial][Feature:CSISkipAttach]", func() {
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
					driver := createCSIDriver(csics, test.driverAttachable)
					if driver != nil {
						defer csics.CsiV1alpha1().CSIDrivers().Delete(driver.Name, nil)
					}
				}

				By("Creating pod")
				t := driver.createStorageClassTest(node)
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
				attachmentHash := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", handle, t.provisioner, node.Name)))
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

func createCSIDriver(csics csiclient.Interface, attachable bool) *csiv1alpha1.CSIDriver {
	By("Creating CSIDriver instance")
	driver := &csiv1alpha1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "csi-hostpath",
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
	// re-get the claim to the the latest state with bound volume
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

func startPausePod(cs clientset.Interface, t storageClassTest, ns string) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
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

	if len(t.nodeName) != 0 {
		pod.Spec.NodeName = t.nodeName
	}
	pod, err = cs.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return class, claim, pod
}

type hostpathCSIDriver struct {
	combinedClusterRoleNames []string
	serviceAccount           *v1.ServiceAccount

	f      *framework.Framework
	config framework.VolumeTestConfig
}

func initCSIHostpath(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver {
	return &hostpathCSIDriver{
		combinedClusterRoleNames: []string{
			csiExternalAttacherClusterRoleName,
			csiExternalProvisionerClusterRoleName,
			csiDriverRegistrarClusterRoleName,
		},
		f:      f,
		config: config,
	}
}

func (h *hostpathCSIDriver) createStorageClassTest(node v1.Node) storageClassTest {
	return storageClassTest{
		name:         "csi-hostpath",
		provisioner:  "csi-hostpath",
		parameters:   map[string]string{},
		claimSize:    "1Gi",
		expectedSize: "1Gi",
		nodeName:     node.Name,
	}
}

func (h *hostpathCSIDriver) createCSIDriver() {
	By("deploying csi hostpath driver")
	f := h.f
	cs := f.ClientSet
	config := h.config
	h.serviceAccount = csiServiceAccount(cs, config, "hostpath", false)
	csiClusterRoleBindings(cs, config, false, h.serviceAccount, h.combinedClusterRoleNames)
	csiHostPathPod(cs, config, false, f, h.serviceAccount)
}

func (h *hostpathCSIDriver) cleanupCSIDriver() {
	By("uninstalling csi hostpath driver")
	f := h.f
	cs := f.ClientSet
	config := h.config
	csiHostPathPod(cs, config, true, f, h.serviceAccount)
	csiClusterRoleBindings(cs, config, true, h.serviceAccount, h.combinedClusterRoleNames)
	csiServiceAccount(cs, config, "hostpath", true)
}

type gcePDCSIDriver struct {
	controllerClusterRoles   []string
	nodeClusterRoles         []string
	controllerServiceAccount *v1.ServiceAccount
	nodeServiceAccount       *v1.ServiceAccount

	f      *framework.Framework
	config framework.VolumeTestConfig
}

func initCSIgcePD(f *framework.Framework, config framework.VolumeTestConfig) csiTestDriver {
	cs := f.ClientSet
	framework.SkipUnlessProviderIs("gce", "gke")
	framework.SkipIfMultizone(cs)

	// TODO(#62561): Use credentials through external pod identity when that goes GA instead of downloading keys.
	createGCESecrets(cs, config)

	framework.SkipUnlessSecretExistsAfterWait(cs, "cloud-sa", config.Namespace, 3*time.Minute)

	return &gcePDCSIDriver{
		nodeClusterRoles: []string{
			csiDriverRegistrarClusterRoleName,
		},
		controllerClusterRoles: []string{
			csiExternalAttacherClusterRoleName,
			csiExternalProvisionerClusterRoleName,
		},
		f:      f,
		config: config,
	}
}

func (g *gcePDCSIDriver) createStorageClassTest(node v1.Node) storageClassTest {
	return storageClassTest{
		name:         "com.google.csi.gcepd",
		provisioner:  "com.google.csi.gcepd",
		parameters:   map[string]string{"type": "pd-standard"},
		claimSize:    "5Gi",
		expectedSize: "5Gi",
		nodeName:     node.Name,
	}
}

func (g *gcePDCSIDriver) createCSIDriver() {
	By("deploying gce-pd driver")
	f := g.f
	cs := f.ClientSet
	config := g.config
	g.controllerServiceAccount = csiServiceAccount(cs, config, "gce-controller", false /* teardown */)
	g.nodeServiceAccount = csiServiceAccount(cs, config, "gce-node", false /* teardown */)
	csiClusterRoleBindings(cs, config, false /* teardown */, g.controllerServiceAccount, g.controllerClusterRoles)
	csiClusterRoleBindings(cs, config, false /* teardown */, g.nodeServiceAccount, g.nodeClusterRoles)
	utils.PrivilegedTestPSPClusterRoleBinding(cs, config.Namespace,
		false /* teardown */, []string{g.controllerServiceAccount.Name, g.nodeServiceAccount.Name})
	deployGCEPDCSIDriver(cs, config, false /* teardown */, f, g.nodeServiceAccount, g.controllerServiceAccount)
}

func (g *gcePDCSIDriver) cleanupCSIDriver() {
	By("uninstalling gce-pd driver")
	f := g.f
	cs := f.ClientSet
	config := g.config
	deployGCEPDCSIDriver(cs, config, true /* teardown */, f, g.nodeServiceAccount, g.controllerServiceAccount)
	csiClusterRoleBindings(cs, config, true /* teardown */, g.controllerServiceAccount, g.controllerClusterRoles)
	csiClusterRoleBindings(cs, config, true /* teardown */, g.nodeServiceAccount, g.nodeClusterRoles)
	utils.PrivilegedTestPSPClusterRoleBinding(cs, config.Namespace,
		true /* teardown */, []string{g.controllerServiceAccount.Name, g.nodeServiceAccount.Name})
	csiServiceAccount(cs, config, "gce-controller", true /* teardown */)
	csiServiceAccount(cs, config, "gce-node", true /* teardown */)
}
