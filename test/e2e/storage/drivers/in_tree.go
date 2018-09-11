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

/*
 * This file defines various in-tree volume test drivers for TestSuites.
 *
 * There are two ways, how to prepare test drivers:
 * 1) With containerized server (NFS, Ceph, Gluster, iSCSI, ...)
 * It creates a server pod which defines one volume for the tests.
 * These tests work only when privileged containers are allowed, exporting
 * various filesystems (NFS, GlusterFS, ...) usually needs some mounting or
 * other privileged magic in the server pod.
 *
 * Note that the server containers are for testing purposes only and should not
 * be used in production.
 *
 * 2) With server or cloud provider outside of Kubernetes (Cinder, GCE, AWS, Azure, ...)
 * Appropriate server or cloud provider must exist somewhere outside
 * the tested Kubernetes cluster. CreateVolume will create a new volume to be
 * used in the TestSuites for inlineVolume or DynamicPV tests.
 */

package drivers

import (
	"fmt"
	"math/rand"
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	vspheretest "k8s.io/kubernetes/test/e2e/storage/vsphere"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// NFS
type nfsDriver struct {
	serverIP               string
	serverPod              *v1.Pod
	externalProvisionerPod *v1.Pod
	externalPluginName     string

	driverInfo DriverInfo
}

var _ TestDriver = &nfsDriver{}
var _ PreprovisionedVolumeTestDriver = &nfsDriver{}
var _ InlineVolumeTestDriver = &nfsDriver{}
var _ PreprovisionedPVTestDriver = &nfsDriver{}
var _ DynamicPVTestDriver = &nfsDriver{}

// InitNFSDriver returns nfsDriver that implements TestDriver interface
func InitNFSDriver() TestDriver {
	return &nfsDriver{
		driverInfo: DriverInfo{
			Name:        "nfs",
			MaxFileSize: testpatterns.FileSizeLarge,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			IsPersistent:       true,
			IsFsGroupSupported: false,
			IsBlockSupported:   false,
		},
	}
}

func (n *nfsDriver) GetDriverInfo() *DriverInfo {
	return &n.driverInfo
}

func (n *nfsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (n *nfsDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	return &v1.VolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   n.serverIP,
			Path:     "/",
			ReadOnly: readOnly,
		},
	}
}

func (n *nfsDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	return &v1.PersistentVolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   n.serverIP,
			Path:     "/",
			ReadOnly: readOnly,
		},
	}
}

func (n *nfsDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := n.externalPluginName
	parameters := map[string]string{"mountOptions": "vers=4.1"}
	ns := n.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", n.driverInfo.Name)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (n *nfsDriver) CreateDriver() {
	f := n.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace
	n.externalPluginName = fmt.Sprintf("example.com/nfs-%s", ns.Name)

	// TODO(mkimuram): cluster-admin gives too much right but system:persistent-volume-provisioner
	// is not enough. We should create new clusterrole for testing.
	framework.BindClusterRole(cs.RbacV1beta1(), "cluster-admin", ns.Name,
		rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: ns.Name, Name: "default"})

	err := framework.WaitForAuthorizationUpdate(cs.AuthorizationV1beta1(),
		serviceaccount.MakeUsername(ns.Name, "default"),
		"", "get", schema.GroupResource{Group: "storage.k8s.io", Resource: "storageclasses"}, true)
	framework.ExpectNoError(err, "Failed to update authorization: %v", err)

	By("creating an external dynamic provisioner pod")
	n.externalProvisionerPod = utils.StartExternalProvisioner(cs, ns.Name, n.externalPluginName)
}

func (n *nfsDriver) CleanupDriver() {
	f := n.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	framework.ExpectNoError(framework.DeletePodWithWait(f, cs, n.externalProvisionerPod))
	clusterRoleBindingName := ns.Name + "--" + "cluster-admin"
	cs.RbacV1beta1().ClusterRoleBindings().Delete(clusterRoleBindingName, metav1.NewDeleteOptions(0))
}

func (n *nfsDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := n.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	// NewNFSServer creates a pod for InlineVolume and PreprovisionedPV,
	// and startExternalProvisioner creates a pods for DynamicPV.
	// Therefore, we need a different CreateDriver logic for volType.
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		n.driverInfo.Config, n.serverPod, n.serverIP = framework.NewNFSServer(cs, ns.Name, []string{})
	case testpatterns.DynamicPV:
		// Do nothing
	default:
		framework.Failf("Unsupported volType:%v is specified", volType)
	}
}

func (n *nfsDriver) DeleteVolume(volType testpatterns.TestVolType) {
	f := n.driverInfo.Framework

	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		framework.CleanUpVolumeServer(f, n.serverPod)
	case testpatterns.DynamicPV:
		// Do nothing
	default:
		framework.Failf("Unsupported volType:%v is specified", volType)
	}
}

// Gluster
type glusterFSDriver struct {
	serverIP  string
	serverPod *v1.Pod

	driverInfo DriverInfo
}

var _ TestDriver = &glusterFSDriver{}
var _ PreprovisionedVolumeTestDriver = &glusterFSDriver{}
var _ InlineVolumeTestDriver = &glusterFSDriver{}
var _ PreprovisionedPVTestDriver = &glusterFSDriver{}

// InitGlusterFSDriver returns glusterFSDriver that implements TestDriver interface
func InitGlusterFSDriver() TestDriver {
	return &glusterFSDriver{
		driverInfo: DriverInfo{
			Name:        "gluster",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			IsPersistent:       true,
			IsFsGroupSupported: false,
			IsBlockSupported:   false,
		},
	}
}

func (g *glusterFSDriver) GetDriverInfo() *DriverInfo {
	return &g.driverInfo
}

func (g *glusterFSDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessNodeOSDistroIs("gci", "ubuntu", "custom")
	if pattern.FsType == "xfs" {
		framework.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
}

func (g *glusterFSDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	name := g.driverInfo.Config.Prefix + "-server"
	return &v1.VolumeSource{
		Glusterfs: &v1.GlusterfsVolumeSource{
			EndpointsName: name,
			// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
			Path:     "test_vol",
			ReadOnly: readOnly,
		},
	}
}

func (g *glusterFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	name := g.driverInfo.Config.Prefix + "-server"
	return &v1.PersistentVolumeSource{
		Glusterfs: &v1.GlusterfsVolumeSource{
			EndpointsName: name,
			// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
			Path:     "test_vol",
			ReadOnly: readOnly,
		},
	}
}

func (g *glusterFSDriver) CreateDriver() {
}

func (g *glusterFSDriver) CleanupDriver() {
}

func (g *glusterFSDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := g.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	g.driverInfo.Config, g.serverPod, g.serverIP = framework.NewGlusterfsServer(cs, ns.Name)
}

func (g *glusterFSDriver) DeleteVolume(volType testpatterns.TestVolType) {
	f := g.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	name := g.driverInfo.Config.Prefix + "-server"

	framework.Logf("Deleting Gluster endpoints %q...", name)
	err := cs.CoreV1().Endpoints(ns.Name).Delete(name, nil)
	if err != nil {
		if !errors.IsNotFound(err) {
			framework.Failf("Gluster delete endpoints failed: %v", err)
		}
		framework.Logf("Gluster endpoints %q not found, assuming deleted", name)
	}
	framework.Logf("Deleting Gluster server pod %q...", g.serverPod.Name)
	err = framework.DeletePodWithWait(f, cs, g.serverPod)
	if err != nil {
		framework.Failf("Gluster server pod delete failed: %v", err)
	}
}

// iSCSI
// The iscsiadm utility and iscsi target kernel modules must be installed on all nodes.
type iSCSIDriver struct {
	serverIP  string
	serverPod *v1.Pod

	driverInfo DriverInfo
}

var _ TestDriver = &iSCSIDriver{}
var _ PreprovisionedVolumeTestDriver = &iSCSIDriver{}
var _ InlineVolumeTestDriver = &iSCSIDriver{}
var _ PreprovisionedPVTestDriver = &iSCSIDriver{}

// InitISCSIDriver returns iSCSIDriver that implements TestDriver interface
func InitISCSIDriver() TestDriver {
	return &iSCSIDriver{
		driverInfo: DriverInfo{
			Name:        "iscsi",
			FeatureTag:  "[Feature:Volumes]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				// TODO: fix iSCSI driver can work with ext3
				//"ext3",
				"ext4",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   true,
		},
	}
}

func (i *iSCSIDriver) GetDriverInfo() *DriverInfo {
	return &i.driverInfo
}

func (i *iSCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (i *iSCSIDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	volSource := v1.VolumeSource{
		ISCSI: &v1.ISCSIVolumeSource{
			TargetPortal: i.serverIP + ":3260",
			// from test/images/volume/iscsi/initiatorname.iscsi
			IQN:      "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
			Lun:      0,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.ISCSI.FSType = fsType
	}
	return &volSource
}

func (i *iSCSIDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	pvSource := v1.PersistentVolumeSource{
		ISCSI: &v1.ISCSIPersistentVolumeSource{
			TargetPortal: i.serverIP + ":3260",
			IQN:          "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
			Lun:          0,
			ReadOnly:     readOnly,
		},
	}
	if fsType != "" {
		pvSource.ISCSI.FSType = fsType
	}
	return &pvSource
}

func (i *iSCSIDriver) CreateDriver() {
}

func (i *iSCSIDriver) CleanupDriver() {
}

func (i *iSCSIDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := i.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	i.driverInfo.Config, i.serverPod, i.serverIP = framework.NewISCSIServer(cs, ns.Name)
}

func (i *iSCSIDriver) DeleteVolume(volType testpatterns.TestVolType) {
	f := i.driverInfo.Framework

	framework.CleanUpVolumeServer(f, i.serverPod)
}

// Ceph RBD
type rbdDriver struct {
	serverIP  string
	serverPod *v1.Pod
	secret    *v1.Secret

	driverInfo DriverInfo
}

var _ TestDriver = &rbdDriver{}
var _ PreprovisionedVolumeTestDriver = &rbdDriver{}
var _ InlineVolumeTestDriver = &rbdDriver{}
var _ PreprovisionedPVTestDriver = &rbdDriver{}

// InitRbdDriver returns rbdDriver that implements TestDriver interface
func InitRbdDriver() TestDriver {
	return &rbdDriver{
		driverInfo: DriverInfo{
			Name:        "rbd",
			FeatureTag:  "[Feature:Volumes]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				// TODO: fix rbd driver can work with ext3
				//"ext3",
				"ext4",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   true},
	}
}

func (r *rbdDriver) GetDriverInfo() *DriverInfo {
	return &r.driverInfo
}

func (r *rbdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (r *rbdDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	volSource := v1.VolumeSource{
		RBD: &v1.RBDVolumeSource{
			CephMonitors: []string{r.serverIP},
			RBDPool:      "rbd",
			RBDImage:     "foo",
			RadosUser:    "admin",
			SecretRef: &v1.LocalObjectReference{
				Name: r.secret.Name,
			},
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.RBD.FSType = fsType
	}
	return &volSource
}

func (r *rbdDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	f := r.driverInfo.Framework
	ns := f.Namespace
	pvSource := v1.PersistentVolumeSource{
		RBD: &v1.RBDPersistentVolumeSource{
			CephMonitors: []string{r.serverIP},
			RBDPool:      "rbd",
			RBDImage:     "foo",
			RadosUser:    "admin",
			SecretRef: &v1.SecretReference{
				Name:      r.secret.Name,
				Namespace: ns.Name,
			},
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.RBD.FSType = fsType
	}
	return &pvSource
}

func (r *rbdDriver) CreateDriver() {
}

func (r *rbdDriver) CleanupDriver() {
}

func (r *rbdDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := r.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	r.driverInfo.Config, r.serverPod, r.secret, r.serverIP = framework.NewRBDServer(cs, ns.Name)
}

func (r *rbdDriver) DeleteVolume(volType testpatterns.TestVolType) {
	f := r.driverInfo.Framework

	framework.CleanUpVolumeServerWithSecret(f, r.serverPod, r.secret)
}

// Ceph
type cephFSDriver struct {
	serverIP  string
	serverPod *v1.Pod
	secret    *v1.Secret

	driverInfo DriverInfo
}

var _ TestDriver = &cephFSDriver{}
var _ PreprovisionedVolumeTestDriver = &cephFSDriver{}
var _ InlineVolumeTestDriver = &cephFSDriver{}
var _ PreprovisionedPVTestDriver = &cephFSDriver{}

// InitCephFSDriver returns cephFSDriver that implements TestDriver interface
func InitCephFSDriver() TestDriver {
	return &cephFSDriver{
		driverInfo: DriverInfo{
			Name:        "ceph",
			FeatureTag:  "[Feature:Volumes]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			IsPersistent:       true,
			IsFsGroupSupported: false,
			IsBlockSupported:   false,
		},
	}
}

func (c *cephFSDriver) GetDriverInfo() *DriverInfo {
	return &c.driverInfo
}

func (c *cephFSDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (c *cephFSDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	return &v1.VolumeSource{
		CephFS: &v1.CephFSVolumeSource{
			Monitors: []string{c.serverIP + ":6789"},
			User:     "kube",
			SecretRef: &v1.LocalObjectReference{
				Name: c.secret.Name,
			},
			ReadOnly: readOnly,
		},
	}
}

func (c *cephFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	f := c.driverInfo.Framework
	ns := f.Namespace

	return &v1.PersistentVolumeSource{
		CephFS: &v1.CephFSPersistentVolumeSource{
			Monitors: []string{c.serverIP + ":6789"},
			User:     "kube",
			SecretRef: &v1.SecretReference{
				Name:      c.secret.Name,
				Namespace: ns.Name,
			},
			ReadOnly: readOnly,
		},
	}
}

func (c *cephFSDriver) CreateDriver() {
}

func (c *cephFSDriver) CleanupDriver() {
}

func (c *cephFSDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := c.driverInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace

	c.driverInfo.Config, c.serverPod, c.secret, c.serverIP = framework.NewRBDServer(cs, ns.Name)
}

func (c *cephFSDriver) DeleteVolume(volType testpatterns.TestVolType) {
	f := c.driverInfo.Framework

	framework.CleanUpVolumeServerWithSecret(f, c.serverPod, c.secret)
}

// Hostpath
type hostPathDriver struct {
	node v1.Node

	driverInfo DriverInfo
}

var _ TestDriver = &hostPathDriver{}
var _ PreprovisionedVolumeTestDriver = &hostPathDriver{}
var _ InlineVolumeTestDriver = &hostPathDriver{}

// InitHostpathDriver returns hostPathDriver that implements TestDriver interface
func InitHostPathDriver() TestDriver {
	return &hostPathDriver{
		driverInfo: DriverInfo{
			Name:        "hostPath",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			IsPersistent:       true,
			IsFsGroupSupported: false,
			IsBlockSupported:   false,
		},
	}
}

func (h *hostPathDriver) GetDriverInfo() *DriverInfo {
	return &h.driverInfo
}

func (h *hostPathDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostPathDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	// hostPath doesn't support readOnly volume
	if readOnly {
		return nil
	}
	return &v1.VolumeSource{
		HostPath: &v1.HostPathVolumeSource{
			Path: "/tmp",
		},
	}
}

func (h *hostPathDriver) CreateDriver() {
}

func (h *hostPathDriver) CleanupDriver() {
}

func (h *hostPathDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := h.driverInfo.Framework
	cs := f.ClientSet

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	h.driverInfo.Config.ClientNodeName = node.Name
}

func (h *hostPathDriver) DeleteVolume(volType testpatterns.TestVolType) {
}

// HostPathSymlink
type hostPathSymlinkDriver struct {
	node       v1.Node
	sourcePath string
	targetPath string
	prepPod    *v1.Pod

	driverInfo DriverInfo
}

var _ TestDriver = &hostPathSymlinkDriver{}
var _ PreprovisionedVolumeTestDriver = &hostPathSymlinkDriver{}
var _ InlineVolumeTestDriver = &hostPathSymlinkDriver{}

// InitHostPathSymlinkDriver returns hostPathSymlinkDriver that implements TestDriver interface
func InitHostPathSymlinkDriver() TestDriver {
	return &hostPathSymlinkDriver{
		driverInfo: DriverInfo{
			Name:        "hostPathSymlink",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			IsPersistent:       true,
			IsFsGroupSupported: false,
			IsBlockSupported:   false,
		},
	}
}

func (h *hostPathSymlinkDriver) GetDriverInfo() *DriverInfo {
	return &h.driverInfo
}

func (h *hostPathSymlinkDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostPathSymlinkDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	// hostPathSymlink doesn't support readOnly volume
	if readOnly {
		return nil
	}
	return &v1.VolumeSource{
		HostPath: &v1.HostPathVolumeSource{
			Path: h.targetPath,
		},
	}
}

func (h *hostPathSymlinkDriver) CreateDriver() {
}

func (h *hostPathSymlinkDriver) CleanupDriver() {
}

func (h *hostPathSymlinkDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := h.driverInfo.Framework
	cs := f.ClientSet

	h.sourcePath = fmt.Sprintf("/tmp/%v", f.Namespace.Name)
	h.targetPath = fmt.Sprintf("/tmp/%v-link", f.Namespace.Name)
	volumeName := "test-volume"

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	h.driverInfo.Config.ClientNodeName = node.Name

	cmd := fmt.Sprintf("mkdir %v -m 777 && ln -s %v %v", h.sourcePath, h.sourcePath, h.targetPath)
	privileged := true

	// Launch pod to initialize hostPath directory and symlink
	h.prepPod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("hostpath-symlink-prep-%s", f.Namespace.Name),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("init-volume-%s", f.Namespace.Name),
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh", "-ec", cmd},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: "/tmp",
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/tmp",
						},
					},
				},
			},
			NodeName: node.Name,
		},
	}
	// h.prepPod will be reused in cleanupDriver.
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(h.prepPod)
	Expect(err).ToNot(HaveOccurred(), "while creating hostPath init pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for hostPath init pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting hostPath init pod")
}

func (h *hostPathSymlinkDriver) DeleteVolume(volType testpatterns.TestVolType) {
	f := h.driverInfo.Framework

	cmd := fmt.Sprintf("rm -rf %v&& rm -rf %v", h.targetPath, h.sourcePath)
	h.prepPod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", cmd}

	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(h.prepPod)
	Expect(err).ToNot(HaveOccurred(), "while creating hostPath teardown pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for hostPath teardown pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting hostPath teardown pod")
}

// emptydir
type emptydirDriver struct {
	driverInfo DriverInfo
}

var _ TestDriver = &emptydirDriver{}
var _ PreprovisionedVolumeTestDriver = &emptydirDriver{}
var _ InlineVolumeTestDriver = &emptydirDriver{}

// InitEmptydirDriver returns emptydirDriver that implements TestDriver interface
func InitEmptydirDriver() TestDriver {
	return &emptydirDriver{
		driverInfo: DriverInfo{
			Name:        "emptydir",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			IsPersistent:       false,
			IsFsGroupSupported: false,
			IsBlockSupported:   false,
		},
	}
}

func (e *emptydirDriver) GetDriverInfo() *DriverInfo {
	return &e.driverInfo
}

func (e *emptydirDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (e *emptydirDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	// emptydir doesn't support readOnly volume
	if readOnly {
		return nil
	}
	return &v1.VolumeSource{
		EmptyDir: &v1.EmptyDirVolumeSource{},
	}
}

func (e *emptydirDriver) CreateVolume(volType testpatterns.TestVolType) {
}

func (e *emptydirDriver) DeleteVolume(volType testpatterns.TestVolType) {
}

func (e *emptydirDriver) CreateDriver() {
}

func (e *emptydirDriver) CleanupDriver() {
}

// Cinder
// This driver assumes that OpenStack client tools are installed
// (/usr/bin/nova, /usr/bin/cinder and /usr/bin/keystone)
// and that the usual OpenStack authentication env. variables are set
// (OS_USERNAME, OS_PASSWORD, OS_TENANT_NAME at least).
type cinderDriver struct {
	volumeName string
	volumeID   string

	driverInfo DriverInfo
}

var _ TestDriver = &cinderDriver{}
var _ PreprovisionedVolumeTestDriver = &cinderDriver{}
var _ InlineVolumeTestDriver = &cinderDriver{}
var _ PreprovisionedPVTestDriver = &cinderDriver{}
var _ DynamicPVTestDriver = &cinderDriver{}

// InitCinderDriver returns cinderDriver that implements TestDriver interface
func InitCinderDriver() TestDriver {
	return &cinderDriver{
		driverInfo: DriverInfo{
			Name:        "cinder",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   false,
		},
	}
}

func (c *cinderDriver) GetDriverInfo() *DriverInfo {
	return &c.driverInfo
}

func (c *cinderDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("openstack")
}

func (c *cinderDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	volSource := v1.VolumeSource{
		Cinder: &v1.CinderVolumeSource{
			VolumeID: c.volumeID,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.Cinder.FSType = fsType
	}
	return &volSource
}

func (c *cinderDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	pvSource := v1.PersistentVolumeSource{
		Cinder: &v1.CinderPersistentVolumeSource{
			VolumeID: c.volumeID,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.Cinder.FSType = fsType
	}
	return &pvSource
}

func (c *cinderDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/cinder"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := c.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", c.driverInfo.Name)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (c *cinderDriver) CreateDriver() {
}

func (c *cinderDriver) CleanupDriver() {
}

func (c *cinderDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := c.driverInfo.Framework
	ns := f.Namespace

	// We assume that namespace.Name is a random string
	c.volumeName = ns.Name
	By("creating a test Cinder volume")
	output, err := exec.Command("cinder", "create", "--display-name="+c.volumeName, "1").CombinedOutput()
	outputString := string(output[:])
	framework.Logf("cinder output:\n%s", outputString)
	Expect(err).NotTo(HaveOccurred())

	// Parse 'id'' from stdout. Expected format:
	// |     attachments     |                  []                  |
	// |  availability_zone  |                 nova                 |
	// ...
	// |          id         | 1d6ff08f-5d1c-41a4-ad72-4ef872cae685 |
	c.volumeID = ""
	for _, line := range strings.Split(outputString, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 5 {
			continue
		}
		if fields[1] != "id" {
			continue
		}
		c.volumeID = fields[3]
		break
	}
	framework.Logf("Volume ID: %s", c.volumeID)
	Expect(c.volumeID).NotTo(Equal(""))
}

func (c *cinderDriver) DeleteVolume(volType testpatterns.TestVolType) {
	deleteCinderVolume(c.volumeName)
}

func deleteCinderVolume(name string) error {
	// Try to delete the volume for several seconds - it takes
	// a while for the plugin to detach it.
	var output []byte
	var err error
	timeout := time.Second * 120

	framework.Logf("Waiting up to %v for removal of cinder volume %s", timeout, name)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		output, err = exec.Command("cinder", "delete", name).CombinedOutput()
		if err == nil {
			framework.Logf("Cinder volume %s deleted", name)
			return nil
		}
		framework.Logf("Failed to delete volume %s: %v", name, err)
	}
	framework.Logf("Giving up deleting volume %s: %v\n%s", name, err, string(output[:]))
	return err
}

// GCE
type gcePdDriver struct {
	volumeName string

	driverInfo DriverInfo
}

var _ TestDriver = &gcePdDriver{}
var _ PreprovisionedVolumeTestDriver = &gcePdDriver{}
var _ InlineVolumeTestDriver = &gcePdDriver{}
var _ PreprovisionedPVTestDriver = &gcePdDriver{}
var _ DynamicPVTestDriver = &gcePdDriver{}

// InitGceDriver returns gcePdDriver that implements TestDriver interface
func InitGcePdDriver() TestDriver {
	return &gcePdDriver{
		driverInfo: DriverInfo{
			Name:        "gcepd",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   true,
		},
	}
}

func (g *gcePdDriver) GetDriverInfo() *DriverInfo {
	return &g.driverInfo
}

func (g *gcePdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("gce", "gke")
	if pattern.FsType == "xfs" {
		framework.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
}

func (g *gcePdDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	volSource := v1.VolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   g.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.GCEPersistentDisk.FSType = fsType
	}
	return &volSource
}

func (g *gcePdDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	pvSource := v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   g.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.GCEPersistentDisk.FSType = fsType
	}
	return &pvSource
}

func (g *gcePdDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/gce-pd"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := g.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (g *gcePdDriver) CreateDriver() {
}

func (g *gcePdDriver) CleanupDriver() {
}

func (g *gcePdDriver) CreateVolume(volType testpatterns.TestVolType) {
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		g.driverInfo.Config.NodeSelector = map[string]string{
			kubeletapis.LabelZoneFailureDomain: framework.TestContext.CloudConfig.Zone,
		}
	}
	By("creating a test gce pd volume")
	var err error
	g.volumeName, err = framework.CreatePDWithRetry()
	Expect(err).NotTo(HaveOccurred())
}

func (g *gcePdDriver) DeleteVolume(volType testpatterns.TestVolType) {
	framework.DeletePDWithRetry(g.volumeName)
}

// vSphere
type vSphereDriver struct {
	volumePath string
	nodeInfo   *vspheretest.NodeInfo

	driverInfo DriverInfo
}

var _ TestDriver = &vSphereDriver{}
var _ PreprovisionedVolumeTestDriver = &vSphereDriver{}
var _ InlineVolumeTestDriver = &vSphereDriver{}
var _ PreprovisionedPVTestDriver = &vSphereDriver{}
var _ DynamicPVTestDriver = &vSphereDriver{}

// InitVSphereDriver returns vSphereDriver that implements TestDriver interface
func InitVSphereDriver() TestDriver {
	return &vSphereDriver{
		driverInfo: DriverInfo{
			Name:        "vSphere",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   false,
		},
	}
}
func (v *vSphereDriver) GetDriverInfo() *DriverInfo {
	return &v.driverInfo
}

func (v *vSphereDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("vsphere")
}

func (v *vSphereDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	// vSphere driver doesn't seem to support readOnly volume
	// TODO: check if it is correct
	if readOnly {
		return nil
	}
	volSource := v1.VolumeSource{
		VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
			VolumePath: v.volumePath,
		},
	}
	if fsType != "" {
		volSource.VsphereVolume.FSType = fsType
	}
	return &volSource
}

func (v *vSphereDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	// vSphere driver doesn't seem to support readOnly volume
	// TODO: check if it is correct
	if readOnly {
		return nil
	}
	pvSource := v1.PersistentVolumeSource{
		VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
			VolumePath: v.volumePath,
		},
	}
	if fsType != "" {
		pvSource.VsphereVolume.FSType = fsType
	}
	return &pvSource
}

func (v *vSphereDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/vsphere-volume"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := v.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", v.driverInfo.Name)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (v *vSphereDriver) CreateDriver() {
}

func (v *vSphereDriver) CleanupDriver() {
}

func (v *vSphereDriver) CreateVolume(volType testpatterns.TestVolType) {
	f := v.driverInfo.Framework
	vspheretest.Bootstrap(f)
	v.nodeInfo = vspheretest.GetReadySchedulableRandomNodeInfo()
	var err error
	v.volumePath, err = v.nodeInfo.VSphere.CreateVolume(&vspheretest.VolumeOptions{}, v.nodeInfo.DataCenterRef)
	Expect(err).NotTo(HaveOccurred())
}

func (v *vSphereDriver) DeleteVolume(volType testpatterns.TestVolType) {
	v.nodeInfo.VSphere.DeleteVolume(v.volumePath, v.nodeInfo.DataCenterRef)
}

// Azure
type azureDriver struct {
	volumeName string

	driverInfo DriverInfo
}

var _ TestDriver = &azureDriver{}
var _ PreprovisionedVolumeTestDriver = &azureDriver{}
var _ InlineVolumeTestDriver = &azureDriver{}
var _ PreprovisionedPVTestDriver = &azureDriver{}
var _ DynamicPVTestDriver = &azureDriver{}

// InitAzureDriver returns azureDriver that implements TestDriver interface
func InitAzureDriver() TestDriver {
	return &azureDriver{
		driverInfo: DriverInfo{
			Name:        "azure",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   true,
		},
	}
}

func (a *azureDriver) GetDriverInfo() *DriverInfo {
	return &a.driverInfo
}

func (a *azureDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("azure")
}

func (a *azureDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	diskName := a.volumeName[(strings.LastIndex(a.volumeName, "/") + 1):]

	volSource := v1.VolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: a.volumeName,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		volSource.AzureDisk.FSType = &fsType
	}
	return &volSource
}

func (a *azureDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	diskName := a.volumeName[(strings.LastIndex(a.volumeName, "/") + 1):]

	pvSource := v1.PersistentVolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: a.volumeName,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		pvSource.AzureDisk.FSType = &fsType
	}
	return &pvSource
}

func (a *azureDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/azure-disk"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := a.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (a *azureDriver) CreateDriver() {
}

func (a *azureDriver) CleanupDriver() {
}

func (a *azureDriver) CreateVolume(volType testpatterns.TestVolType) {
	By("creating a test azure disk volume")
	var err error
	a.volumeName, err = framework.CreatePDWithRetry()
	Expect(err).NotTo(HaveOccurred())
}

func (a *azureDriver) DeleteVolume(volType testpatterns.TestVolType) {
	framework.DeletePDWithRetry(a.volumeName)
}

// AWS
type awsDriver struct {
	volumeName string

	driverInfo DriverInfo
}

var _ TestDriver = &awsDriver{}

// TODO: Fix authorization error in attach operation and uncomment below
//var _ PreprovisionedVolumeTestDriver = &awsDriver{}
//var _ InlineVolumeTestDriver = &awsDriver{}
//var _ PreprovisionedPVTestDriver = &awsDriver{}
var _ DynamicPVTestDriver = &awsDriver{}

// InitAwsDriver returns awsDriver that implements TestDriver interface
func InitAwsDriver() TestDriver {
	return &awsDriver{
		driverInfo: DriverInfo{
			Name:        "aws",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
			),
			IsPersistent:       true,
			IsFsGroupSupported: true,
			IsBlockSupported:   true,
		},
	}
}

func (a *awsDriver) GetDriverInfo() *DriverInfo {
	return &a.driverInfo
}

func (a *awsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("aws")
}

// TODO: Fix authorization error in attach operation and uncomment below
/*
func (a *awsDriver) GetVolumeSource(readOnly bool, fsType string) *v1.VolumeSource {
	volSource := v1.VolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: a.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.AWSElasticBlockStore.FSType = fsType
	}
	return &volSource
}

func (a *awsDriver) GetPersistentVolumeSource(readOnly bool, fsType string) *v1.PersistentVolumeSource {
	pvSource := v1.PersistentVolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: a.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.AWSElasticBlockStore.FSType = fsType
	}
	return &pvSource
}
*/

func (a *awsDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/aws-ebs"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := a.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (a *awsDriver) CreateDriver() {
}

func (a *awsDriver) CleanupDriver() {
}

// TODO: Fix authorization error in attach operation and uncomment below
/*
func (a *awsDriver) CreateVolume(volType testpatterns.TestVolType) {
	By("creating a test aws volume")
	var err error
	a.volumeName, err = framework.CreatePDWithRetry()
	Expect(err).NotTo(HaveOccurred())
}

func (a *awsDriver) DeleteVolume(volType testpatterns.TestVolType) {
	framework.DeletePDWithRetry(a.volumeName)
}
*/
