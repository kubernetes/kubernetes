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
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	vspheretest "k8s.io/kubernetes/test/e2e/storage/vsphere"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// NFS
type nfsDriver struct {
	externalProvisionerPod *v1.Pod
	externalPluginName     string

	driverInfo testsuites.DriverInfo
}

type nfsVolume struct {
	serverIP  string
	serverPod *v1.Pod
	f         *framework.Framework
}

var _ testsuites.TestDriver = &nfsDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &nfsDriver{}
var _ testsuites.InlineVolumeTestDriver = &nfsDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &nfsDriver{}
var _ testsuites.DynamicPVTestDriver = &nfsDriver{}

// InitNFSDriver returns nfsDriver that implements TestDriver interface
func InitNFSDriver() testsuites.TestDriver {
	return &nfsDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "nfs",
			MaxFileSize: testpatterns.FileSizeLarge,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			SupportedMountOption: sets.NewString("proto=tcp", "relatime"),
			RequiredMountOption:  sets.NewString("vers=4.1"),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (n *nfsDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &n.driverInfo
}

func (n *nfsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (n *nfsDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	nv, ok := volume.(*nfsVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to NFS test volume")
	return &v1.VolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   nv.serverIP,
			Path:     "/",
			ReadOnly: readOnly,
		},
	}
}

func (n *nfsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	nv, ok := volume.(*nfsVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to NFS test volume")
	return &v1.PersistentVolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   nv.serverIP,
			Path:     "/",
			ReadOnly: readOnly,
		},
	}, nil
}

func (n *nfsDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := n.externalPluginName
	parameters := map[string]string{"mountOptions": "vers=4.1"}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", n.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (n *nfsDriver) GetClaimSize() string {
	return "5Gi"
}

func (n *nfsDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
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

	return &testsuites.PerTestConfig{
			Driver:    n,
			Prefix:    "nfs",
			Framework: f,
		}, func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, n.externalProvisionerPod))
			clusterRoleBindingName := ns.Name + "--" + "cluster-admin"
			cs.RbacV1beta1().ClusterRoleBindings().Delete(clusterRoleBindingName, metav1.NewDeleteOptions(0))
		}
}

func (n *nfsDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	// NewNFSServer creates a pod for InlineVolume and PreprovisionedPV,
	// and startExternalProvisioner creates a pods for DynamicPV.
	// Therefore, we need a different PrepareTest logic for volType.
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		c, serverPod, serverIP := framework.NewNFSServer(cs, ns.Name, []string{})
		config.ServerConfig = &c
		return &nfsVolume{
			serverIP:  serverIP,
			serverPod: serverPod,
			f:         f,
		}
	case testpatterns.DynamicPV:
		// Do nothing
	default:
		framework.Failf("Unsupported volType:%v is specified", volType)
	}
	return nil
}

func (v *nfsVolume) DeleteVolume() {
	framework.CleanUpVolumeServer(v.f, v.serverPod)
}

// Gluster
type glusterFSDriver struct {
	driverInfo testsuites.DriverInfo
}

type glusterVolume struct {
	prefix    string
	serverPod *v1.Pod
	f         *framework.Framework
}

var _ testsuites.TestDriver = &glusterFSDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &glusterFSDriver{}
var _ testsuites.InlineVolumeTestDriver = &glusterFSDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &glusterFSDriver{}

// InitGlusterFSDriver returns glusterFSDriver that implements TestDriver interface
func InitGlusterFSDriver() testsuites.TestDriver {
	return &glusterFSDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "gluster",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (g *glusterFSDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *glusterFSDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessNodeOSDistroIs("gci", "ubuntu", "custom")
}

func (g *glusterFSDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	gv, ok := volume.(*glusterVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Gluster test volume")

	name := gv.prefix + "-server"
	return &v1.VolumeSource{
		Glusterfs: &v1.GlusterfsVolumeSource{
			EndpointsName: name,
			// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
			Path:     "test_vol",
			ReadOnly: readOnly,
		},
	}
}

func (g *glusterFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	gv, ok := volume.(*glusterVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Gluster test volume")

	name := gv.prefix + "-server"
	return &v1.PersistentVolumeSource{
		Glusterfs: &v1.GlusterfsPersistentVolumeSource{
			EndpointsName: name,
			// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
			Path:     "test_vol",
			ReadOnly: readOnly,
		},
	}, nil
}

func (g *glusterFSDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    g,
		Prefix:    "gluster",
		Framework: f,
	}, func() {}
}

func (g *glusterFSDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	c, serverPod, _ := framework.NewGlusterfsServer(cs, ns.Name)
	config.ServerConfig = &c
	return &glusterVolume{
		prefix:    config.Prefix,
		serverPod: serverPod,
		f:         f,
	}
}

func (v *glusterVolume) DeleteVolume() {
	f := v.f
	cs := f.ClientSet
	ns := f.Namespace

	name := v.prefix + "-server"

	framework.Logf("Deleting Gluster endpoints %q...", name)
	err := cs.CoreV1().Endpoints(ns.Name).Delete(name, nil)
	if err != nil {
		if !errors.IsNotFound(err) {
			framework.Failf("Gluster delete endpoints failed: %v", err)
		}
		framework.Logf("Gluster endpoints %q not found, assuming deleted", name)
	}
	framework.Logf("Deleting Gluster server pod %q...", v.serverPod.Name)
	err = framework.DeletePodWithWait(f, cs, v.serverPod)
	if err != nil {
		framework.Failf("Gluster server pod delete failed: %v", err)
	}
}

// iSCSI
// The iscsiadm utility and iscsi target kernel modules must be installed on all nodes.
type iSCSIDriver struct {
	driverInfo testsuites.DriverInfo
}
type iSCSIVolume struct {
	serverPod *v1.Pod
	serverIP  string
	f         *framework.Framework
}

var _ testsuites.TestDriver = &iSCSIDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &iSCSIDriver{}
var _ testsuites.InlineVolumeTestDriver = &iSCSIDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &iSCSIDriver{}

// InitISCSIDriver returns iSCSIDriver that implements TestDriver interface
func InitISCSIDriver() testsuites.TestDriver {
	return &iSCSIDriver{
		driverInfo: testsuites.DriverInfo{
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
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (i *iSCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &i.driverInfo
}

func (i *iSCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (i *iSCSIDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	iv, ok := volume.(*iSCSIVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to iSCSI test volume")

	volSource := v1.VolumeSource{
		ISCSI: &v1.ISCSIVolumeSource{
			TargetPortal: iv.serverIP + ":3260",
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

func (i *iSCSIDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	iv, ok := volume.(*iSCSIVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to iSCSI test volume")

	pvSource := v1.PersistentVolumeSource{
		ISCSI: &v1.ISCSIPersistentVolumeSource{
			TargetPortal: iv.serverIP + ":3260",
			IQN:          "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
			Lun:          0,
			ReadOnly:     readOnly,
		},
	}
	if fsType != "" {
		pvSource.ISCSI.FSType = fsType
	}
	return &pvSource, nil
}

func (i *iSCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    i,
		Prefix:    "iscsi",
		Framework: f,
	}, func() {}
}

func (i *iSCSIDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	c, serverPod, serverIP := framework.NewISCSIServer(cs, ns.Name)
	config.ServerConfig = &c
	return &iSCSIVolume{
		serverPod: serverPod,
		serverIP:  serverIP,
		f:         f,
	}
}

func (v *iSCSIVolume) DeleteVolume() {
	framework.CleanUpVolumeServer(v.f, v.serverPod)
}

// Ceph RBD
type rbdDriver struct {
	driverInfo testsuites.DriverInfo
}

type rbdVolume struct {
	serverPod *v1.Pod
	serverIP  string
	secret    *v1.Secret
	f         *framework.Framework
}

var _ testsuites.TestDriver = &rbdDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &rbdDriver{}
var _ testsuites.InlineVolumeTestDriver = &rbdDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &rbdDriver{}

// InitRbdDriver returns rbdDriver that implements TestDriver interface
func InitRbdDriver() testsuites.TestDriver {
	return &rbdDriver{
		driverInfo: testsuites.DriverInfo{
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
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (r *rbdDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &r.driverInfo
}

func (r *rbdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (r *rbdDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	rv, ok := volume.(*rbdVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to RBD test volume")

	volSource := v1.VolumeSource{
		RBD: &v1.RBDVolumeSource{
			CephMonitors: []string{rv.serverIP},
			RBDPool:      "rbd",
			RBDImage:     "foo",
			RadosUser:    "admin",
			SecretRef: &v1.LocalObjectReference{
				Name: rv.secret.Name,
			},
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.RBD.FSType = fsType
	}
	return &volSource
}

func (r *rbdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	rv, ok := volume.(*rbdVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to RBD test volume")

	f := rv.f
	ns := f.Namespace

	pvSource := v1.PersistentVolumeSource{
		RBD: &v1.RBDPersistentVolumeSource{
			CephMonitors: []string{rv.serverIP},
			RBDPool:      "rbd",
			RBDImage:     "foo",
			RadosUser:    "admin",
			SecretRef: &v1.SecretReference{
				Name:      rv.secret.Name,
				Namespace: ns.Name,
			},
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.RBD.FSType = fsType
	}
	return &pvSource, nil
}

func (r *rbdDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    r,
		Prefix:    "rbd",
		Framework: f,
	}, func() {}
}

func (r *rbdDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	c, serverPod, secret, serverIP := framework.NewRBDServer(cs, ns.Name)
	config.ServerConfig = &c
	return &rbdVolume{
		serverPod: serverPod,
		serverIP:  serverIP,
		secret:    secret,
		f:         f,
	}
}

func (v *rbdVolume) DeleteVolume() {
	framework.CleanUpVolumeServerWithSecret(v.f, v.serverPod, v.secret)
}

// Ceph
type cephFSDriver struct {
	serverIP  string
	serverPod *v1.Pod
	secret    *v1.Secret

	driverInfo testsuites.DriverInfo
}

type cephVolume struct {
	serverPod *v1.Pod
	serverIP  string
	secret    *v1.Secret
	f         *framework.Framework
}

var _ testsuites.TestDriver = &cephFSDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &cephFSDriver{}
var _ testsuites.InlineVolumeTestDriver = &cephFSDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &cephFSDriver{}

// InitCephFSDriver returns cephFSDriver that implements TestDriver interface
func InitCephFSDriver() testsuites.TestDriver {
	return &cephFSDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "ceph",
			FeatureTag:  "[Feature:Volumes]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (c *cephFSDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &c.driverInfo
}

func (c *cephFSDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (c *cephFSDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	cv, ok := volume.(*cephVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Ceph test volume")

	return &v1.VolumeSource{
		CephFS: &v1.CephFSVolumeSource{
			Monitors: []string{cv.serverIP + ":6789"},
			User:     "kube",
			SecretRef: &v1.LocalObjectReference{
				Name: cv.secret.Name,
			},
			ReadOnly: readOnly,
		},
	}
}

func (c *cephFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	cv, ok := volume.(*cephVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Ceph test volume")

	ns := cv.f.Namespace

	return &v1.PersistentVolumeSource{
		CephFS: &v1.CephFSPersistentVolumeSource{
			Monitors: []string{cv.serverIP + ":6789"},
			User:     "kube",
			SecretRef: &v1.SecretReference{
				Name:      cv.secret.Name,
				Namespace: ns.Name,
			},
			ReadOnly: readOnly,
		},
	}, nil
}

func (c *cephFSDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    c,
		Prefix:    "cephfs",
		Framework: f,
	}, func() {}
}

func (c *cephFSDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	cfg, serverPod, secret, serverIP := framework.NewRBDServer(cs, ns.Name)
	config.ServerConfig = &cfg
	return &cephVolume{
		serverPod: serverPod,
		serverIP:  serverIP,
		secret:    secret,
		f:         f,
	}
}

func (v *cephVolume) DeleteVolume() {
	framework.CleanUpVolumeServerWithSecret(v.f, v.serverPod, v.secret)
}

// Hostpath
type hostPathDriver struct {
	node v1.Node

	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &hostPathDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &hostPathDriver{}
var _ testsuites.InlineVolumeTestDriver = &hostPathDriver{}

// InitHostPathDriver returns hostPathDriver that implements TestDriver interface
func InitHostPathDriver() testsuites.TestDriver {
	return &hostPathDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "hostPath",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
			},
		},
	}
}

func (h *hostPathDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &h.driverInfo
}

func (h *hostPathDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostPathDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
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

func (h *hostPathDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    h,
		Prefix:    "hostpath",
		Framework: f,
	}, func() {}
}

func (h *hostPathDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	config.ClientNodeName = node.Name
	return nil
}

// HostPathSymlink
type hostPathSymlinkDriver struct {
	node v1.Node

	driverInfo testsuites.DriverInfo
}

type hostPathSymlinkVolume struct {
	targetPath string
	sourcePath string
	prepPod    *v1.Pod
	f          *framework.Framework
}

var _ testsuites.TestDriver = &hostPathSymlinkDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &hostPathSymlinkDriver{}
var _ testsuites.InlineVolumeTestDriver = &hostPathSymlinkDriver{}

// InitHostPathSymlinkDriver returns hostPathSymlinkDriver that implements TestDriver interface
func InitHostPathSymlinkDriver() testsuites.TestDriver {
	return &hostPathSymlinkDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "hostPathSymlink",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
			},
		},
	}
}

func (h *hostPathSymlinkDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &h.driverInfo
}

func (h *hostPathSymlinkDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostPathSymlinkDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	hv, ok := volume.(*hostPathSymlinkVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Hostpath Symlink test volume")

	// hostPathSymlink doesn't support readOnly volume
	if readOnly {
		return nil
	}
	return &v1.VolumeSource{
		HostPath: &v1.HostPathVolumeSource{
			Path: hv.targetPath,
		},
	}
}

func (h *hostPathSymlinkDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    h,
		Prefix:    "hostpathsymlink",
		Framework: f,
	}, func() {}
}

func (h *hostPathSymlinkDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet

	sourcePath := fmt.Sprintf("/tmp/%v", f.Namespace.Name)
	targetPath := fmt.Sprintf("/tmp/%v-link", f.Namespace.Name)
	volumeName := "test-volume"

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	config.ClientNodeName = node.Name

	cmd := fmt.Sprintf("mkdir %v -m 777 && ln -s %v %v", sourcePath, sourcePath, targetPath)
	privileged := true

	// Launch pod to initialize hostPath directory and symlink
	prepPod := &v1.Pod{
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
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(prepPod)
	Expect(err).ToNot(HaveOccurred(), "while creating hostPath init pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for hostPath init pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting hostPath init pod")
	return &hostPathSymlinkVolume{
		sourcePath: sourcePath,
		targetPath: targetPath,
		prepPod:    prepPod,
		f:          f,
	}
}

func (v *hostPathSymlinkVolume) DeleteVolume() {
	f := v.f

	cmd := fmt.Sprintf("rm -rf %v&& rm -rf %v", v.targetPath, v.sourcePath)
	v.prepPod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", cmd}

	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(v.prepPod)
	Expect(err).ToNot(HaveOccurred(), "while creating hostPath teardown pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for hostPath teardown pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting hostPath teardown pod")
}

// emptydir
type emptydirDriver struct {
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &emptydirDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &emptydirDriver{}
var _ testsuites.InlineVolumeTestDriver = &emptydirDriver{}

// InitEmptydirDriver returns emptydirDriver that implements TestDriver interface
func InitEmptydirDriver() testsuites.TestDriver {
	return &emptydirDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "emptydir",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapExec: true,
			},
		},
	}
}

func (e *emptydirDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &e.driverInfo
}

func (e *emptydirDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (e *emptydirDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	// emptydir doesn't support readOnly volume
	if readOnly {
		return nil
	}
	return &v1.VolumeSource{
		EmptyDir: &v1.EmptyDirVolumeSource{},
	}
}

func (e *emptydirDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	return nil
}

func (e *emptydirDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    e,
		Prefix:    "emptydir",
		Framework: f,
	}, func() {}
}

// Cinder
// This driver assumes that OpenStack client tools are installed
// (/usr/bin/nova, /usr/bin/cinder and /usr/bin/keystone)
// and that the usual OpenStack authentication env. variables are set
// (OS_USERNAME, OS_PASSWORD, OS_TENANT_NAME at least).
type cinderDriver struct {
	driverInfo testsuites.DriverInfo
}

type cinderVolume struct {
	volumeName string
	volumeID   string
}

var _ testsuites.TestDriver = &cinderDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &cinderDriver{}
var _ testsuites.InlineVolumeTestDriver = &cinderDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &cinderDriver{}
var _ testsuites.DynamicPVTestDriver = &cinderDriver{}

// InitCinderDriver returns cinderDriver that implements TestDriver interface
func InitCinderDriver() testsuites.TestDriver {
	return &cinderDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "cinder",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (c *cinderDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &c.driverInfo
}

func (c *cinderDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("openstack")
}

func (c *cinderDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	cv, ok := volume.(*cinderVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Cinder test volume")

	volSource := v1.VolumeSource{
		Cinder: &v1.CinderVolumeSource{
			VolumeID: cv.volumeID,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.Cinder.FSType = fsType
	}
	return &volSource
}

func (c *cinderDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	cv, ok := volume.(*cinderVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Cinder test volume")

	pvSource := v1.PersistentVolumeSource{
		Cinder: &v1.CinderPersistentVolumeSource{
			VolumeID: cv.volumeID,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.Cinder.FSType = fsType
	}
	return &pvSource, nil
}

func (c *cinderDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/cinder"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", c.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (c *cinderDriver) GetClaimSize() string {
	return "5Gi"
}

func (c *cinderDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    c,
		Prefix:    "cinder",
		Framework: f,
	}, func() {}
}

func (c *cinderDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	ns := f.Namespace

	// We assume that namespace.Name is a random string
	volumeName := ns.Name
	By("creating a test Cinder volume")
	output, err := exec.Command("cinder", "create", "--display-name="+volumeName, "1").CombinedOutput()
	outputString := string(output[:])
	framework.Logf("cinder output:\n%s", outputString)
	Expect(err).NotTo(HaveOccurred())

	// Parse 'id'' from stdout. Expected format:
	// |     attachments     |                  []                  |
	// |  availability_zone  |                 nova                 |
	// ...
	// |          id         | 1d6ff08f-5d1c-41a4-ad72-4ef872cae685 |
	volumeID := ""
	for _, line := range strings.Split(outputString, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 5 {
			continue
		}
		if fields[1] != "id" {
			continue
		}
		volumeID = fields[3]
		break
	}
	framework.Logf("Volume ID: %s", volumeID)
	Expect(volumeID).NotTo(Equal(""))
	return &cinderVolume{
		volumeName: volumeName,
		volumeID:   volumeID,
	}
}

func (v *cinderVolume) DeleteVolume() {
	name := v.volumeName

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
			return
		}
		framework.Logf("Failed to delete volume %s: %v", name, err)
	}
	framework.Logf("Giving up deleting volume %s: %v\n%s", name, err, string(output[:]))
}

// GCE
type gcePdDriver struct {
	driverInfo testsuites.DriverInfo
}

type gcePdVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &gcePdDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &gcePdDriver{}
var _ testsuites.InlineVolumeTestDriver = &gcePdDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &gcePdDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePdDriver{}

// InitGcePdDriver returns gcePdDriver that implements TestDriver interface
func InitGcePdDriver() testsuites.TestDriver {
	// In current test structure, it first initialize the driver and then set up
	// the new framework, so we cannot get the correct OS here. So here set to
	// support all fs types including both linux and windows. We have code to check Node OS later
	// during test.
	supportedTypes := sets.NewString(
		"", // Default fsType
		"ext2",
		"ext3",
		"ext4",
		"xfs",
		"ntfs",
	)
	return &gcePdDriver{
		driverInfo: testsuites.DriverInfo{
			Name:                 "gcepd",
			MaxFileSize:          testpatterns.FileSizeMedium,
			SupportedFsType:      supportedTypes,
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (g *gcePdDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("gce", "gke")
	if pattern.FeatureTag == "[sig-windows]" {
		framework.SkipUnlessNodeOSDistroIs("windows")
	}
}

func (g *gcePdDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	gv, ok := volume.(*gcePdVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to GCE PD test volume")
	volSource := v1.VolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   gv.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.GCEPersistentDisk.FSType = fsType
	}
	return &volSource
}

func (g *gcePdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	gv, ok := volume.(*gcePdVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to GCE PD test volume")
	pvSource := v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   gv.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.GCEPersistentDisk.FSType = fsType
	}
	return &pvSource, nil
}

func (g *gcePdDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/gce-pd"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (h *gcePdDriver) GetClaimSize() string {
	return "5Gi"
}

func (g *gcePdDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	config := &testsuites.PerTestConfig{
		Driver:    g,
		Prefix:    "gcepd",
		Framework: f,
	}
	if framework.NodeOSDistroIs("windows") {
		config.ClientNodeSelector = map[string]string{
			"beta.kubernetes.io/os": "windows",
		}
	}
	return config, func() {}

}

func (g *gcePdDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelector = map[string]string{
			v1.LabelZoneFailureDomain: framework.TestContext.CloudConfig.Zone,
		}
	}
	By("creating a test gce pd volume")
	vname, err := framework.CreatePDWithRetry()
	Expect(err).NotTo(HaveOccurred())
	return &gcePdVolume{
		volumeName: vname,
	}
}

func (v *gcePdVolume) DeleteVolume() {
	framework.DeletePDWithRetry(v.volumeName)
}

// vSphere
type vSphereDriver struct {
	driverInfo testsuites.DriverInfo
}

type vSphereVolume struct {
	volumePath string
	nodeInfo   *vspheretest.NodeInfo
}

var _ testsuites.TestDriver = &vSphereDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &vSphereDriver{}
var _ testsuites.InlineVolumeTestDriver = &vSphereDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &vSphereDriver{}
var _ testsuites.DynamicPVTestDriver = &vSphereDriver{}

// InitVSphereDriver returns vSphereDriver that implements TestDriver interface
func InitVSphereDriver() testsuites.TestDriver {
	return &vSphereDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "vSphere",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
			},
		},
	}
}
func (v *vSphereDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &v.driverInfo
}

func (v *vSphereDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("vsphere")
}

func (v *vSphereDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	vsv, ok := volume.(*vSphereVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to vSphere test volume")

	// vSphere driver doesn't seem to support readOnly volume
	// TODO: check if it is correct
	if readOnly {
		return nil
	}
	volSource := v1.VolumeSource{
		VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
			VolumePath: vsv.volumePath,
		},
	}
	if fsType != "" {
		volSource.VsphereVolume.FSType = fsType
	}
	return &volSource
}

func (v *vSphereDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	vsv, ok := volume.(*vSphereVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to vSphere test volume")

	// vSphere driver doesn't seem to support readOnly volume
	// TODO: check if it is correct
	if readOnly {
		return nil, nil
	}
	pvSource := v1.PersistentVolumeSource{
		VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
			VolumePath: vsv.volumePath,
		},
	}
	if fsType != "" {
		pvSource.VsphereVolume.FSType = fsType
	}
	return &pvSource, nil
}

func (v *vSphereDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/vsphere-volume"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", v.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (v *vSphereDriver) GetClaimSize() string {
	return "5Gi"
}

func (v *vSphereDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    v,
		Prefix:    "vsphere",
		Framework: f,
	}, func() {}
}

func (v *vSphereDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	vspheretest.Bootstrap(f)
	nodeInfo := vspheretest.GetReadySchedulableRandomNodeInfo()
	volumePath, err := nodeInfo.VSphere.CreateVolume(&vspheretest.VolumeOptions{}, nodeInfo.DataCenterRef)
	Expect(err).NotTo(HaveOccurred())
	return &vSphereVolume{
		volumePath: volumePath,
		nodeInfo:   nodeInfo,
	}
}

func (v *vSphereVolume) DeleteVolume() {
	v.nodeInfo.VSphere.DeleteVolume(v.volumePath, v.nodeInfo.DataCenterRef)
}

// Azure
type azureDriver struct {
	driverInfo testsuites.DriverInfo
}

type azureVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &azureDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &azureDriver{}
var _ testsuites.InlineVolumeTestDriver = &azureDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &azureDriver{}
var _ testsuites.DynamicPVTestDriver = &azureDriver{}

// InitAzureDriver returns azureDriver that implements TestDriver interface
func InitAzureDriver() testsuites.TestDriver {
	return &azureDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "azure",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (a *azureDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &a.driverInfo
}

func (a *azureDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("azure")
}

func (a *azureDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
	av, ok := volume.(*azureVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Azure test volume")

	diskName := av.volumeName[(strings.LastIndex(av.volumeName, "/") + 1):]

	volSource := v1.VolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: av.volumeName,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		volSource.AzureDisk.FSType = &fsType
	}
	return &volSource
}

func (a *azureDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	av, ok := volume.(*azureVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to Azure test volume")

	diskName := av.volumeName[(strings.LastIndex(av.volumeName, "/") + 1):]

	pvSource := v1.PersistentVolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: av.volumeName,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		pvSource.AzureDisk.FSType = &fsType
	}
	return &pvSource, nil
}

func (a *azureDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/azure-disk"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (a *azureDriver) GetClaimSize() string {
	return "5Gi"
}

func (a *azureDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    a,
		Prefix:    "azure",
		Framework: f,
	}, func() {}
}

func (a *azureDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	By("creating a test azure disk volume")
	volumeName, err := framework.CreatePDWithRetry()
	Expect(err).NotTo(HaveOccurred())
	return &azureVolume{
		volumeName: volumeName,
	}
}

func (v *azureVolume) DeleteVolume() {
	framework.DeletePDWithRetry(v.volumeName)
}

// AWS
type awsDriver struct {
	volumeName string

	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &awsDriver{}

// TODO: Fix authorization error in attach operation and uncomment below
//var _ testsuites.PreprovisionedVolumeTestDriver = &awsDriver{}
//var _ testsuites.InlineVolumeTestDriver = &awsDriver{}
//var _ testsuites.PreprovisionedPVTestDriver = &awsDriver{}
var _ testsuites.DynamicPVTestDriver = &awsDriver{}

// InitAwsDriver returns awsDriver that implements TestDriver interface
func InitAwsDriver() testsuites.TestDriver {
	return &awsDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "aws",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
			),
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
			},
		},
	}
}

func (a *awsDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &a.driverInfo
}

func (a *awsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("aws")
}

// TODO: Fix authorization error in attach operation and uncomment below
/*
func (a *awsDriver) GetVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) *v1.VolumeSource {
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

func (a *awsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (a *awsDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/aws-ebs"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (a *awsDriver) GetClaimSize() string {
	return "5Gi"
}

func (a *awsDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    a,
		Prefix:    "aws",
		Framework: f,
	}, func() {}
}

// TODO: Fix authorization error in attach operation and uncomment below
/*
func (a *awsDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	By("creating a test aws volume")
	var err error
	a.volumeName, err = framework.CreatePDWithRetry()
	Expect(err).NotTo(HaveOccurred())
}

DeleteVolume() {
	framework.DeletePDWithRetry(a.volumeName)
}
*/

// local
type localDriver struct {
	driverInfo testsuites.DriverInfo
	node       *v1.Node
	hostExec   utils.HostExec
	// volumeType represents local volume type we are testing, e.g.  tmpfs,
	// directory, block device.
	volumeType utils.LocalVolumeType
	ltrMgr     utils.LocalTestResourceManager
}

type localVolume struct {
	ltrMgr utils.LocalTestResourceManager
	ltr    *utils.LocalTestResource
}

var (
	// capabilities
	defaultLocalVolumeCapabilities = map[testsuites.Capability]bool{
		testsuites.CapPersistence: true,
		testsuites.CapFsGroup:     true,
		testsuites.CapBlock:       false,
		testsuites.CapExec:        true,
	}
	localVolumeCapabitilies = map[utils.LocalVolumeType]map[testsuites.Capability]bool{
		utils.LocalVolumeBlock: {
			testsuites.CapPersistence: true,
			testsuites.CapFsGroup:     true,
			testsuites.CapBlock:       true,
			testsuites.CapExec:        true,
		},
	}
	// fstype
	defaultLocalVolumeSupportedFsTypes = sets.NewString("")
	localVolumeSupportedFsTypes        = map[utils.LocalVolumeType]sets.String{
		utils.LocalVolumeBlock: sets.NewString(
			"", // Default fsType
			"ext2",
			"ext3",
			"ext4",
			//"xfs", disabled see issue https://github.com/kubernetes/kubernetes/issues/74095
		),
	}
	// max file size
	defaultLocalVolumeMaxFileSize = testpatterns.FileSizeSmall
	localVolumeMaxFileSizes       = map[utils.LocalVolumeType]int64{}
)

var _ testsuites.TestDriver = &localDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &localDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &localDriver{}

func InitLocalDriverWithVolumeType(volumeType utils.LocalVolumeType) func() testsuites.TestDriver {
	maxFileSize := defaultLocalVolumeMaxFileSize
	if maxFileSizeByVolType, ok := localVolumeMaxFileSizes[volumeType]; ok {
		maxFileSize = maxFileSizeByVolType
	}
	supportedFsTypes := defaultLocalVolumeSupportedFsTypes
	if supportedFsTypesByType, ok := localVolumeSupportedFsTypes[volumeType]; ok {
		supportedFsTypes = supportedFsTypesByType
	}
	capabilities := defaultLocalVolumeCapabilities
	if capabilitiesByType, ok := localVolumeCapabitilies[volumeType]; ok {
		capabilities = capabilitiesByType
	}
	return func() testsuites.TestDriver {
		// custom tag to distinguish from tests of other volume types
		featureTag := fmt.Sprintf("[LocalVolumeType: %s]", volumeType)
		// For GCE Local SSD volumes, we must run serially
		if volumeType == utils.LocalVolumeGCELocalSSD {
			featureTag += " [Serial]"
		}
		return &localDriver{
			driverInfo: testsuites.DriverInfo{
				Name:            "local",
				FeatureTag:      featureTag,
				MaxFileSize:     maxFileSize,
				SupportedFsType: supportedFsTypes,
				Capabilities:    capabilities,
			},
			volumeType: volumeType,
		}
	}
}

func (l *localDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &l.driverInfo
}

func (l *localDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (l *localDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	// choose a randome node to test against
	nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	l.node = &nodes.Items[rand.Intn(len(nodes.Items))]

	l.hostExec = utils.NewHostExec(f)
	l.ltrMgr = utils.NewLocalResourceManager("local-driver", l.hostExec, "/tmp")

	// This can't be done in SkipUnsupportedTest because the test framework is not initialized yet
	if l.volumeType == utils.LocalVolumeGCELocalSSD {
		ssdInterface := "scsi"
		filesystemType := "fs"
		ssdCmd := fmt.Sprintf("ls -1 /mnt/disks/by-uuid/google-local-ssds-%s-%s/ | wc -l", ssdInterface, filesystemType)
		res, err := l.hostExec.IssueCommandWithResult(ssdCmd, l.node)
		Expect(err).NotTo(HaveOccurred())
		num, err := strconv.Atoi(strings.TrimSpace(res))
		Expect(err).NotTo(HaveOccurred())
		if num < 1 {
			framework.Skipf("Requires at least 1 %s %s localSSD ", ssdInterface, filesystemType)
		}
	}

	return &testsuites.PerTestConfig{
			Driver:         l,
			Prefix:         "local",
			Framework:      f,
			ClientNodeName: l.node.Name,
		}, func() {
			l.hostExec.Cleanup()
		}
}

func (l *localDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	switch volType {
	case testpatterns.PreprovisionedPV:
		node := l.node
		// assign this to schedule pod on this node
		config.ClientNodeName = node.Name
		return &localVolume{
			ltrMgr: l.ltrMgr,
			ltr:    l.ltrMgr.Create(node, l.volumeType, nil),
		}
	default:
		framework.Failf("Unsupported volType: %v is specified", volType)
	}
	return nil
}

func (v *localVolume) DeleteVolume() {
	v.ltrMgr.Remove(v.ltr)
}

func (l *localDriver) nodeAffinityForNode(node *v1.Node) *v1.VolumeNodeAffinity {
	nodeKey := "kubernetes.io/hostname"
	if node.Labels == nil {
		framework.Failf("Node does not have labels")
	}
	nodeValue, found := node.Labels[nodeKey]
	if !found {
		framework.Failf("Node does not have required label %q", nodeKey)
	}
	return &v1.VolumeNodeAffinity{
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
	}
}

func (l *localDriver) GetPersistentVolumeSource(readOnly bool, fsType string, volume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	lv, ok := volume.(*localVolume)
	Expect(ok).To(BeTrue(), "Failed to cast test volume to local test volume")
	return &v1.PersistentVolumeSource{
		Local: &v1.LocalVolumeSource{
			Path:   lv.ltr.Path,
			FSType: &fsType,
		},
	}, l.nodeAffinityForNode(lv.ltr.Node)
}
