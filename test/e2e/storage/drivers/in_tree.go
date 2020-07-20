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
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// Template for iSCSI IQN.
	iSCSIIQNTemplate = "iqn.2003-01.io.k8s:e2e.%s"
)

// NFS
type nfsDriver struct {
	externalProvisionerPod *v1.Pod
	externalPluginName     string

	driverInfo testsuites.DriverInfo
}

type nfsVolume struct {
	serverHost string
	serverPod  *v1.Pod
	f          *framework.Framework
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
			Name:             "nfs",
			InTreePluginName: "kubernetes.io/nfs",
			MaxFileSize:      testpatterns.FileSizeLarge,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			SupportedMountOption: sets.NewString("proto=tcp", "relatime"),
			RequiredMountOption:  sets.NewString("vers=4.1"),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapExec:        true,
				testsuites.CapRWX:         true,
				testsuites.CapMultiPODs:   true,
			},
		},
	}
}

func (n *nfsDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &n.driverInfo
}

func (n *nfsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (n *nfsDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	nv, ok := e2evolume.(*nfsVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to NFS test volume")
	return &v1.VolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   nv.serverHost,
			Path:     "/",
			ReadOnly: readOnly,
		},
	}
}

func (n *nfsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	nv, ok := e2evolume.(*nfsVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to NFS test volume")
	return &v1.PersistentVolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   nv.serverHost,
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

func (n *nfsDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	cs := f.ClientSet
	ns := f.Namespace
	n.externalPluginName = fmt.Sprintf("example.com/nfs-%s", ns.Name)

	// TODO(mkimuram): cluster-admin gives too much right but system:persistent-volume-provisioner
	// is not enough. We should create new clusterrole for testing.
	err := e2eauth.BindClusterRole(cs.RbacV1(), "cluster-admin", ns.Name,
		rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: ns.Name, Name: "default"})
	framework.ExpectNoError(err)

	err = e2eauth.WaitForAuthorizationUpdate(cs.AuthorizationV1(),
		serviceaccount.MakeUsername(ns.Name, "default"),
		"", "get", schema.GroupResource{Group: "storage.k8s.io", Resource: "storageclasses"}, true)
	framework.ExpectNoError(err, "Failed to update authorization: %v", err)

	ginkgo.By("creating an external dynamic provisioner pod")
	n.externalProvisionerPod = utils.StartExternalProvisioner(cs, ns.Name, n.externalPluginName)

	return &testsuites.PerTestConfig{
			Driver:    n,
			Prefix:    "nfs",
			Framework: f,
		}, func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(cs, n.externalProvisionerPod))
			clusterRoleBindingName := ns.Name + "--" + "cluster-admin"
			cs.RbacV1().ClusterRoleBindings().Delete(context.TODO(), clusterRoleBindingName, *metav1.NewDeleteOptions(0))
		}
}

func (n *nfsDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	// NewNFSServer creates a pod for InlineVolume and PreprovisionedPV,
	// and startExternalProvisioner creates a pod for DynamicPV.
	// Therefore, we need a different PrepareTest logic for volType.
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		c, serverPod, serverHost := e2evolume.NewNFSServer(cs, ns.Name, []string{})
		config.ServerConfig = &c
		return &nfsVolume{
			serverHost: serverHost,
			serverPod:  serverPod,
			f:          f,
		}
	case testpatterns.DynamicPV:
		// Do nothing
	default:
		framework.Failf("Unsupported volType:%v is specified", volType)
	}
	return nil
}

func (v *nfsVolume) DeleteVolume() {
	cleanUpVolumeServer(v.f, v.serverPod)
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
			Name:             "gluster",
			InTreePluginName: "kubernetes.io/glusterfs",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapExec:        true,
				testsuites.CapRWX:         true,
				testsuites.CapMultiPODs:   true,
			},
		},
	}
}

func (g *glusterFSDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *glusterFSDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessNodeOSDistroIs("gci", "ubuntu", "custom")
}

func (g *glusterFSDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	gv, ok := e2evolume.(*glusterVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Gluster test volume")

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

func (g *glusterFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	gv, ok := e2evolume.(*glusterVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Gluster test volume")

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

	c, serverPod, _ := e2evolume.NewGlusterfsServer(cs, ns.Name)
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
	err := cs.CoreV1().Endpoints(ns.Name).Delete(context.TODO(), name, metav1.DeleteOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			framework.Failf("Gluster delete endpoints failed: %v", err)
		}
		framework.Logf("Gluster endpoints %q not found, assuming deleted", name)
	}
	framework.Logf("Deleting Gluster server pod %q...", v.serverPod.Name)
	err = e2epod.DeletePodWithWait(cs, v.serverPod)
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
	iqn       string
}

var _ testsuites.TestDriver = &iSCSIDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &iSCSIDriver{}
var _ testsuites.InlineVolumeTestDriver = &iSCSIDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &iSCSIDriver{}

// InitISCSIDriver returns iSCSIDriver that implements TestDriver interface
func InitISCSIDriver() testsuites.TestDriver {
	return &iSCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "iscsi",
			InTreePluginName: "kubernetes.io/iscsi",
			FeatureTag:       "[Feature:Volumes]",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				// TODO: fix iSCSI driver can work with ext3
				//"ext3",
				"ext4",
			),
			TopologyKeys: []string{v1.LabelHostname},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
				testsuites.CapTopology:    true,
			},
		},
	}
}

func (i *iSCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &i.driverInfo
}

func (i *iSCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (i *iSCSIDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	iv, ok := e2evolume.(*iSCSIVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to iSCSI test volume")

	volSource := v1.VolumeSource{
		ISCSI: &v1.ISCSIVolumeSource{
			TargetPortal: "127.0.0.1:3260",
			IQN:          iv.iqn,
			Lun:          0,
			ReadOnly:     readOnly,
		},
	}
	if fsType != "" {
		volSource.ISCSI.FSType = fsType
	}
	return &volSource
}

func (i *iSCSIDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	iv, ok := e2evolume.(*iSCSIVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to iSCSI test volume")

	pvSource := v1.PersistentVolumeSource{
		ISCSI: &v1.ISCSIPersistentVolumeSource{
			TargetPortal: "127.0.0.1:3260",
			IQN:          iv.iqn,
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

	c, serverPod, serverIP, iqn := newISCSIServer(cs, ns.Name)
	config.ServerConfig = &c
	config.ClientNodeSelection = c.ClientNodeSelection
	return &iSCSIVolume{
		serverPod: serverPod,
		serverIP:  serverIP,
		iqn:       iqn,
		f:         f,
	}
}

// newISCSIServer is an iSCSI-specific wrapper for CreateStorageServer.
func newISCSIServer(cs clientset.Interface, namespace string) (config e2evolume.TestConfig, pod *v1.Pod, ip, iqn string) {
	// Generate cluster-wide unique IQN
	iqn = fmt.Sprintf(iSCSIIQNTemplate, namespace)
	config = e2evolume.TestConfig{
		Namespace:   namespace,
		Prefix:      "iscsi",
		ServerImage: imageutils.GetE2EImage(imageutils.VolumeISCSIServer),
		ServerArgs:  []string{iqn},
		ServerVolumes: map[string]string{
			// iSCSI container needs to insert modules from the host
			"/lib/modules": "/lib/modules",
			// iSCSI container needs to configure kernel
			"/sys/kernel": "/sys/kernel",
			// iSCSI source "block devices" must be available on the host
			"/srv/iscsi": "/srv/iscsi",
		},
		ServerReadyMessage: "iscsi target started",
		ServerHostNetwork:  true,
	}
	pod, ip = e2evolume.CreateStorageServer(cs, config)
	// Make sure the client runs on the same node as server so we don't need to open any firewalls.
	config.ClientNodeSelection = e2epod.NodeSelection{Name: pod.Spec.NodeName}
	return config, pod, ip, iqn
}

// newRBDServer is a CephRBD-specific wrapper for CreateStorageServer.
func newRBDServer(cs clientset.Interface, namespace string) (config e2evolume.TestConfig, pod *v1.Pod, secret *v1.Secret, ip string) {
	config = e2evolume.TestConfig{
		Namespace:   namespace,
		Prefix:      "rbd",
		ServerImage: imageutils.GetE2EImage(imageutils.VolumeRBDServer),
		ServerPorts: []int{6789},
		ServerVolumes: map[string]string{
			"/lib/modules": "/lib/modules",
		},
		ServerReadyMessage: "Ceph is ready",
	}
	pod, ip = e2evolume.CreateStorageServer(cs, config)
	// create secrets for the server
	secret = &v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-secret",
		},
		Data: map[string][]byte{
			// from test/images/volumes-tester/rbd/keyring
			"key": []byte("AQDRrKNVbEevChAAEmRC+pW/KBVHxa0w/POILA=="),
		},
		Type: "kubernetes.io/rbd",
	}

	secret, err := cs.CoreV1().Secrets(config.Namespace).Create(context.TODO(), secret, metav1.CreateOptions{})
	if err != nil {
		framework.Failf("Failed to create secrets for Ceph RBD: %v", err)
	}

	return config, pod, secret, ip
}

func (v *iSCSIVolume) DeleteVolume() {
	cleanUpVolumeServer(v.f, v.serverPod)
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
			Name:             "rbd",
			InTreePluginName: "kubernetes.io/rbd",
			FeatureTag:       "[Feature:Volumes][Serial]",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
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
				testsuites.CapMultiPODs:   true,
			},
		},
	}
}

func (r *rbdDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &r.driverInfo
}

func (r *rbdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (r *rbdDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	rv, ok := e2evolume.(*rbdVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to RBD test volume")

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

func (r *rbdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	rv, ok := e2evolume.(*rbdVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to RBD test volume")

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

	c, serverPod, secret, serverIP := newRBDServer(cs, ns.Name)
	config.ServerConfig = &c
	return &rbdVolume{
		serverPod: serverPod,
		serverIP:  serverIP,
		secret:    secret,
		f:         f,
	}
}

func (v *rbdVolume) DeleteVolume() {
	cleanUpVolumeServerWithSecret(v.f, v.serverPod, v.secret)
}

// Ceph
type cephFSDriver struct {
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
			Name:             "ceph",
			InTreePluginName: "kubernetes.io/cephfs",
			FeatureTag:       "[Feature:Volumes][Serial]",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapExec:        true,
				testsuites.CapRWX:         true,
				testsuites.CapMultiPODs:   true,
			},
		},
	}
}

func (c *cephFSDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &c.driverInfo
}

func (c *cephFSDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (c *cephFSDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	cv, ok := e2evolume.(*cephVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Ceph test volume")

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

func (c *cephFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	cv, ok := e2evolume.(*cephVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Ceph test volume")

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

	cfg, serverPod, secret, serverIP := newRBDServer(cs, ns.Name)
	config.ServerConfig = &cfg
	return &cephVolume{
		serverPod: serverPod,
		serverIP:  serverIP,
		secret:    secret,
		f:         f,
	}
}

func (v *cephVolume) DeleteVolume() {
	cleanUpVolumeServerWithSecret(v.f, v.serverPod, v.secret)
}

// Hostpath
type hostPathDriver struct {
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &hostPathDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &hostPathDriver{}
var _ testsuites.InlineVolumeTestDriver = &hostPathDriver{}

// InitHostPathDriver returns hostPathDriver that implements TestDriver interface
func InitHostPathDriver() testsuites.TestDriver {
	return &hostPathDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "hostPath",
			InTreePluginName: "kubernetes.io/host-path",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			TopologyKeys: []string{v1.LabelHostname},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:      true,
				testsuites.CapMultiPODs:        true,
				testsuites.CapSingleNodeVolume: true,
				testsuites.CapTopology:         true,
			},
		},
	}
}

func (h *hostPathDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &h.driverInfo
}

func (h *hostPathDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostPathDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
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
	node, err := e2enode.GetRandomReadySchedulableNode(cs)
	framework.ExpectNoError(err)
	config.ClientNodeSelection = e2epod.NodeSelection{Name: node.Name}
	return nil
}

// HostPathSymlink
type hostPathSymlinkDriver struct {
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
			Name:             "hostPathSymlink",
			InTreePluginName: "kubernetes.io/host-path",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			TopologyKeys: []string{v1.LabelHostname},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:      true,
				testsuites.CapMultiPODs:        true,
				testsuites.CapSingleNodeVolume: true,
				testsuites.CapTopology:         true,
			},
		},
	}
}

func (h *hostPathSymlinkDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &h.driverInfo
}

func (h *hostPathSymlinkDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostPathSymlinkDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	hv, ok := e2evolume.(*hostPathSymlinkVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Hostpath Symlink test volume")

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
	node, err := e2enode.GetRandomReadySchedulableNode(cs)
	framework.ExpectNoError(err)
	config.ClientNodeSelection = e2epod.NodeSelection{Name: node.Name}

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
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), prepPod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating hostPath init pod")

	err = e2epod.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	framework.ExpectNoError(err, "while waiting for hostPath init pod to succeed")

	err = e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting hostPath init pod")
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

	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), v.prepPod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating hostPath teardown pod")

	err = e2epod.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	framework.ExpectNoError(err, "while waiting for hostPath teardown pod to succeed")

	err = e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting hostPath teardown pod")
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
			Name:             "emptydir",
			InTreePluginName: "kubernetes.io/empty-dir",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapExec:             true,
				testsuites.CapSingleNodeVolume: true,
			},
		},
	}
}

func (e *emptydirDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &e.driverInfo
}

func (e *emptydirDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (e *emptydirDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
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
		testsuites.CapPersistence:      true,
		testsuites.CapFsGroup:          true,
		testsuites.CapBlock:            false,
		testsuites.CapExec:             true,
		testsuites.CapMultiPODs:        true,
		testsuites.CapSingleNodeVolume: true,
	}
	localVolumeCapabitilies = map[utils.LocalVolumeType]map[testsuites.Capability]bool{
		utils.LocalVolumeBlock: {
			testsuites.CapPersistence:      true,
			testsuites.CapFsGroup:          true,
			testsuites.CapBlock:            true,
			testsuites.CapExec:             true,
			testsuites.CapMultiPODs:        true,
			testsuites.CapSingleNodeVolume: true,
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

// InitLocalDriverWithVolumeType initializes the local driver based on the volume type.
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
				Name:             "local",
				InTreePluginName: "kubernetes.io/local-volume",
				FeatureTag:       featureTag,
				MaxFileSize:      maxFileSize,
				SupportedFsType:  supportedFsTypes,
				Capabilities:     capabilities,
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
	var err error
	l.node, err = e2enode.GetRandomReadySchedulableNode(f.ClientSet)
	framework.ExpectNoError(err)

	l.hostExec = utils.NewHostExec(f)
	l.ltrMgr = utils.NewLocalResourceManager("local-driver", l.hostExec, "/tmp")

	// This can't be done in SkipUnsupportedTest because the test framework is not initialized yet
	if l.volumeType == utils.LocalVolumeGCELocalSSD {
		ssdInterface := "scsi"
		filesystemType := "fs"
		ssdCmd := fmt.Sprintf("ls -1 /mnt/disks/by-uuid/google-local-ssds-%s-%s/ | wc -l", ssdInterface, filesystemType)
		res, err := l.hostExec.IssueCommandWithResult(ssdCmd, l.node)
		framework.ExpectNoError(err)
		num, err := strconv.Atoi(strings.TrimSpace(res))
		framework.ExpectNoError(err)
		if num < 1 {
			e2eskipper.Skipf("Requires at least 1 %s %s localSSD ", ssdInterface, filesystemType)
		}
	}

	return &testsuites.PerTestConfig{
			Driver:              l,
			Prefix:              "local",
			Framework:           f,
			ClientNodeSelection: e2epod.NodeSelection{Name: l.node.Name},
		}, func() {
			l.hostExec.Cleanup()
		}
}

func (l *localDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	switch volType {
	case testpatterns.PreprovisionedPV:
		node := l.node
		// assign this to schedule pod on this node
		config.ClientNodeSelection = e2epod.NodeSelection{Name: node.Name}
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

func (l *localDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	lv, ok := e2evolume.(*localVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to local test volume")
	return &v1.PersistentVolumeSource{
		Local: &v1.LocalVolumeSource{
			Path:   lv.ltr.Path,
			FSType: &fsType,
		},
	}, l.nodeAffinityForNode(lv.ltr.Node)
}

// cleanUpVolumeServer is a wrapper of cleanup function for volume server without secret created by specific CreateStorageServer function.
func cleanUpVolumeServer(f *framework.Framework, serverPod *v1.Pod) {
	cleanUpVolumeServerWithSecret(f, serverPod, nil)
}

func getInlineVolumeZone(f *framework.Framework) string {
	if framework.TestContext.CloudConfig.Zone != "" {
		return framework.TestContext.CloudConfig.Zone
	}
	// if zone is not specified we will randomly pick a zone from schedulable nodes for inline tests
	node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
	framework.ExpectNoError(err)
	zone, ok := node.Labels[v1.LabelZoneFailureDomain]
	if ok {
		return zone
	}
	return ""
}

// cleanUpVolumeServerWithSecret is a wrapper of cleanup function for volume server with secret created by specific CreateStorageServer function.
func cleanUpVolumeServerWithSecret(f *framework.Framework, serverPod *v1.Pod, secret *v1.Secret) {
	cs := f.ClientSet
	ns := f.Namespace

	if secret != nil {
		framework.Logf("Deleting server secret %q...", secret.Name)
		err := cs.CoreV1().Secrets(ns.Name).Delete(context.TODO(), secret.Name, metav1.DeleteOptions{})
		if err != nil {
			framework.Logf("Delete secret failed: %v", err)
		}
	}

	framework.Logf("Deleting server pod %q...", serverPod.Name)
	err := e2epod.DeletePodWithWait(cs, serverPod)
	if err != nil {
		framework.Logf("Server pod delete failed: %v", err)
	}
}
