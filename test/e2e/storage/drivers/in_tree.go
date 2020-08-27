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
	"os/exec"
	"strconv"
	"strings"
	"time"

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
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	vspheretest "k8s.io/kubernetes/test/e2e/storage/vsphere"
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
				Min: "1Gi",
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
				Min: "1Gi",
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

	nameSpaceName := fmt.Sprintf("%s/%s", ns.Name, name)

	framework.Logf("Deleting Gluster endpoints %s...", nameSpaceName)
	err := cs.CoreV1().Endpoints(ns.Name).Delete(context.TODO(), name, metav1.DeleteOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			framework.Failf("Gluster deleting endpoint %s failed: %v", nameSpaceName, err)
		}
		framework.Logf("Gluster endpoints %q not found, assuming deleted", nameSpaceName)
	}

	framework.Logf("Deleting Gluster service %s...", nameSpaceName)
	err = cs.CoreV1().Services(ns.Name).Delete(context.TODO(), name, metav1.DeleteOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			framework.Failf("Gluster deleting service %s failed: %v", nameSpaceName, err)
		}
		framework.Logf("Gluster service %q not found, assuming deleted", nameSpaceName)
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
				Min: "1Gi",
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
				Min: "1Gi",
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
			Name:             "cinder",
			InTreePluginName: "kubernetes.io/cinder",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
			),
			TopologyKeys: []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapBlock:       true,
				// Cinder supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (c *cinderDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &c.driverInfo
}

func (c *cinderDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("openstack")
}

func (c *cinderDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	cv, ok := e2evolume.(*cinderVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Cinder test volume")

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

func (c *cinderDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	cv, ok := e2evolume.(*cinderVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Cinder test volume")

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
	ginkgo.By("creating a test Cinder volume")
	output, err := exec.Command("cinder", "create", "--display-name="+volumeName, "1").CombinedOutput()
	outputString := string(output[:])
	framework.Logf("cinder output:\n%s", outputString)
	framework.ExpectNoError(err)

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
	framework.ExpectNotEqual(volumeID, "")
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
			Name:             "gcepd",
			InTreePluginName: "kubernetes.io/gce-pd",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType:      supportedTypes,
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			TopologyKeys:         []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:         true,
				testsuites.CapFsGroup:             true,
				testsuites.CapBlock:               true,
				testsuites.CapExec:                true,
				testsuites.CapMultiPODs:           true,
				testsuites.CapControllerExpansion: true,
				testsuites.CapNodeExpansion:       true,
				// GCE supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (g *gcePdDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("gce", "gke")
	if pattern.FeatureTag == "[sig-windows]" {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	}
}

func (g *gcePdDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	gv, ok := e2evolume.(*gcePdVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to GCE PD test volume")
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

func (g *gcePdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	gv, ok := e2evolume.(*gcePdVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to GCE PD test volume")
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
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (g *gcePdDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	config := &testsuites.PerTestConfig{
		Driver:    g,
		Prefix:    "gcepd",
		Framework: f,
	}

	if framework.NodeOSDistroIs("windows") {
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				"kubernetes.io/os": "windows",
			},
		}
	}
	return config, func() {}

}

func (g *gcePdDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	zone := getInlineVolumeZone(config.Framework)
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelZoneFailureDomain: zone,
			},
		}
	}
	ginkgo.By("creating a test gce pd volume")
	vname, err := e2epv.CreatePDWithRetryAndZone(zone)
	framework.ExpectNoError(err)
	return &gcePdVolume{
		volumeName: vname,
	}
}

func (v *gcePdVolume) DeleteVolume() {
	e2epv.DeletePDWithRetry(v.volumeName)
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
			Name:             "vsphere",
			InTreePluginName: "kubernetes.io/vsphere-volume",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			TopologyKeys: []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
				testsuites.CapTopology:    true,
			},
		},
	}
}
func (v *vSphereDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &v.driverInfo
}

func (v *vSphereDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("vsphere")
}

func (v *vSphereDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	vsv, ok := e2evolume.(*vSphereVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to vSphere test volume")

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

func (v *vSphereDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	vsv, ok := e2evolume.(*vSphereVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to vSphere test volume")

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
	framework.ExpectNoError(err)
	return &vSphereVolume{
		volumePath: volumePath,
		nodeInfo:   nodeInfo,
	}
}

func (v *vSphereVolume) DeleteVolume() {
	v.nodeInfo.VSphere.DeleteVolume(v.volumePath, v.nodeInfo.DataCenterRef)
}

// Azure Disk
type azureDiskDriver struct {
	driverInfo testsuites.DriverInfo
}

type azureDiskVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &azureDiskDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &azureDiskDriver{}
var _ testsuites.InlineVolumeTestDriver = &azureDiskDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &azureDiskDriver{}
var _ testsuites.DynamicPVTestDriver = &azureDiskDriver{}

// InitAzureDiskDriver returns azureDiskDriver that implements TestDriver interface
func InitAzureDiskDriver() testsuites.TestDriver {
	return &azureDiskDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "azure-disk",
			InTreePluginName: "kubernetes.io/azure-disk",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
				"ext4",
				"xfs",
			),
			TopologyKeys: []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
				// Azure supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (a *azureDiskDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &a.driverInfo
}

func (a *azureDiskDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("azure")
}

func (a *azureDiskDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	av, ok := e2evolume.(*azureDiskVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Azure test volume")
	diskName := av.volumeName[(strings.LastIndex(av.volumeName, "/") + 1):]

	kind := v1.AzureManagedDisk
	volSource := v1.VolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: av.volumeName,
			Kind:        &kind,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		volSource.AzureDisk.FSType = &fsType
	}
	return &volSource
}

func (a *azureDiskDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	av, ok := e2evolume.(*azureDiskVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Azure test volume")

	diskName := av.volumeName[(strings.LastIndex(av.volumeName, "/") + 1):]

	kind := v1.AzureManagedDisk
	pvSource := v1.PersistentVolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: av.volumeName,
			Kind:        &kind,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		pvSource.AzureDisk.FSType = &fsType
	}
	return &pvSource, nil
}

func (a *azureDiskDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/azure-disk"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (a *azureDiskDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    a,
		Prefix:    "azure",
		Framework: f,
	}, func() {}
}

func (a *azureDiskDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	ginkgo.By("creating a test azure disk volume")
	zone := getInlineVolumeZone(config.Framework)
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelZoneFailureDomain: zone,
			},
		}
	}
	volumeName, err := e2epv.CreatePDWithRetryAndZone(zone)
	framework.ExpectNoError(err)
	return &azureDiskVolume{
		volumeName: volumeName,
	}
}

func (v *azureDiskVolume) DeleteVolume() {
	e2epv.DeletePDWithRetry(v.volumeName)
}

// AWS
type awsDriver struct {
	driverInfo testsuites.DriverInfo
}

type awsVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &awsDriver{}

var _ testsuites.PreprovisionedVolumeTestDriver = &awsDriver{}
var _ testsuites.InlineVolumeTestDriver = &awsDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &awsDriver{}
var _ testsuites.DynamicPVTestDriver = &awsDriver{}

// InitAwsDriver returns awsDriver that implements TestDriver interface
func InitAwsDriver() testsuites.TestDriver {
	return &awsDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "aws",
			InTreePluginName: "kubernetes.io/aws-ebs",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
				"ntfs",
			),
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			TopologyKeys:         []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:         true,
				testsuites.CapFsGroup:             true,
				testsuites.CapBlock:               true,
				testsuites.CapExec:                true,
				testsuites.CapMultiPODs:           true,
				testsuites.CapControllerExpansion: true,
				testsuites.CapNodeExpansion:       true,
				// AWS supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (a *awsDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &a.driverInfo
}

func (a *awsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("aws")
}

func (a *awsDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	av, ok := e2evolume.(*awsVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to AWS test volume")
	volSource := v1.VolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: av.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.AWSElasticBlockStore.FSType = fsType
	}
	return &volSource
}

func (a *awsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	av, ok := e2evolume.(*awsVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to AWS test volume")
	pvSource := v1.PersistentVolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: av.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.AWSElasticBlockStore.FSType = fsType
	}
	return &pvSource, nil
}

func (a *awsDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/aws-ebs"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (a *awsDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	config := &testsuites.PerTestConfig{
		Driver:    a,
		Prefix:    "aws",
		Framework: f,
	}

	if framework.NodeOSDistroIs("windows") {
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				"kubernetes.io/os": "windows",
			},
		}
	}
	return config, func() {}
}

func (a *awsDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	zone := getInlineVolumeZone(config.Framework)
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelZoneFailureDomain: zone,
			},
		}
	}
	ginkgo.By("creating a test aws volume")
	vname, err := e2epv.CreatePDWithRetryAndZone(zone)
	framework.ExpectNoError(err)
	return &awsVolume{
		volumeName: vname,
	}
}

func (v *awsVolume) DeleteVolume() {
	e2epv.DeletePDWithRetry(v.volumeName)
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
