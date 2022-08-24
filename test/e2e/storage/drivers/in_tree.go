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
 * 2) With server or cloud provider outside of Kubernetes (GCE, AWS, Azure, ...)
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
	"time"

	"github.com/onsi/ginkgo/v2"
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
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
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

	driverInfo storageframework.DriverInfo
}

type nfsVolume struct {
	serverHost string
	serverPod  *v1.Pod
	f          *framework.Framework
}

var _ storageframework.TestDriver = &nfsDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &nfsDriver{}
var _ storageframework.InlineVolumeTestDriver = &nfsDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &nfsDriver{}
var _ storageframework.DynamicPVTestDriver = &nfsDriver{}

// InitNFSDriver returns nfsDriver that implements TestDriver interface
func InitNFSDriver() storageframework.TestDriver {
	return &nfsDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "nfs",
			InTreePluginName: "kubernetes.io/nfs",
			MaxFileSize:      storageframework.FileSizeLarge,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			SupportedMountOption: sets.NewString("relatime"),
			RequiredMountOption:  sets.NewString("vers=4.1"),
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapExec:        true,
				storageframework.CapRWX:         true,
				storageframework.CapMultiPODs:   true,
			},
		},
	}
}

func (n *nfsDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &n.driverInfo
}

func (n *nfsDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (n *nfsDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (n *nfsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (n *nfsDriver) GetDynamicProvisionStorageClass(config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := n.externalPluginName
	parameters := map[string]string{"mountOptions": "vers=4.1"}
	ns := config.Framework.Namespace.Name

	return storageframework.GetStorageClass(provisioner, parameters, nil, ns)
}

func (n *nfsDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
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

	return &storageframework.PerTestConfig{
			Driver:    n,
			Prefix:    "nfs",
			Framework: f,
		}, func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(cs, n.externalProvisionerPod))
			clusterRoleBindingName := ns.Name + "--" + "cluster-admin"
			cs.RbacV1().ClusterRoleBindings().Delete(context.TODO(), clusterRoleBindingName, *metav1.NewDeleteOptions(0))
		}
}

func (n *nfsDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	f := config.Framework
	cs := f.ClientSet
	ns := f.Namespace

	// NewNFSServer creates a pod for InlineVolume and PreprovisionedPV,
	// and startExternalProvisioner creates a pod for DynamicPV.
	// Therefore, we need a different PrepareTest logic for volType.
	switch volType {
	case storageframework.InlineVolume:
		fallthrough
	case storageframework.PreprovisionedPV:
		c, serverPod, serverHost := e2evolume.NewNFSServer(cs, ns.Name, []string{})
		config.ServerConfig = &c
		return &nfsVolume{
			serverHost: serverHost,
			serverPod:  serverPod,
			f:          f,
		}
	case storageframework.DynamicPV:
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
	driverInfo storageframework.DriverInfo
}

type glusterVolume struct {
	prefix    string
	serverPod *v1.Pod
	f         *framework.Framework
}

var _ storageframework.TestDriver = &glusterFSDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &glusterFSDriver{}
var _ storageframework.InlineVolumeTestDriver = &glusterFSDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &glusterFSDriver{}

// InitGlusterFSDriver returns glusterFSDriver that implements TestDriver interface
func InitGlusterFSDriver() storageframework.TestDriver {
	return &glusterFSDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "gluster",
			InTreePluginName: "kubernetes.io/glusterfs",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapExec:        true,
				storageframework.CapRWX:         true,
				storageframework.CapMultiPODs:   true,
			},
		},
	}
}

func (g *glusterFSDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &g.driverInfo
}

func (g *glusterFSDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	e2eskipper.SkipUnlessNodeOSDistroIs("gci", "ubuntu", "custom")
}

func (g *glusterFSDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (g *glusterFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (g *glusterFSDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    g,
		Prefix:    "gluster",
		Framework: f,
	}, func() {}
}

func (g *glusterFSDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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
	driverInfo storageframework.DriverInfo
}
type iSCSIVolume struct {
	serverPod *v1.Pod
	serverIP  string
	f         *framework.Framework
	iqn       string
}

var _ storageframework.TestDriver = &iSCSIDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &iSCSIDriver{}
var _ storageframework.InlineVolumeTestDriver = &iSCSIDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &iSCSIDriver{}

// InitISCSIDriver returns iSCSIDriver that implements TestDriver interface
func InitISCSIDriver() storageframework.TestDriver {
	return &iSCSIDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "iscsi",
			InTreePluginName: "kubernetes.io/iscsi",
			FeatureTag:       "[Feature:Volumes]",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			TopologyKeys: []string{v1.LabelHostname},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapFsGroup:     true,
				storageframework.CapBlock:       true,
				storageframework.CapExec:        true,
				storageframework.CapMultiPODs:   true,
				storageframework.CapTopology:    true,
			},
		},
	}
}

func (i *iSCSIDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &i.driverInfo
}

func (i *iSCSIDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (i *iSCSIDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (i *iSCSIDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (i *iSCSIDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    i,
		Prefix:    "iscsi",
		Framework: f,
	}, func() {}
}

func (i *iSCSIDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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
	driverInfo storageframework.DriverInfo
}

type rbdVolume struct {
	serverPod *v1.Pod
	serverIP  string
	secret    *v1.Secret
	f         *framework.Framework
}

var _ storageframework.TestDriver = &rbdDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &rbdDriver{}
var _ storageframework.InlineVolumeTestDriver = &rbdDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &rbdDriver{}

// InitRbdDriver returns rbdDriver that implements TestDriver interface
func InitRbdDriver() storageframework.TestDriver {
	return &rbdDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "rbd",
			InTreePluginName: "kubernetes.io/rbd",
			FeatureTag:       "[Feature:Volumes][Serial]",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapFsGroup:     true,
				storageframework.CapBlock:       true,
				storageframework.CapExec:        true,
				storageframework.CapMultiPODs:   true,
			},
		},
	}
}

func (r *rbdDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &r.driverInfo
}

func (r *rbdDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (r *rbdDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (r *rbdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (r *rbdDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    r,
		Prefix:    "rbd",
		Framework: f,
	}, func() {}
}

func (r *rbdDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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
	driverInfo storageframework.DriverInfo
}

type cephVolume struct {
	serverPod *v1.Pod
	serverIP  string
	secret    *v1.Secret
	f         *framework.Framework
}

var _ storageframework.TestDriver = &cephFSDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &cephFSDriver{}
var _ storageframework.InlineVolumeTestDriver = &cephFSDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &cephFSDriver{}

// InitCephFSDriver returns cephFSDriver that implements TestDriver interface
func InitCephFSDriver() storageframework.TestDriver {
	return &cephFSDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "ceph",
			InTreePluginName: "kubernetes.io/cephfs",
			FeatureTag:       "[Feature:Volumes][Serial]",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapExec:        true,
				storageframework.CapRWX:         true,
				storageframework.CapMultiPODs:   true,
			},
		},
	}
}

func (c *cephFSDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &c.driverInfo
}

func (c *cephFSDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (c *cephFSDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (c *cephFSDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (c *cephFSDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    c,
		Prefix:    "cephfs",
		Framework: f,
	}, func() {}
}

func (c *cephFSDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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
	driverInfo storageframework.DriverInfo
}

var _ storageframework.TestDriver = &hostPathDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &hostPathDriver{}
var _ storageframework.InlineVolumeTestDriver = &hostPathDriver{}

// InitHostPathDriver returns hostPathDriver that implements TestDriver interface
func InitHostPathDriver() storageframework.TestDriver {
	return &hostPathDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "hostPath",
			InTreePluginName: "kubernetes.io/host-path",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			TopologyKeys: []string{v1.LabelHostname},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence:      true,
				storageframework.CapMultiPODs:        true,
				storageframework.CapSingleNodeVolume: true,
				storageframework.CapTopology:         true,
			},
		},
	}
}

func (h *hostPathDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &h.driverInfo
}

func (h *hostPathDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (h *hostPathDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (h *hostPathDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    h,
		Prefix:    "hostpath",
		Framework: f,
	}, func() {}
}

func (h *hostPathDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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
	driverInfo storageframework.DriverInfo
}

type hostPathSymlinkVolume struct {
	targetPath string
	sourcePath string
	prepPod    *v1.Pod
	f          *framework.Framework
}

var _ storageframework.TestDriver = &hostPathSymlinkDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &hostPathSymlinkDriver{}
var _ storageframework.InlineVolumeTestDriver = &hostPathSymlinkDriver{}

// InitHostPathSymlinkDriver returns hostPathSymlinkDriver that implements TestDriver interface
func InitHostPathSymlinkDriver() storageframework.TestDriver {
	return &hostPathSymlinkDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "hostPathSymlink",
			InTreePluginName: "kubernetes.io/host-path",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			TopologyKeys: []string{v1.LabelHostname},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence:      true,
				storageframework.CapMultiPODs:        true,
				storageframework.CapSingleNodeVolume: true,
				storageframework.CapTopology:         true,
			},
		},
	}
}

func (h *hostPathSymlinkDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &h.driverInfo
}

func (h *hostPathSymlinkDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (h *hostPathSymlinkDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (h *hostPathSymlinkDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    h,
		Prefix:    "hostpathsymlink",
		Framework: f,
	}, func() {}
}

func (h *hostPathSymlinkDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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

	err = e2epod.WaitForPodSuccessInNamespaceTimeout(f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStart)
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

	err = e2epod.WaitForPodSuccessInNamespaceTimeout(f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStart)
	framework.ExpectNoError(err, "while waiting for hostPath teardown pod to succeed")

	err = e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting hostPath teardown pod")
}

// emptydir
type emptydirDriver struct {
	driverInfo storageframework.DriverInfo
}

var _ storageframework.TestDriver = &emptydirDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &emptydirDriver{}
var _ storageframework.InlineVolumeTestDriver = &emptydirDriver{}

// InitEmptydirDriver returns emptydirDriver that implements TestDriver interface
func InitEmptydirDriver() storageframework.TestDriver {
	return &emptydirDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "emptydir",
			InTreePluginName: "kubernetes.io/empty-dir",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapExec:             true,
				storageframework.CapSingleNodeVolume: true,
			},
		},
	}
}

func (e *emptydirDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &e.driverInfo
}

func (e *emptydirDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (e *emptydirDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
	// emptydir doesn't support readOnly volume
	if readOnly {
		return nil
	}
	return &v1.VolumeSource{
		EmptyDir: &v1.EmptyDirVolumeSource{},
	}
}

func (e *emptydirDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	return nil
}

func (e *emptydirDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    e,
		Prefix:    "emptydir",
		Framework: f,
	}, func() {}
}

// GCE
type gcePdDriver struct {
	driverInfo storageframework.DriverInfo
}

type gcePdVolume struct {
	volumeName string
}

var _ storageframework.TestDriver = &gcePdDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &gcePdDriver{}
var _ storageframework.InlineVolumeTestDriver = &gcePdDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &gcePdDriver{}
var _ storageframework.DynamicPVTestDriver = &gcePdDriver{}

// InitGcePdDriver returns gcePdDriver that implements TestDriver interface
func InitGcePdDriver() storageframework.TestDriver {
	supportedTypes := sets.NewString(
		"", // Default fsType
		"ext2",
		"ext3",
		"ext4",
		"xfs",
	)
	return &gcePdDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "gcepd",
			InTreePluginName: "kubernetes.io/gce-pd",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType:      supportedTypes,
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			TopologyKeys:         []string{v1.LabelTopologyZone},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence:         true,
				storageframework.CapFsGroup:             true,
				storageframework.CapBlock:               true,
				storageframework.CapExec:                true,
				storageframework.CapMultiPODs:           true,
				storageframework.CapControllerExpansion: true,
				storageframework.CapOfflineExpansion:    true,
				storageframework.CapOnlineExpansion:     true,
				storageframework.CapNodeExpansion:       true,
				// GCE supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				storageframework.CapVolumeLimits: false,
				storageframework.CapTopology:     true,
			},
		},
	}
}

// InitWindowsGcePdDriver returns gcePdDriver running on Windows cluster that implements TestDriver interface
// In current test structure, it first initialize the driver and then set up
// the new framework, so we cannot get the correct OS here and select which file system is supported.
// So here uses a separate Windows in-tree gce pd driver
func InitWindowsGcePdDriver() storageframework.TestDriver {
	supportedTypes := sets.NewString(
		"ntfs",
	)
	return &gcePdDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "windows-gcepd",
			InTreePluginName: "kubernetes.io/gce-pd",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: supportedTypes,
			TopologyKeys:    []string{v1.LabelZoneFailureDomain},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapControllerExpansion: false,
				storageframework.CapPersistence:         true,
				storageframework.CapExec:                true,
				storageframework.CapMultiPODs:           true,
				// GCE supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				storageframework.CapVolumeLimits: false,
				storageframework.CapTopology:     true,
			},
		},
	}
}

func (g *gcePdDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &g.driverInfo
}

func (g *gcePdDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("gce", "gke")
	if pattern.FeatureTag == "[Feature:Windows]" {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	}
}

func (g *gcePdDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (g *gcePdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (g *gcePdDriver) GetDynamicProvisionStorageClass(config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/gce-pd"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return storageframework.GetStorageClass(provisioner, parameters, &delayedBinding, ns)
}

func (g *gcePdDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	config := &storageframework.PerTestConfig{
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

func (g *gcePdDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	zone := getInlineVolumeZone(config.Framework)
	if volType == storageframework.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelTopologyZone: zone,
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
	driverInfo storageframework.DriverInfo
}

type vSphereVolume struct {
	volumePath string
	nodeInfo   *vspheretest.NodeInfo
}

var _ storageframework.TestDriver = &vSphereDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &vSphereDriver{}
var _ storageframework.InlineVolumeTestDriver = &vSphereDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &vSphereDriver{}
var _ storageframework.DynamicPVTestDriver = &vSphereDriver{}

// InitVSphereDriver returns vSphereDriver that implements TestDriver interface
func InitVSphereDriver() storageframework.TestDriver {
	return &vSphereDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "vsphere",
			InTreePluginName: "kubernetes.io/vsphere-volume",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
				"ntfs",
			),
			TopologyKeys: []string{v1.LabelFailureDomainBetaZone},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapFsGroup:     true,
				storageframework.CapExec:        true,
				storageframework.CapMultiPODs:   true,
				storageframework.CapTopology:    true,
			},
		},
	}
}
func (v *vSphereDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &v.driverInfo
}

func (v *vSphereDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("vsphere")
}

func (v *vSphereDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (v *vSphereDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (v *vSphereDriver) GetDynamicProvisionStorageClass(config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/vsphere-volume"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name

	return storageframework.GetStorageClass(provisioner, parameters, nil, ns)
}

func (v *vSphereDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
			Driver:    v,
			Prefix:    "vsphere",
			Framework: f,
		}, func() {
			// Driver Cleanup function
			// Logout each vSphere client connection to prevent session leakage
			nodes := vspheretest.GetReadySchedulableNodeInfos()
			for _, node := range nodes {
				if node.VSphere.Client != nil {
					node.VSphere.Client.Logout(context.TODO())
				}
			}
		}
}

func (v *vSphereDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
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
	driverInfo storageframework.DriverInfo
}

type azureDiskVolume struct {
	volumeName string
}

var _ storageframework.TestDriver = &azureDiskDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &azureDiskDriver{}
var _ storageframework.InlineVolumeTestDriver = &azureDiskDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &azureDiskDriver{}
var _ storageframework.DynamicPVTestDriver = &azureDiskDriver{}
var _ storageframework.CustomTimeoutsTestDriver = &azureDiskDriver{}

// InitAzureDiskDriver returns azureDiskDriver that implements TestDriver interface
func InitAzureDiskDriver() storageframework.TestDriver {
	return &azureDiskDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "azure-disk",
			InTreePluginName: "kubernetes.io/azure-disk",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
				"xfs",
			),
			TopologyKeys: []string{v1.LabelFailureDomainBetaZone},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence: true,
				storageframework.CapFsGroup:     true,
				storageframework.CapBlock:       true,
				storageframework.CapExec:        true,
				storageframework.CapMultiPODs:   true,
				// Azure supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				storageframework.CapVolumeLimits: false,
				storageframework.CapTopology:     true,
			},
		},
	}
}

func (a *azureDiskDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &a.driverInfo
}

func (a *azureDiskDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("azure")
}

func (a *azureDiskDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (a *azureDiskDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (a *azureDiskDriver) GetDynamicProvisionStorageClass(config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/azure-disk"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return storageframework.GetStorageClass(provisioner, parameters, &delayedBinding, ns)
}

func (a *azureDiskDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    a,
		Prefix:    "azure",
		Framework: f,
	}, func() {}
}

func (a *azureDiskDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	ginkgo.By("creating a test azure disk volume")
	zone := getInlineVolumeZone(config.Framework)
	if volType == storageframework.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelFailureDomainBetaZone: zone,
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
	driverInfo storageframework.DriverInfo
}

type awsVolume struct {
	volumeName string
}

var _ storageframework.TestDriver = &awsDriver{}

var _ storageframework.PreprovisionedVolumeTestDriver = &awsDriver{}
var _ storageframework.InlineVolumeTestDriver = &awsDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &awsDriver{}
var _ storageframework.DynamicPVTestDriver = &awsDriver{}

// InitAwsDriver returns awsDriver that implements TestDriver interface
func InitAwsDriver() storageframework.TestDriver {
	return &awsDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "aws",
			InTreePluginName: "kubernetes.io/aws-ebs",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
				"xfs",
				"ntfs",
			),
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			TopologyKeys:         []string{v1.LabelTopologyZone},
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence:         true,
				storageframework.CapFsGroup:             true,
				storageframework.CapBlock:               true,
				storageframework.CapExec:                true,
				storageframework.CapMultiPODs:           true,
				storageframework.CapControllerExpansion: true,
				storageframework.CapNodeExpansion:       true,
				storageframework.CapOfflineExpansion:    true,
				storageframework.CapOnlineExpansion:     true,
				// AWS supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				storageframework.CapVolumeLimits: false,
				storageframework.CapTopology:     true,
			},
		},
	}
}

func (a *awsDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &a.driverInfo
}

func (a *awsDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("aws")
}

func (a *awsDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
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

func (a *awsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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

func (a *awsDriver) GetDynamicProvisionStorageClass(config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/aws-ebs"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return storageframework.GetStorageClass(provisioner, parameters, &delayedBinding, ns)
}

func (a *awsDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	config := &storageframework.PerTestConfig{
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

func (a *awsDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	zone := getInlineVolumeZone(config.Framework)
	if volType == storageframework.InlineVolume || volType == storageframework.PreprovisionedPV {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelTopologyZone: zone,
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
	driverInfo storageframework.DriverInfo
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
	defaultLocalVolumeCapabilities = map[storageframework.Capability]bool{
		storageframework.CapPersistence:      true,
		storageframework.CapFsGroup:          true,
		storageframework.CapBlock:            false,
		storageframework.CapExec:             true,
		storageframework.CapMultiPODs:        true,
		storageframework.CapSingleNodeVolume: true,
	}
	localVolumeCapabitilies = map[utils.LocalVolumeType]map[storageframework.Capability]bool{
		utils.LocalVolumeBlock: {
			storageframework.CapPersistence:      true,
			storageframework.CapFsGroup:          true,
			storageframework.CapBlock:            true,
			storageframework.CapExec:             true,
			storageframework.CapMultiPODs:        true,
			storageframework.CapSingleNodeVolume: true,
		},
	}
	// fstype
	defaultLocalVolumeSupportedFsTypes = sets.NewString("")
	localVolumeSupportedFsTypes        = map[utils.LocalVolumeType]sets.String{
		utils.LocalVolumeBlock: sets.NewString(
			"", // Default fsType
			"ext4",
			//"xfs", disabled see issue https://github.com/kubernetes/kubernetes/issues/74095
		),
	}
	// max file size
	defaultLocalVolumeMaxFileSize = storageframework.FileSizeSmall
	localVolumeMaxFileSizes       = map[utils.LocalVolumeType]int64{}
)

var _ storageframework.TestDriver = &localDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &localDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &localDriver{}

// InitLocalDriverWithVolumeType initializes the local driver based on the volume type.
func InitLocalDriverWithVolumeType(volumeType utils.LocalVolumeType) func() storageframework.TestDriver {
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
	return func() storageframework.TestDriver {
		// custom tag to distinguish from tests of other volume types
		featureTag := fmt.Sprintf("[LocalVolumeType: %s]", volumeType)
		// For GCE Local SSD volumes, we must run serially
		if volumeType == utils.LocalVolumeGCELocalSSD {
			featureTag += " [Serial]"
		}
		return &localDriver{
			driverInfo: storageframework.DriverInfo{
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

func (l *localDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &l.driverInfo
}

func (l *localDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
}

func (l *localDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
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

	return &storageframework.PerTestConfig{
			Driver:              l,
			Prefix:              "local",
			Framework:           f,
			ClientNodeSelection: e2epod.NodeSelection{Name: l.node.Name},
		}, func() {
			l.hostExec.Cleanup()
		}
}

func (l *localDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	switch volType {
	case storageframework.PreprovisionedPV:
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

func (l *localDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
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
	zone, ok := node.Labels[v1.LabelFailureDomainBetaZone]
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

// Azure File
type azureFileDriver struct {
	driverInfo storageframework.DriverInfo
}

type azureFileVolume struct {
	accountName     string
	shareName       string
	secretName      string
	secretNamespace string
}

var _ storageframework.TestDriver = &azureFileDriver{}
var _ storageframework.PreprovisionedVolumeTestDriver = &azureFileDriver{}
var _ storageframework.InlineVolumeTestDriver = &azureFileDriver{}
var _ storageframework.PreprovisionedPVTestDriver = &azureFileDriver{}
var _ storageframework.DynamicPVTestDriver = &azureFileDriver{}

// InitAzureFileDriver returns azureFileDriver that implements TestDriver interface
func InitAzureFileDriver() storageframework.TestDriver {
	return &azureFileDriver{
		driverInfo: storageframework.DriverInfo{
			Name:             "azure-file",
			InTreePluginName: "kubernetes.io/azure-file",
			MaxFileSize:      storageframework.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[storageframework.Capability]bool{
				storageframework.CapPersistence:         true,
				storageframework.CapExec:                true,
				storageframework.CapRWX:                 true,
				storageframework.CapMultiPODs:           true,
				storageframework.CapControllerExpansion: true,
				storageframework.CapNodeExpansion:       true,
			},
		},
	}
}

func (a *azureFileDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &a.driverInfo
}

func (a *azureFileDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("azure")
}

func (a *azureFileDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) *v1.VolumeSource {
	av, ok := e2evolume.(*azureFileVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Azure test volume")
	volSource := v1.VolumeSource{
		AzureFile: &v1.AzureFileVolumeSource{
			SecretName: av.secretName,
			ShareName:  av.shareName,
			ReadOnly:   readOnly,
		},
	}
	return &volSource
}

func (a *azureFileDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume storageframework.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	av, ok := e2evolume.(*azureFileVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Azure test volume")
	pvSource := v1.PersistentVolumeSource{
		AzureFile: &v1.AzureFilePersistentVolumeSource{
			SecretName:      av.secretName,
			ShareName:       av.shareName,
			SecretNamespace: &av.secretNamespace,
			ReadOnly:        readOnly,
		},
	}
	return &pvSource, nil
}

func (a *azureFileDriver) GetDynamicProvisionStorageClass(config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/azure-file"
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name
	immediateBinding := storagev1.VolumeBindingImmediate
	return storageframework.GetStorageClass(provisioner, parameters, &immediateBinding, ns)
}

func (a *azureFileDriver) PrepareTest(f *framework.Framework) (*storageframework.PerTestConfig, func()) {
	return &storageframework.PerTestConfig{
		Driver:    a,
		Prefix:    "azure-file",
		Framework: f,
	}, func() {}
}

func (a *azureFileDriver) CreateVolume(config *storageframework.PerTestConfig, volType storageframework.TestVolType) storageframework.TestVolume {
	ginkgo.By("creating a test azure file volume")
	accountName, accountKey, shareName, err := e2epv.CreateShare()
	framework.ExpectNoError(err)

	secretName := "azure-storage-account-" + accountName + "-secret"
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: config.Framework.Namespace.Name,
			Name:      secretName,
		},

		Data: map[string][]byte{
			"azurestorageaccountname": []byte(accountName),
			"azurestorageaccountkey":  []byte(accountKey),
		},
		Type: "Opaque",
	}

	_, err = config.Framework.ClientSet.CoreV1().Secrets(config.Framework.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return &azureFileVolume{
		accountName:     accountName,
		shareName:       shareName,
		secretName:      secretName,
		secretNamespace: config.Framework.Namespace.Name,
	}
}

func (v *azureFileVolume) DeleteVolume() {
	err := e2epv.DeleteShare(v.accountName, v.shareName)
	framework.ExpectNoError(err)
}

func (a *azureDiskDriver) GetTimeouts() *framework.TimeoutContext {
	return &framework.TimeoutContext{
		PodStart:  time.Minute * 15,
		PodDelete: time.Minute * 15,
		PVDelete:  time.Minute * 20,
	}
}
