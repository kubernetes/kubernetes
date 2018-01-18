/*
Copyright 2017 The Kubernetes Authors.

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

package controlplane

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestGetEtcdCertVolumes(t *testing.T) {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	k8sCertifcatesDir := "/etc/kubernetes/pki"
	var tests = []struct {
		ca, cert, key string
		vol           []v1.Volume
		volMount      []v1.VolumeMount
	}{
		{
			// Should ignore files in /etc/ssl/certs
			ca:       "/etc/ssl/certs/my-etcd-ca.crt",
			cert:     "/etc/ssl/certs/my-etcd.crt",
			key:      "/etc/ssl/certs/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			// Should ignore files in subdirs of /etc/ssl/certs
			ca:       "/etc/ssl/certs/etcd/my-etcd-ca.crt",
			cert:     "/etc/ssl/certs/etcd/my-etcd.crt",
			key:      "/etc/ssl/certs/etcd/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			// Should ignore files in /etc/pki
			ca:       "/etc/pki/my-etcd-ca.crt",
			cert:     "/etc/pki/my-etcd.crt",
			key:      "/etc/pki/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			// Should ignore files in Kubernetes PKI directory (and subdirs)
			ca:       k8sCertifcatesDir + "/ca/my-etcd-ca.crt",
			cert:     k8sCertifcatesDir + "/my-etcd.crt",
			key:      k8sCertifcatesDir + "/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			// All in the same dir
			ca:   "/var/lib/certs/etcd/my-etcd-ca.crt",
			cert: "/var/lib/certs/etcd/my-etcd.crt",
			key:  "/var/lib/certs/etcd/my-etcd.key",
			vol: []v1.Volume{
				{
					Name: "etcd-certs-0",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
			},
			volMount: []v1.VolumeMount{
				{
					Name:      "etcd-certs-0",
					MountPath: "/var/lib/certs/etcd",
					ReadOnly:  true,
				},
			},
		},
		{
			// One file + two files in separate dirs
			ca:   "/etc/certs/etcd/my-etcd-ca.crt",
			cert: "/var/lib/certs/etcd/my-etcd.crt",
			key:  "/var/lib/certs/etcd/my-etcd.key",
			vol: []v1.Volume{
				{
					Name: "etcd-certs-0",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/etc/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
				{
					Name: "etcd-certs-1",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
			},
			volMount: []v1.VolumeMount{
				{
					Name:      "etcd-certs-0",
					MountPath: "/etc/certs/etcd",
					ReadOnly:  true,
				},
				{
					Name:      "etcd-certs-1",
					MountPath: "/var/lib/certs/etcd",
					ReadOnly:  true,
				},
			},
		},
		{
			// All three files in different directories
			ca:   "/etc/certs/etcd/my-etcd-ca.crt",
			cert: "/var/lib/certs/etcd/my-etcd.crt",
			key:  "/var/lib/certs/private/my-etcd.key",
			vol: []v1.Volume{
				{
					Name: "etcd-certs-0",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/etc/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
				{
					Name: "etcd-certs-1",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
				{
					Name: "etcd-certs-2",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/lib/certs/private",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
			},
			volMount: []v1.VolumeMount{
				{
					Name:      "etcd-certs-0",
					MountPath: "/etc/certs/etcd",
					ReadOnly:  true,
				},
				{
					Name:      "etcd-certs-1",
					MountPath: "/var/lib/certs/etcd",
					ReadOnly:  true,
				},
				{
					Name:      "etcd-certs-2",
					MountPath: "/var/lib/certs/private",
					ReadOnly:  true,
				},
			},
		},
		{
			// The most top-level dir should be used
			ca:   "/etc/certs/etcd/my-etcd-ca.crt",
			cert: "/etc/certs/etcd/serving/my-etcd.crt",
			key:  "/etc/certs/etcd/serving/my-etcd.key",
			vol: []v1.Volume{
				{
					Name: "etcd-certs-0",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/etc/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
			},
			volMount: []v1.VolumeMount{
				{
					Name:      "etcd-certs-0",
					MountPath: "/etc/certs/etcd",
					ReadOnly:  true,
				},
			},
		},
		{
			// The most top-level dir should be used, regardless of order
			ca:   "/etc/certs/etcd/ca/my-etcd-ca.crt",
			cert: "/etc/certs/etcd/my-etcd.crt",
			key:  "/etc/certs/etcd/my-etcd.key",
			vol: []v1.Volume{
				{
					Name: "etcd-certs-0",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/etc/certs/etcd",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
			},
			volMount: []v1.VolumeMount{
				{
					Name:      "etcd-certs-0",
					MountPath: "/etc/certs/etcd",
					ReadOnly:  true,
				},
			},
		},
	}

	for _, rt := range tests {
		actualVol, actualVolMount := getEtcdCertVolumes(kubeadmapi.Etcd{
			CAFile:   rt.ca,
			CertFile: rt.cert,
			KeyFile:  rt.key,
		}, k8sCertifcatesDir)
		if !reflect.DeepEqual(actualVol, rt.vol) {
			t.Errorf(
				"failed getEtcdCertVolumes:\n\texpected: %v\n\t  actual: %v",
				rt.vol,
				actualVol,
			)
		}
		if !reflect.DeepEqual(actualVolMount, rt.volMount) {
			t.Errorf(
				"failed getEtcdCertVolumes:\n\texpected: %v\n\t  actual: %v",
				rt.volMount,
				actualVolMount,
			)
		}
	}
}

func TestGetHostPathVolumesForTheControlPlane(t *testing.T) {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	hostPathFileOrCreate := v1.HostPathFileOrCreate
	volMap := make(map[string]map[string]v1.Volume)
	volMap[kubeadmconstants.KubeAPIServer] = map[string]v1.Volume{}
	volMap[kubeadmconstants.KubeAPIServer]["k8s-certs"] = v1.Volume{
		Name: "k8s-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: testCertsDir,
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap[kubeadmconstants.KubeAPIServer]["ca-certs"] = v1.Volume{
		Name: "ca-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/ssl/certs",
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap[kubeadmconstants.KubeControllerManager] = map[string]v1.Volume{}
	volMap[kubeadmconstants.KubeControllerManager]["k8s-certs"] = v1.Volume{
		Name: "k8s-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: testCertsDir,
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap[kubeadmconstants.KubeControllerManager]["ca-certs"] = v1.Volume{
		Name: "ca-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/ssl/certs",
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap[kubeadmconstants.KubeControllerManager]["kubeconfig"] = v1.Volume{
		Name: "kubeconfig",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/kubernetes/controller-manager.conf",
				Type: &hostPathFileOrCreate,
			},
		},
	}
	volMap[kubeadmconstants.KubeScheduler] = map[string]v1.Volume{}
	volMap[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.Volume{
		Name: "kubeconfig",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/kubernetes/scheduler.conf",
				Type: &hostPathFileOrCreate,
			},
		},
	}
	volMountMap := make(map[string]map[string]v1.VolumeMount)
	volMountMap[kubeadmconstants.KubeAPIServer] = map[string]v1.VolumeMount{}
	volMountMap[kubeadmconstants.KubeAPIServer]["k8s-certs"] = v1.VolumeMount{
		Name:      "k8s-certs",
		MountPath: testCertsDir,
		ReadOnly:  true,
	}
	volMountMap[kubeadmconstants.KubeAPIServer]["ca-certs"] = v1.VolumeMount{
		Name:      "ca-certs",
		MountPath: "/etc/ssl/certs",
		ReadOnly:  true,
	}
	volMountMap[kubeadmconstants.KubeControllerManager] = map[string]v1.VolumeMount{}
	volMountMap[kubeadmconstants.KubeControllerManager]["k8s-certs"] = v1.VolumeMount{
		Name:      "k8s-certs",
		MountPath: testCertsDir,
		ReadOnly:  true,
	}
	volMountMap[kubeadmconstants.KubeControllerManager]["ca-certs"] = v1.VolumeMount{
		Name:      "ca-certs",
		MountPath: "/etc/ssl/certs",
		ReadOnly:  true,
	}
	volMountMap[kubeadmconstants.KubeControllerManager]["kubeconfig"] = v1.VolumeMount{
		Name:      "kubeconfig",
		MountPath: "/etc/kubernetes/controller-manager.conf",
		ReadOnly:  true,
	}
	volMountMap[kubeadmconstants.KubeScheduler] = map[string]v1.VolumeMount{}
	volMountMap[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.VolumeMount{
		Name:      "kubeconfig",
		MountPath: "/etc/kubernetes/scheduler.conf",
		ReadOnly:  true,
	}

	volMap2 := make(map[string]map[string]v1.Volume)
	volMap2[kubeadmconstants.KubeAPIServer] = map[string]v1.Volume{}
	volMap2[kubeadmconstants.KubeAPIServer]["k8s-certs"] = v1.Volume{
		Name: "k8s-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: testCertsDir,
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeAPIServer]["ca-certs"] = v1.Volume{
		Name: "ca-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/ssl/certs",
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeAPIServer]["etcd-certs-0"] = v1.Volume{
		Name: "etcd-certs-0",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/certs/etcd",
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeAPIServer]["etcd-certs-1"] = v1.Volume{
		Name: "etcd-certs-1",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/var/lib/etcd/certs",
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeControllerManager] = map[string]v1.Volume{}
	volMap2[kubeadmconstants.KubeControllerManager]["k8s-certs"] = v1.Volume{
		Name: "k8s-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: testCertsDir,
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeControllerManager]["ca-certs"] = v1.Volume{
		Name: "ca-certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/ssl/certs",
				Type: &hostPathDirectoryOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeControllerManager]["kubeconfig"] = v1.Volume{
		Name: "kubeconfig",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/kubernetes/controller-manager.conf",
				Type: &hostPathFileOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeScheduler] = map[string]v1.Volume{}
	volMap2[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.Volume{
		Name: "kubeconfig",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/etc/kubernetes/scheduler.conf",
				Type: &hostPathFileOrCreate,
			},
		},
	}
	volMountMap2 := make(map[string]map[string]v1.VolumeMount)
	volMountMap2[kubeadmconstants.KubeAPIServer] = map[string]v1.VolumeMount{}
	volMountMap2[kubeadmconstants.KubeAPIServer]["k8s-certs"] = v1.VolumeMount{
		Name:      "k8s-certs",
		MountPath: testCertsDir,
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeAPIServer]["ca-certs"] = v1.VolumeMount{
		Name:      "ca-certs",
		MountPath: "/etc/ssl/certs",
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeAPIServer]["etcd-certs-0"] = v1.VolumeMount{
		Name:      "etcd-certs-0",
		MountPath: "/etc/certs/etcd",
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeAPIServer]["etcd-certs-1"] = v1.VolumeMount{
		Name:      "etcd-certs-1",
		MountPath: "/var/lib/etcd/certs",
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeControllerManager] = map[string]v1.VolumeMount{}
	volMountMap2[kubeadmconstants.KubeControllerManager]["k8s-certs"] = v1.VolumeMount{
		Name:      "k8s-certs",
		MountPath: testCertsDir,
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeControllerManager]["ca-certs"] = v1.VolumeMount{
		Name:      "ca-certs",
		MountPath: "/etc/ssl/certs",
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeControllerManager]["kubeconfig"] = v1.VolumeMount{
		Name:      "kubeconfig",
		MountPath: "/etc/kubernetes/controller-manager.conf",
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeScheduler] = map[string]v1.VolumeMount{}
	volMountMap2[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.VolumeMount{
		Name:      "kubeconfig",
		MountPath: "/etc/kubernetes/scheduler.conf",
		ReadOnly:  true,
	}
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		vol      map[string]map[string]v1.Volume
		volMount map[string]map[string]v1.VolumeMount
	}{
		{
			// Should ignore files in /etc/ssl/certs
			cfg: &kubeadmapi.MasterConfiguration{
				CertificatesDir: testCertsDir,
				Etcd:            kubeadmapi.Etcd{},
			},
			vol:      volMap,
			volMount: volMountMap,
		},
		{
			// Should ignore files in /etc/ssl/certs and in CertificatesDir
			cfg: &kubeadmapi.MasterConfiguration{
				CertificatesDir: testCertsDir,
				Etcd: kubeadmapi.Etcd{
					Endpoints: []string{"foo"},
					CAFile:    "/etc/certs/etcd/my-etcd-ca.crt",
					CertFile:  testCertsDir + "/etcd/my-etcd.crt",
					KeyFile:   "/var/lib/etcd/certs/my-etcd.key",
				},
			},
			vol:      volMap2,
			volMount: volMountMap2,
		},
	}

	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	// set up tmp caCertsPkiVolumePath for testing
	caCertsPkiVolumePath = fmt.Sprintf("%s/etc/pki", tmpdir)
	defer func() { caCertsPkiVolumePath = "/etc/pki" }()

	for _, rt := range tests {
		mounts := getHostPathVolumesForTheControlPlane(rt.cfg)

		// Avoid unit test errors when the flexvolume is mounted
		if _, ok := mounts.volumes[kubeadmconstants.KubeControllerManager][flexvolumeDirVolumeName]; ok {
			delete(mounts.volumes[kubeadmconstants.KubeControllerManager], flexvolumeDirVolumeName)
		}
		if _, ok := mounts.volumeMounts[kubeadmconstants.KubeControllerManager][flexvolumeDirVolumeName]; ok {
			delete(mounts.volumeMounts[kubeadmconstants.KubeControllerManager], flexvolumeDirVolumeName)
		}
		if _, ok := mounts.volumeMounts[kubeadmconstants.KubeControllerManager][cloudConfigVolumeName]; ok {
			delete(mounts.volumeMounts[kubeadmconstants.KubeControllerManager], cloudConfigVolumeName)
		}
		if !reflect.DeepEqual(mounts.volumes, rt.vol) {
			t.Errorf(
				"failed getHostPathVolumesForTheControlPlane:\n\texpected: %v\n\t  actual: %v",
				rt.vol,
				mounts.volumes,
			)
		}
		if !reflect.DeepEqual(mounts.volumeMounts, rt.volMount) {
			t.Errorf(
				"failed getHostPathVolumesForTheControlPlane:\n\texpected: %v\n\t  actual: %v",
				rt.volMount,
				mounts.volumeMounts,
			)
		}
	}
}

func TestAddExtraHostPathMounts(t *testing.T) {
	mounts := newControlPlaneHostPathMounts()
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	hostPathFileOrCreate := v1.HostPathFileOrCreate
	vols := []v1.Volume{
		{
			Name: "foo",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/tmp/foo",
					Type: &hostPathDirectoryOrCreate,
				},
			},
		},
		{
			Name: "bar",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/tmp/bar",
					Type: &hostPathFileOrCreate,
				},
			},
		},
	}
	volMounts := []v1.VolumeMount{
		{
			Name:      "foo",
			MountPath: "/tmp/foo",
			ReadOnly:  true,
		},
		{
			Name:      "bar",
			MountPath: "/tmp/bar",
			ReadOnly:  true,
		},
	}
	mounts.AddHostPathMounts("component", vols, volMounts)
	hostPathMounts := []kubeadmapi.HostPathMount{
		{
			Name:      "foo",
			HostPath:  "/tmp/qux",
			MountPath: "/tmp/qux",
		},
	}
	mounts.AddExtraHostPathMounts("component", hostPathMounts, true, &hostPathDirectoryOrCreate)
	if _, ok := mounts.volumes["component"]["foo"]; !ok {
		t.Errorf("Expected to find volume %q", "foo")
	}
	vol, _ := mounts.volumes["component"]["foo"]
	if vol.Name != "foo" {
		t.Errorf("Expected volume name %q", "foo")
	}
	if vol.HostPath.Path != "/tmp/qux" {
		t.Errorf("Expected host path %q", "/tmp/qux")
	}
	if _, ok := mounts.volumeMounts["component"]["foo"]; !ok {
		t.Errorf("Expected to find volume mount %q", "foo")
	}
	volMount, _ := mounts.volumeMounts["component"]["foo"]
	if volMount.Name != "foo" {
		t.Errorf("Expected volume mount name %q", "foo")
	}
	if volMount.MountPath != "/tmp/qux" {
		t.Errorf("Expected container path %q", "/tmp/qux")
	}
}
