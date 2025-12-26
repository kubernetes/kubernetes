//go:build !windows

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
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestGetEtcdCertVolumes(t *testing.T) {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	k8sCertificatesDir := "/etc/kubernetes/pki"
	var tests = []struct {
		name, ca, cert, key string
		vol                 []v1.Volume
		volMount            []v1.VolumeMount
	}{
		{
			name:     "Should ignore files in /etc/ssl/certs",
			ca:       "/etc/ssl/certs/my-etcd-ca.crt",
			cert:     "/etc/ssl/certs/my-etcd.crt",
			key:      "/etc/ssl/certs/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			name:     "Should ignore files in subdirs of /etc/ssl/certs",
			ca:       "/etc/ssl/certs/etcd/my-etcd-ca.crt",
			cert:     "/etc/ssl/certs/etcd/my-etcd.crt",
			key:      "/etc/ssl/certs/etcd/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			name:     "Should ignore files in /etc/pki/ca-trust",
			ca:       "/etc/pki/ca-trust/my-etcd-ca.crt",
			cert:     "/etc/pki/ca-trust/my-etcd.crt",
			key:      "/etc/pki/ca-trust/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			name:     "Should ignore files in Kubernetes PKI directory (and subdirs)",
			ca:       k8sCertificatesDir + "/ca/my-etcd-ca.crt",
			cert:     k8sCertificatesDir + "/my-etcd.crt",
			key:      k8sCertificatesDir + "/my-etcd.key",
			vol:      []v1.Volume{},
			volMount: []v1.VolumeMount{},
		},
		{
			name: "All certs are in the same dir",
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
			name: "One file + two files in separate dirs",
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
			name: "All three files in different directories",
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
			name: "The most top-level dir should be used",
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
			name: "The most top-level dir should be used, regardless of order",
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
		t.Run(rt.name, func(t *testing.T) {
			actualVol, actualVolMount := getEtcdCertVolumes(&kubeadmapi.ExternalEtcd{
				CAFile:   rt.ca,
				CertFile: rt.cert,
				KeyFile:  rt.key,
			}, k8sCertificatesDir)
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
		})
	}
}

func TestGetHostPathVolumesForTheControlPlane(t *testing.T) {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	hostPathFileOrCreate := v1.HostPathFileOrCreate
	controllerManagerConfig := filepath.FromSlash("/etc/kubernetes/controller-manager.conf")
	schedulerConfig := filepath.FromSlash("/etc/kubernetes/scheduler.conf")
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
				Path: controllerManagerConfig,
				Type: &hostPathFileOrCreate,
			},
		},
	}
	volMap[kubeadmconstants.KubeScheduler] = map[string]v1.Volume{}
	volMap[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.Volume{
		Name: "kubeconfig",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: schedulerConfig,
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
		MountPath: controllerManagerConfig,
		ReadOnly:  true,
	}
	volMountMap[kubeadmconstants.KubeScheduler] = map[string]v1.VolumeMount{}
	volMountMap[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.VolumeMount{
		Name:      "kubeconfig",
		MountPath: schedulerConfig,
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
				Path: controllerManagerConfig,
				Type: &hostPathFileOrCreate,
			},
		},
	}
	volMap2[kubeadmconstants.KubeScheduler] = map[string]v1.Volume{}
	volMap2[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.Volume{
		Name: "kubeconfig",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: schedulerConfig,
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
		MountPath: controllerManagerConfig,
		ReadOnly:  true,
	}
	volMountMap2[kubeadmconstants.KubeScheduler] = map[string]v1.VolumeMount{}
	volMountMap2[kubeadmconstants.KubeScheduler]["kubeconfig"] = v1.VolumeMount{
		Name:      "kubeconfig",
		MountPath: schedulerConfig,
		ReadOnly:  true,
	}
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		vol      map[string]map[string]v1.Volume
		volMount map[string]map[string]v1.VolumeMount
	}{
		{
			name: "Should ignore files in /etc/ssl/certs",
			cfg: &kubeadmapi.ClusterConfiguration{
				CertificatesDir: testCertsDir,
				Etcd:            kubeadmapi.Etcd{},
			},
			vol:      volMap,
			volMount: volMountMap,
		},
		{
			name: "Should ignore files in /etc/ssl/certs and in CertificatesDir",
			cfg: &kubeadmapi.ClusterConfiguration{
				CertificatesDir: testCertsDir,
				Etcd: kubeadmapi.Etcd{
					External: &kubeadmapi.ExternalEtcd{
						Endpoints: []string{"foo"},
						CAFile:    "/etc/certs/etcd/my-etcd-ca.crt",
						CertFile:  testCertsDir + "/etcd/my-etcd.crt",
						KeyFile:   "/var/lib/etcd/certs/my-etcd.key",
					},
				},
			},
			vol:      volMap2,
			volMount: volMountMap2,
		},
	}

	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	// set up tmp caCertsExtraVolumePaths for testing
	originalCACertsExtraVolumePaths := caCertsExtraVolumePaths
	caCertsExtraVolumePaths = []string{fmt.Sprintf("%s/etc/pki/ca-trust", tmpdir), fmt.Sprintf("%s/usr/share/ca-certificates", tmpdir)}
	defer func() { caCertsExtraVolumePaths = originalCACertsExtraVolumePaths }()

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			mounts := getHostPathVolumesForTheControlPlane(rt.cfg)

			// Avoid unit test errors when the flexvolume is mounted
			delete(mounts.volumes[kubeadmconstants.KubeControllerManager], flexvolumeDirVolumeName)
			delete(mounts.volumeMounts[kubeadmconstants.KubeControllerManager], flexvolumeDirVolumeName)
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
		})
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
			Name:      "foo-0",
			HostPath:  "/tmp/qux-0",
			MountPath: "/tmp/qux-0",
			ReadOnly:  true,
			PathType:  v1.HostPathFile,
		},
		{
			Name:      "bar-0",
			HostPath:  "/tmp/asd-0",
			MountPath: "/tmp/asd-0",
			ReadOnly:  false,
			PathType:  v1.HostPathDirectory,
		},
		{
			Name:      "foo-1",
			HostPath:  "/tmp/qux-1",
			MountPath: "/tmp/qux-1",
			ReadOnly:  true,
			PathType:  v1.HostPathFileOrCreate,
		},
		{
			Name:      "bar-1",
			HostPath:  "/tmp/asd-1",
			MountPath: "/tmp/asd-1",
			ReadOnly:  false,
			PathType:  v1.HostPathDirectoryOrCreate,
		},
	}
	mounts.AddExtraHostPathMounts("component", hostPathMounts)
	for _, hostMount := range hostPathMounts {
		t.Run(hostMount.Name, func(t *testing.T) {
			volumeName := hostMount.Name
			if _, ok := mounts.volumes["component"][volumeName]; !ok {
				t.Errorf("Expected to find volume %q", volumeName)
			}
			vol := mounts.volumes["component"][volumeName]
			if vol.Name != volumeName {
				t.Errorf("Expected volume name %q", volumeName)
			}
			if vol.HostPath.Path != hostMount.HostPath {
				t.Errorf("Expected host path %q", hostMount.HostPath)
			}
			if _, ok := mounts.volumeMounts["component"][volumeName]; !ok {
				t.Errorf("Expected to find volume mount %q", volumeName)
			}
			if *vol.HostPath.Type != hostMount.PathType {
				t.Errorf("Expected to host path type %q", hostMount.PathType)
			}
			volMount := mounts.volumeMounts["component"][volumeName]
			if volMount.Name != volumeName {
				t.Errorf("Expected volume mount name %q", volumeName)
			}
			if volMount.MountPath != hostMount.MountPath {
				t.Errorf("Expected container path %q", hostMount.MountPath)
			}
			if volMount.ReadOnly != hostMount.ReadOnly {
				t.Errorf("Expected volume readOnly setting %t", hostMount.ReadOnly)
			}
		})
	}
}
