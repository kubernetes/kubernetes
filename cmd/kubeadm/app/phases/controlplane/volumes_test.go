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
		})
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
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		vol      map[string][]v1.Volume
		volMount map[string][]v1.VolumeMount
	}{
		{
			// Should ignore files in /etc/ssl/certs
			cfg: &kubeadmapi.MasterConfiguration{
				CertificatesDir: testCertsDir,
				Etcd:            kubeadmapi.Etcd{},
			},
			vol: map[string][]v1.Volume{
				kubeadmconstants.KubeAPIServer: {
					{
						Name: "k8s-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: testCertsDir,
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
				},
				kubeadmconstants.KubeControllerManager: {
					{
						Name: "k8s-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: testCertsDir,
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "kubeconfig",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/controller-manager.conf",
								Type: &hostPathFileOrCreate,
							},
						},
					},
					{
						Name: "flexvolume-dir",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
				},
				kubeadmconstants.KubeScheduler: {
					{
						Name: "kubeconfig",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/scheduler.conf",
								Type: &hostPathFileOrCreate,
							},
						},
					},
				},
			},
			volMount: map[string][]v1.VolumeMount{
				kubeadmconstants.KubeAPIServer: {
					{
						Name:      "k8s-certs",
						MountPath: testCertsDir,
						ReadOnly:  true,
					},
					{
						Name:      "ca-certs",
						MountPath: "/etc/ssl/certs",
						ReadOnly:  true,
					},
				},
				kubeadmconstants.KubeControllerManager: {
					{
						Name:      "k8s-certs",
						MountPath: testCertsDir,
						ReadOnly:  true,
					},
					{
						Name:      "ca-certs",
						MountPath: "/etc/ssl/certs",
						ReadOnly:  true,
					},
					{
						Name:      "kubeconfig",
						MountPath: "/etc/kubernetes/controller-manager.conf",
						ReadOnly:  true,
					},
					{
						Name:      "flexvolume-dir",
						MountPath: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec",
						ReadOnly:  false,
					},
				},
				kubeadmconstants.KubeScheduler: {
					{
						Name:      "kubeconfig",
						MountPath: "/etc/kubernetes/scheduler.conf",
						ReadOnly:  true,
					},
				},
			},
		},
		{
			// Should ignore files in /etc/ssl/certs
			cfg: &kubeadmapi.MasterConfiguration{
				CertificatesDir: testCertsDir,
				Etcd: kubeadmapi.Etcd{
					Endpoints: []string{"foo"},
					CAFile:    "/etc/certs/etcd/my-etcd-ca.crt",
					CertFile:  "/var/lib/certs/etcd/my-etcd.crt",
					KeyFile:   "/var/lib/certs/etcd/my-etcd.key",
				},
			},
			vol: map[string][]v1.Volume{
				kubeadmconstants.KubeAPIServer: {
					{
						Name: "k8s-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: testCertsDir,
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
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
				kubeadmconstants.KubeControllerManager: {
					{
						Name: "k8s-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: testCertsDir,
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "ca-certs",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/ssl/certs",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
					{
						Name: "kubeconfig",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/controller-manager.conf",
								Type: &hostPathFileOrCreate,
							},
						},
					},
					{
						Name: "flexvolume-dir",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec",
								Type: &hostPathDirectoryOrCreate,
							},
						},
					},
				},
				kubeadmconstants.KubeScheduler: {
					{
						Name: "kubeconfig",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/kubernetes/scheduler.conf",
								Type: &hostPathFileOrCreate,
							},
						},
					},
				},
			},
			volMount: map[string][]v1.VolumeMount{
				kubeadmconstants.KubeAPIServer: {
					{
						Name:      "k8s-certs",
						MountPath: testCertsDir,
						ReadOnly:  true,
					},
					{
						Name:      "ca-certs",
						MountPath: "/etc/ssl/certs",
						ReadOnly:  true,
					},
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
				kubeadmconstants.KubeControllerManager: {
					{
						Name:      "k8s-certs",
						MountPath: testCertsDir,
						ReadOnly:  true,
					},
					{
						Name:      "ca-certs",
						MountPath: "/etc/ssl/certs",
						ReadOnly:  true,
					},
					{
						Name:      "kubeconfig",
						MountPath: "/etc/kubernetes/controller-manager.conf",
						ReadOnly:  true,
					},
					{
						Name:      "flexvolume-dir",
						MountPath: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec",
						ReadOnly:  false,
					},
				},
				kubeadmconstants.KubeScheduler: {
					{
						Name:      "kubeconfig",
						MountPath: "/etc/kubernetes/scheduler.conf",
						ReadOnly:  true,
					},
				},
			},
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
