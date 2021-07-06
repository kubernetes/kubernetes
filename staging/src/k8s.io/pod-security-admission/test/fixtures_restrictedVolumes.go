/*
Copyright 2021 The Kubernetes Authors.

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

package test

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/pod-security-admission/api"
)

func init() {
	// volumeType := "ext4"
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "restricted volume types",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			return []*corev1.Pod{
				// pod that has all allowed volume types
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-configmap",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: "volume-configmap-test",
									},
								},
							},
						},
						{
							Name: "volume-downwardapi",
							VolumeSource: corev1.VolumeSource{
								DownwardAPI: &corev1.DownwardAPIVolumeSource{
									Items: []corev1.DownwardAPIVolumeFile{
										{
											Path: "labels",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: "metadata.labels",
											},
										},
									},
								},
							},
						},
						{
							Name: "volume-emptydir",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
						{
							Name: "volume-pvc",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: "test",
								},
							},
						},
						{
							Name: "volume-projects",
							VolumeSource: corev1.VolumeSource{
								Projected: &corev1.ProjectedVolumeSource{
									Sources: []corev1.VolumeProjection{},
								},
							},
						},
						{
							Name: "volume-secret",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName: "test",
								},
							},
						},
						// TODO: Uncomment this volume when CSIInlineVolume hits GA.
						//
						// {
						// 	Name: "volume-csi",
						// 	VolumeSource: corev1.VolumeSource{
						// 		CSI: &corev1.CSIVolumeSource{
						// 			Driver: "inline.storage.kubernetes.io",
						// 			VolumeAttributes: map[string]string{
						// 				"foo": "bar",
						// 			},
						// 		},
						// 	},
						// },
						//
						// TODO: Uncomment this volume when Ephemeral hits GA.
						//
						// {
						// 	Name: "volume-ephemeral",
						// 	VolumeSource: corev1.VolumeSource{
						// 		Ephemeral: &corev1.EphemeralVolumeSource{
						// 			VolumeClaimTemplate: nil, // exercise for reader
						// 		},
						// 	},
						// },
					}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-gcepersistentdisk",
							VolumeSource: corev1.VolumeSource{
								GCEPersistentDisk: &corev1.GCEPersistentDiskVolumeSource{
									PDName: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-awselasticblockstore",
							VolumeSource: corev1.VolumeSource{
								AWSElasticBlockStore: &corev1.AWSElasticBlockStoreVolumeSource{
									VolumeID: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-gitrepo",
							VolumeSource: corev1.VolumeSource{
								GitRepo: &corev1.GitRepoVolumeSource{
									Repository: "github.com/kubernetes/kubernetes",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-nfs",
							VolumeSource: corev1.VolumeSource{
								NFS: &corev1.NFSVolumeSource{
									Server: "testing",
									Path:   "/testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-iscsi",
							VolumeSource: corev1.VolumeSource{
								ISCSI: &corev1.ISCSIVolumeSource{
									TargetPortal: "testing",
									IQN:          "iqn.2001-04.com.example:storage.kube.sys1.xyz",
									Lun:          0,
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-glusterfs",
							VolumeSource: corev1.VolumeSource{
								Glusterfs: &corev1.GlusterfsVolumeSource{
									Path:          "testing",
									EndpointsName: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-rbd",
							VolumeSource: corev1.VolumeSource{
								RBD: &corev1.RBDVolumeSource{
									CephMonitors: []string{"testing"},
									RBDImage:     "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-flexvolume",
							VolumeSource: corev1.VolumeSource{
								FlexVolume: &corev1.FlexVolumeSource{
									Driver: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-cinder",
							VolumeSource: corev1.VolumeSource{
								Cinder: &corev1.CinderVolumeSource{
									VolumeID: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-cephfs",
							VolumeSource: corev1.VolumeSource{
								CephFS: &corev1.CephFSVolumeSource{
									Monitors: []string{"testing"},
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-flocker",
							VolumeSource: corev1.VolumeSource{
								Flocker: &corev1.FlockerVolumeSource{
									DatasetName: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-fc",
							VolumeSource: corev1.VolumeSource{
								FC: &corev1.FCVolumeSource{
									WWIDs: []string{"testing"},
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-azurefile",
							VolumeSource: corev1.VolumeSource{
								AzureFile: &corev1.AzureFileVolumeSource{
									SecretName: "testing",
									ShareName:  "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-vsphere",
							VolumeSource: corev1.VolumeSource{
								VsphereVolume: &corev1.VsphereVirtualDiskVolumeSource{
									VolumePath: "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-quobyte",
							VolumeSource: corev1.VolumeSource{
								Quobyte: &corev1.QuobyteVolumeSource{
									Registry: "localhost:1234",
									Volume:   "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-azuredisk",
							VolumeSource: corev1.VolumeSource{
								AzureDisk: &corev1.AzureDiskVolumeSource{
									DiskName:    "testing",
									DataDiskURI: "https://test.blob.core.windows.net/test/test.vhd",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-portworxvolume",
							VolumeSource: corev1.VolumeSource{
								PortworxVolume: &corev1.PortworxVolumeSource{
									VolumeID: "testing",
									FSType:   "ext4",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-scaleio",
							VolumeSource: corev1.VolumeSource{
								ScaleIO: &corev1.ScaleIOVolumeSource{
									VolumeName: "testing",
									Gateway:    "localhost",
									System:     "testing",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-storageos",
							VolumeSource: corev1.VolumeSource{
								StorageOS: &corev1.StorageOSVolumeSource{
									VolumeName: "test",
								},
							},
						},
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{
						{
							Name: "volume-hostpath",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: "/dev/null",
								},
							},
						},
					}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 0), check: "restrictedVolumes"},
		fixtureData_1_0,
	)
}
