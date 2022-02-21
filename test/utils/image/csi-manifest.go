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

/*
This file is generated and managed by test/e2e/testing-manifests/storage-csi/update-hostpath.sh
Do not edit
*/

package image

const (
	// Offset the CSI images so there is no collision
	CSINone = iota + 500
	Hostpathplugin
	Csiexternalhealthmonitorcontroller
	Csinodedriverregistrar
	Livenessprobe
	Csiattacher
	Csiprovisioner
	Csiresizer
	Csisnapshotter
)

type TestCSIImagesStruct struct {
	HostpathpluginImage                     string
	CsiexternalhealthmonitorcontrollerImage string
	CsinodedriverregistrarImage             string
	LivenessprobeImage                      string
	CsiattacherImage                        string
	CsiprovisionerImage                     string
	CsiresizerImage                         string
	CsisnapshotterImage                     string
}

var TestCSIImages TestCSIImagesStruct

func init() {
	TestCSIImages = TestCSIImagesStruct{
		GetE2EImage(Hostpathplugin),
		GetE2EImage(Csiexternalhealthmonitorcontroller),
		GetE2EImage(Csinodedriverregistrar),
		GetE2EImage(Livenessprobe),
		GetE2EImage(Csiattacher),
		GetE2EImage(Csiprovisioner),
		GetE2EImage(Csiresizer),
		GetE2EImage(Csisnapshotter),
	}
}

func initCSIImageConfigs(list RegistryList, configs map[int]Config) {
	configs[Hostpathplugin] = Config{list.SigStorageRegistry, "hostpathplugin", "v1.7.3"}
	configs[Csiexternalhealthmonitorcontroller] = Config{list.SigStorageRegistry, "csi-external-health-monitor-controller", "v0.4.0"}
	configs[Csinodedriverregistrar] = Config{list.SigStorageRegistry, "csi-node-driver-registrar", "v2.3.0"}
	configs[Livenessprobe] = Config{list.SigStorageRegistry, "livenessprobe", "v2.4.0"}
	configs[Csiattacher] = Config{list.SigStorageRegistry, "csi-attacher", "v3.3.0"}
	configs[Csiprovisioner] = Config{list.SigStorageRegistry, "csi-provisioner", "v3.0.0"}
	configs[Csiresizer] = Config{list.SigStorageRegistry, "csi-resizer", "v1.3.0"}
	configs[Csisnapshotter] = Config{list.SigStorageRegistry, "csi-snapshotter", "v4.2.1"}
}
