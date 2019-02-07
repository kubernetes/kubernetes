/*
Copyright 2019 The Kubernetes Authors.

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

// Package constants holds all exported constants of the csi package, available
// for consumption by external packages.
//
// This is needed to break some import cycles, e.g:
//   pkg/volume/csi and pkg/volume need access to the CSIPluginName, however
//   pkg/volume/csi depends on and imports pkg/volume. Therefore the later can
//   not import the former, this would lead to an import cycle.
//   By splitting out the exported constants into their own package, both
//   pkg/volume/csi and pkg/volume can import the package and share those
//   exported constants.
//
// Any other or new exported constant should also go into this package, or not
// be exported after all.
package constants

const (
	// CSIPluginName is the name of the in-tree CSI Plugin
	CSIPluginName = "kubernetes.io/csi"
)
