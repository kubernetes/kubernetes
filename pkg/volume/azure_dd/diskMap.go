/*
Copyright 2016 The Kubernetes Authors.

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

package azure_dd

// TODO
// This map will be exporeted at later stage
// and used by a schedule  predicate to filter VMs at capcity
// of attached disks count
var dataDisksPerVM = map[string]int{
	"Standard_A0": 1,
	"Standard_A1": 2,
	"Standard_A2": 4,
	"Standard_A3": 8,
	"Standard_A4": 16,
	"Standard_A5": 4,
	"Standard_A6": 8,
	"Standard_A7": 16,

	"Standard_A8":  16,
	"Standard_A9":  16,
	"Standard_A10": 16,
	"Standard_A11": 16,

	"Standard_A1_v2":  2,
	"Standard_A2_v2":  4,
	"Standard_A4_v2":  8,
	"Standard_A8_v2":  16,
	"Standard_A2m_v2": 4,
	"Standard_A4m_v2": 8,
	"Standard_A8m_v2": 16,

	"Standard_D1":  2,
	"Standard_D2":  4,
	"Standard_D3":  8,
	"Standard_D4":  16,
	"Standard_D11": 4,
	"Standard_D12": 8,
	"Standard_D13": 16,
	"Standard_D14": 32,

	"Standard_D1_v2":  2,
	"Standard_D2_v2":  4,
	"Standard_D3_v2":  8,
	"Standard_D4_v2":  16,
	"Standard_D5_v2":  32,
	"Standard_D11_v2": 4,
	"Standard_D12_v2": 8,
	"Standard_D13_v2": 16,
	"Standard_D14_v2": 32,
	"Standard_D15_v2": 40,

	"Standard_DS1":  2,
	"Standard_DS2":  4,
	"Standard_DS3":  8,
	"Standard_DS4":  16,
	"Standard_DS11": 4,
	"Standard_DS12": 8,
	"Standard_DS13": 16,
	"Standard_DS14": 32,

	"Standard_DS1_v2":  2,
	"Standard_DS2_v2":  4,
	"Standard_DS3_v2":  8,
	"Standard_DS4_v2":  16,
	"Standard_DS5_v2":  32,
	"Standard_DS11_v2": 4,
	"Standard_DS12_v2": 8,
	"Standard_DS13_v2": 16,
	"Standard_DS14_v2": 32,
	"Standard_DS15_v2": 40,

	"Standard_F1":  2,
	"Standard_F2":  4,
	"Standard_F4":  8,
	"Standard_F8":  16,
	"Standard_F16": 32,

	"Standard_F1s":  2,
	"Standard_F2s":  4,
	"Standard_F4s":  8,
	"Standard_F8s":  16,
	"Standard_F16s": 32,

	"Standard_G1": 4,
	"Standard_G2": 8,
	"Standard_G3": 16,
	"Standard_G4": 32,
	"Standard_G5": 64,

	"Standard_GS1": 4,
	"Standard_GS2": 8,
	"Standard_GS3": 16,
	"Standard_GS4": 32,
	"Standard_GS5": 64,

	"Standard_H8":    16,
	"Standard_H16":   32,
	"Standard_H8m":   16,
	"Standard_H16m":  32,
	"Standard_H16r":  32,
	"Standard_H16mr": 32,
}
