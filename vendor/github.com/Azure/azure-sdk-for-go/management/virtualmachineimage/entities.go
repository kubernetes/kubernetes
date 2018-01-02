// +build go1.7

package virtualmachineimage

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/management"
	vmdisk "github.com/Azure/azure-sdk-for-go/management/virtualmachinedisk"
)

// Client is used to perform operations on Azure VM Images.
type Client struct {
	management.Client
}

type ListVirtualMachineImagesResponse struct {
	VMImages []VMImage `xml:"VMImage"`
}

type VMImage struct {
	Name                   string                  // Specifies the name of the image.
	Label                  string                  // Specifies an identifier for the image.
	Category               string                  // Specifies the repository classification of the image. All user images have the category User.
	Description            string                  // Specifies the description of the image.
	OSDiskConfiguration    OSDiskConfiguration     // Specifies configuration information for the operating system disk that is associated with the image.
	DataDiskConfigurations []DataDiskConfiguration `xml:">DataDiskConfiguration"` // Specifies configuration information for the data disks that are associated with the image. A VM Image might not have data disks associated with it.
	ServiceName            string                  // Specifies the name of the cloud service that contained the Virtual Machine from which the image was created.
	DeploymentName         string                  // Specifies the name of the deployment that contained the Virtual Machine from which the image was created.
	RoleName               string                  // Specifies the name of the Virtual Machine from which the image was created.
	Location               string                  // Specifies the geo-location in which the media is located. The Location value is derived from the storage account that contains the blob in which the media is located. If the storage account belongs to an affinity group the value is NULL and the element is not displayed in the response.
	AffinityGroup          string                  // Specifies the affinity group in which the media is located. The AffinityGroup value is derived from the storage account that contains the blob in which the media is located. If the storage account does not belong to an affinity group the value is NULL and the element is not displayed in the response.
	CreatedTime            string                  // Specifies the time that the image was created.
	ModifiedTime           string                  // Specifies the time that the image was last updated.
	Language               string                  // Specifies the language of the image.
	ImageFamily            string                  // Specifies a value that can be used to group VM Images.
	RecommendedVMSize      string                  // Optional. Specifies the size to use for the Virtual Machine that is created from the VM Image.
	IsPremium              string                  // Indicates whether the image contains software or associated services that will incur charges above the core price for the virtual machine. For additional details, see the PricingDetailLink element.
	Eula                   string                  // Specifies the End User License Agreement that is associated with the image. The value for this element is a string, but it is recommended that the value be a URL that points to a EULA.
	IconURI                string                  `xml:"IconUri"`      // Specifies the URI to the icon that is displayed for the image in the Management Portal.
	SmallIconURI           string                  `xml:"SmallIconUri"` // Specifies the URI to the small icon that is displayed for the image in the Management Portal.
	PrivacyURI             string                  `xml:"PrivacyUri"`   // Specifies the URI that points to a document that contains the privacy policy related to the image.
	PublishedDate          string                  // Specifies the date when the image was added to the image repository.
}

type OSState string

const (
	OSStateGeneralized OSState = "Generalized"
	OSStateSpecialized OSState = "Specialized"
)

type IOType string

const (
	IOTypeProvisioned IOType = "Provisioned"
	IOTypeStandard    IOType = "Standard"
)

// OSDiskConfiguration specifies configuration information for the operating
// system disk that is associated with the image.
type OSDiskConfiguration struct {
	Name            string                 // Specifies the name of the operating system disk.
	HostCaching     vmdisk.HostCachingType // Specifies the caching behavior of the operating system disk.
	OSState         OSState                // Specifies the state of the operating system in the image.
	OS              string                 // Specifies the operating system type of the image.
	MediaLink       string                 // Specifies the location of the blob in Azure storage. The blob location belongs to a storage account in the subscription specified by the <subscription-id> value in the operation call.
	LogicalSizeInGB float64                // Specifies the size, in GB, of the operating system disk.
	IOType          IOType                 // Identifies the type of the storage account for the backing VHD. If the backing VHD is in an Provisioned Storage account, “Provisioned” is returned otherwise “Standard” is returned.
}

// DataDiskConfiguration specifies configuration information for the data disks
// that are associated with the image.
type DataDiskConfiguration struct {
	Name            string                 // Specifies the name of the data disk.
	HostCaching     vmdisk.HostCachingType // Specifies the caching behavior of the data disk.
	Lun             string                 // Specifies the Logical Unit Number (LUN) for the data disk.
	MediaLink       string                 // Specifies the location of the blob in Azure storage. The blob location belongs to a storage account in the subscription specified by the <subscription-id> value in the operation call.
	LogicalSizeInGB float64                // Specifies the size, in GB, of the data disk.
	IOType          IOType                 // Identifies the type of the storage account for the backing VHD. If the backing VHD is in an Provisioned Storage account, “Provisioned” is returned otherwise “Standard” is returned.
}

type CaptureRoleAsVMImageOperation struct {
	XMLName       xml.Name `xml:"http://schemas.microsoft.com/windowsazure CaptureRoleAsVMImageOperation"`
	OperationType string   //CaptureRoleAsVMImageOperation
	OSState       OSState
	VMImageName   string
	VMImageLabel  string
	CaptureParameters
}

type CaptureParameters struct {
	Description       string `xml:",omitempty"`
	Language          string `xml:",omitempty"`
	ImageFamily       string `xml:",omitempty"`
	RecommendedVMSize string `xml:",omitempty"`
}
