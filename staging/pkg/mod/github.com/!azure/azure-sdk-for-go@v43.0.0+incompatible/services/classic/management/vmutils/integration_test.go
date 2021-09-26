// +build go1.7

package vmutils

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
	"encoding/base64"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
	"github.com/Azure/azure-sdk-for-go/services/classic/management/hostedservice"
	"github.com/Azure/azure-sdk-for-go/services/classic/management/location"
	"github.com/Azure/azure-sdk-for-go/services/classic/management/osimage"
	storage "github.com/Azure/azure-sdk-for-go/services/classic/management/storageservice"
	"github.com/Azure/azure-sdk-for-go/services/classic/management/testutils"
	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
	vmimage "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachineimage"
)

func TestDeployPlatformImage(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmname := GenerateName()
	sa := GetTestStorageAccount(t, client)
	location := sa.StorageServiceProperties.Location

	role := NewVMConfiguration(vmname, "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		GetLinuxTestImage(t, client).Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/sdktest/%s.vhd", sa.ServiceName, vmname),
		GenerateName())
	ConfigureForLinux(&role, "myvm", "azureuser", GeneratePassword())
	ConfigureWithPublicSSH(&role)

	testRoleConfiguration(t, client, role, location)
}

func TestDeployPlatformWindowsImage(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmname := GenerateName()
	sa := GetTestStorageAccount(t, client)
	location := sa.StorageServiceProperties.Location

	role := NewVMConfiguration(vmname, "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		GetWindowsTestImage(t, client).Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/sdktest/%s.vhd", sa.ServiceName, vmname),
		GenerateName())
	ConfigureForWindows(&role, vmname, "azureuser", GeneratePassword(), true, "")
	ConfigureWinRMOverHTTP(&role)
	ConfigureWinRMOverHTTPS(&role, "")

	testRoleConfiguration(t, client, role, location)
}

func TestVMImageList(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmic := vmimage.NewClient(client)
	il, _ := vmic.ListVirtualMachineImages(vmimage.ListParameters{})
	for _, im := range il.VMImages {
		t.Logf("%s -%s", im.Name, im.Description)
	}
}

func TestDeployPlatformOSImageCaptureRedeploy(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmname := GenerateName()
	sa := GetTestStorageAccount(t, client)
	location := sa.StorageServiceProperties.Location

	role := NewVMConfiguration(vmname, "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		GetLinuxTestImage(t, client).Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/sdktest/%s.vhd", sa.ServiceName, vmname),
		GenerateName())
	ConfigureForLinux(&role, "myvm", "azureuser", GeneratePassword())
	ConfigureWithPublicSSH(&role)

	t.Logf("Deploying VM: %s", vmname)
	createRoleConfiguration(t, client, role, location)

	t.Logf("Wait for deployment to enter running state")
	vmc := vm.NewClient(client)
	status := vm.DeploymentStatusDeploying
	for status != vm.DeploymentStatusRunning {
		deployment, err := vmc.GetDeployment(vmname, vmname)
		if err != nil {
			t.Error(err)
			break
		}
		status = deployment.Status
	}

	t.Logf("Shutting down VM: %s", vmname)
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.ShutdownRole(vmname, vmname, vmname, vm.PostShutdownActionStopped)
	}); err != nil {
		t.Error(err)
	}

	if err := WaitForDeploymentPowerState(client, vmname, vmname, vm.PowerStateStopped); err != nil {
		t.Fatal(err)
	}

	imagename := GenerateName()
	t.Logf("Capturing OSImage: %s", imagename)
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.CaptureRole(vmname, vmname, vmname, imagename, imagename, nil)
	}); err != nil {
		t.Error(err)
	}

	im := GetUserOSImage(t, client, imagename)
	t.Logf("Found image: %+v", im)

	newvmname := GenerateName()
	role = NewVMConfiguration(newvmname, "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		im.Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/sdktest/%s.vhd", sa.ServiceName, newvmname),
		GenerateName())
	ConfigureForLinux(&role, newvmname, "azureuser", GeneratePassword())
	ConfigureWithPublicSSH(&role)

	t.Logf("Deploying new VM from freshly captured OS image: %s", newvmname)
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.CreateDeployment(role, vmname, vm.CreateDeploymentOptions{})
	}); err != nil {
		t.Error(err)
	}

	deleteHostedService(t, client, vmname)
}

func TestDeployPlatformVMImageCaptureRedeploy(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmname := GenerateName()
	sa := GetTestStorageAccount(t, client)
	location := sa.StorageServiceProperties.Location

	role := NewVMConfiguration(vmname, "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		GetLinuxTestImage(t, client).Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/sdktest/%s.vhd", sa.ServiceName, vmname),
		GenerateName())
	ConfigureForLinux(&role, "myvm", "azureuser", GeneratePassword())
	ConfigureWithPublicSSH(&role)

	t.Logf("Deploying VM: %s", vmname)
	createRoleConfiguration(t, client, role, location)

	t.Logf("Wait for deployment to enter running state")
	vmc := vm.NewClient(client)
	status := vm.DeploymentStatusDeploying
	for status != vm.DeploymentStatusRunning {
		deployment, err := vmc.GetDeployment(vmname, vmname)
		if err != nil {
			t.Error(err)
			break
		}
		status = deployment.Status
	}

	t.Logf("Shutting down VM: %s", vmname)
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.ShutdownRole(vmname, vmname, vmname, vm.PostShutdownActionStopped)
	}); err != nil {
		t.Error(err)
	}

	if err := WaitForDeploymentInstanceStatus(client, vmname, vmname, vm.InstanceStatusStoppedVM); err != nil {
		t.Fatal(err)
	}

	imagename := GenerateName()
	t.Logf("Capturing VMImage: %s", imagename)
	if err := Await(client, func() (management.OperationID, error) {
		return vmimage.NewClient(client).Capture(vmname, vmname, vmname, imagename, imagename, vmimage.OSStateGeneralized, vmimage.CaptureParameters{})
	}); err != nil {
		t.Error(err)
	}

	im := GetUserVMImage(t, client, imagename)
	t.Logf("Found image: %+v", im)

	newvmname := GenerateName()
	role = NewVMConfiguration(newvmname, "Standard_D3")
	ConfigureDeploymentFromUserVMImage(&role, im.Name)
	ConfigureForLinux(&role, newvmname, "azureuser", GeneratePassword())
	ConfigureWithPublicSSH(&role)

	t.Logf("Deploying new VM from freshly captured VM image: %s", newvmname)
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.CreateDeployment(role, vmname, vm.CreateDeploymentOptions{})
	}); err != nil {
		t.Error(err)
	}

	deleteHostedService(t, client, vmname)
}

func TestDeployFromPublishedVmImage(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmname := GenerateName()
	sa := GetTestStorageAccount(t, client)
	location := sa.StorageServiceProperties.Location

	im := GetVMImage(t, client, func(im vmimage.VMImage) bool {
		return im.Name ==
			"fb83b3509582419d99629ce476bcb5c8__SQL-Server-2014-RTM-12.0.2430.0-OLTP-ENU-Win2012R2-cy14su11"
	})

	role := NewVMConfiguration(vmname, "Standard_D4")
	ConfigureDeploymentFromPublishedVMImage(&role, im.Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/%s", sa.ServiceName, vmname), false)
	ConfigureForWindows(&role, vmname, "azureuser", GeneratePassword(), true, "")
	ConfigureWithPublicSSH(&role)

	testRoleConfiguration(t, client, role, location)
}

func TestRoleStateOperations(t *testing.T) {
	client := testutils.GetTestClient(t)
	vmname := GenerateName()
	sa := GetTestStorageAccount(t, client)
	location := sa.StorageServiceProperties.Location

	role := NewVMConfiguration(vmname, "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		GetLinuxTestImage(t, client).Name,
		fmt.Sprintf("http://%s.blob.core.windows.net/sdktest/%s.vhd", sa.ServiceName, vmname),
		GenerateName())
	ConfigureForLinux(&role, "myvm", "azureuser", GeneratePassword())

	createRoleConfiguration(t, client, role, location)

	vmc := vm.NewClient(client)
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.ShutdownRole(vmname, vmname, vmname, vm.PostShutdownActionStopped)
	}); err != nil {
		t.Error(err)
	}
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.StartRole(vmname, vmname, vmname)
	}); err != nil {
		t.Error(err)
	}
	if err := Await(client, func() (management.OperationID, error) {
		return vmc.RestartRole(vmname, vmname, vmname)
	}); err != nil {
		t.Error(err)
	}

	deleteHostedService(t, client, vmname)
}

func testRoleConfiguration(t *testing.T, client management.Client, role vm.Role, location string) {
	createRoleConfiguration(t, client, role, location)

	deleteHostedService(t, client, role.RoleName)
}

func createRoleConfiguration(t *testing.T, client management.Client, role vm.Role, location string) {
	vmc := vm.NewClient(client)
	hsc := hostedservice.NewClient(client)
	vmname := role.RoleName

	if err := hsc.CreateHostedService(hostedservice.CreateHostedServiceParameters{
		ServiceName: vmname, Location: location,
		Label: base64.StdEncoding.EncodeToString([]byte(vmname))}); err != nil {
		t.Error(err)
	}

	if err := Await(client, func() (management.OperationID, error) {
		return vmc.CreateDeployment(role, vmname, vm.CreateDeploymentOptions{})
	}); err != nil {
		t.Error(err)
	}
}

func deleteHostedService(t *testing.T, client management.Client, vmname string) {
	t.Logf("Deleting hosted service: %s", vmname)
	if err := Await(client, func() (management.OperationID, error) {
		return hostedservice.NewClient(client).DeleteHostedService(vmname, true)
	}); err != nil {
		t.Error(err)
	}
}

// === utility funcs ===

func GetTestStorageAccount(t *testing.T, client management.Client) storage.StorageServiceResponse {
	t.Log("Retrieving storage account")
	sc := storage.NewClient(client)
	var sa storage.StorageServiceResponse
	ssl, err := sc.ListStorageServices()
	if err != nil {
		t.Fatal(err)
	}
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	if len(ssl.StorageServices) == 0 {
		t.Log("No storage accounts found, creating a new one")
		lc := location.NewClient(client)
		ll, err := lc.ListLocations()
		if err != nil {
			t.Fatal(err)
		}
		loc := ll.Locations[rnd.Intn(len(ll.Locations))].Name

		t.Logf("Location for new storage account: %s", loc)
		name := GenerateName()
		op, err := sc.CreateStorageService(storage.StorageAccountCreateParameters{
			ServiceName: name,
			Label:       base64.StdEncoding.EncodeToString([]byte(name)),
			Location:    loc,
			AccountType: storage.AccountTypeStandardLRS})
		if err != nil {
			t.Fatal(err)
		}
		if err := client.WaitForOperation(op, nil); err != nil {
			t.Fatal(err)
		}
		sa, err = sc.GetStorageService(name)
		if err != nil {
			t.Fatal(err)
		}
	} else {

		sa = ssl.StorageServices[rnd.Intn(len(ssl.StorageServices))]
	}

	t.Logf("Selected storage account '%s' in location '%s'",
		sa.ServiceName, sa.StorageServiceProperties.Location)

	return sa
}

func GetLinuxTestImage(t *testing.T, client management.Client) osimage.OSImage {
	return GetOSImage(t, client, func(im osimage.OSImage) bool {
		return im.Category == "Public" && im.ImageFamily == "Ubuntu Server 14.04 LTS"
	})
}

func GetWindowsTestImage(t *testing.T, client management.Client) osimage.OSImage {
	return GetOSImage(t, client, func(im osimage.OSImage) bool {
		return im.Category == "Public" && im.ImageFamily == "Windows Server 2012 R2 Datacenter"
	})
}

func GetUserOSImage(t *testing.T, client management.Client, name string) osimage.OSImage {
	return GetOSImage(t, client, func(im osimage.OSImage) bool {
		return im.Category == "User" && im.Name == name
	})
}

func GetOSImage(
	t *testing.T,
	client management.Client,
	filter func(osimage.OSImage) bool) osimage.OSImage {
	t.Log("Selecting OS image")
	osc := osimage.NewClient(client)
	allimages, err := osc.ListOSImages()
	if err != nil {
		t.Fatal(err)
	}
	filtered := []osimage.OSImage{}
	for _, im := range allimages.OSImages {
		if filter(im) {
			filtered = append(filtered, im)
		}
	}
	if len(filtered) == 0 {
		t.Fatal("Filter too restrictive, no images left?")
	}

	image := filtered[0]
	for _, im := range filtered {
		if im.PublishedDate > image.PublishedDate {
			image = im
		}
	}

	t.Logf("Selecting image '%s'", image.Name)
	return image
}

func GetUserVMImage(t *testing.T, client management.Client, name string) vmimage.VMImage {
	return GetVMImage(t, client, func(im vmimage.VMImage) bool {
		return im.Category == "User" && im.Name == name
	})
}

func GetVMImage(
	t *testing.T,
	client management.Client,
	filter func(vmimage.VMImage) bool) vmimage.VMImage {
	t.Log("Selecting VM image")
	allimages, err := vmimage.NewClient(client).ListVirtualMachineImages(vmimage.ListParameters{})
	if err != nil {
		t.Fatal(err)
	}
	filtered := []vmimage.VMImage{}
	for _, im := range allimages.VMImages {
		if filter(im) {
			filtered = append(filtered, im)
		}
	}
	if len(filtered) == 0 {
		t.Fatal("Filter too restrictive, no images left?")
	}

	image := filtered[0]
	for _, im := range filtered {
		if im.PublishedDate > image.PublishedDate {
			image = im
		}
	}

	t.Logf("Selecting image '%s'", image.Name)
	return image
}

func GenerateName() string {
	from := "1234567890abcdefghijklmnopqrstuvwxyz"
	return "sdk" + GenerateString(12, from)
}

func GeneratePassword() string {
	pw := GenerateString(20, "1234567890") +
		GenerateString(20, "abcdefghijklmnopqrstuvwxyz") +
		GenerateString(20, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	i := rnd.Intn(len(pw)-2) + 1

	pw = string(append([]uint8(pw[i:]), pw[:i-1]...))

	return pw
}

func GenerateString(length int, from string) string {
	str := ""
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	for len(str) < length {
		str += string(from[rnd.Intn(len(from))])
	}
	return str
}

type asyncFunc func() (operationId management.OperationID, err error)

func Await(client management.Client, async asyncFunc) error {
	requestID, err := async()
	if err != nil {
		return err
	}
	return client.WaitForOperation(requestID, nil)
}
