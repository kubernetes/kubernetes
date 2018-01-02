// test
package azure

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/json"
	"testing"
)

func TestEnvironmentFromName(t *testing.T) {
	name := "azurechinacloud"
	if env, _ := EnvironmentFromName(name); env != ChinaCloud {
		t.Errorf("Expected to get ChinaCloud for %q", name)
	}

	name = "AzureChinaCloud"
	if env, _ := EnvironmentFromName(name); env != ChinaCloud {
		t.Errorf("Expected to get ChinaCloud for %q", name)
	}

	name = "azuregermancloud"
	if env, _ := EnvironmentFromName(name); env != GermanCloud {
		t.Errorf("Expected to get GermanCloud for %q", name)
	}

	name = "AzureGermanCloud"
	if env, _ := EnvironmentFromName(name); env != GermanCloud {
		t.Errorf("Expected to get GermanCloud for %q", name)
	}

	name = "azurepubliccloud"
	if env, _ := EnvironmentFromName(name); env != PublicCloud {
		t.Errorf("Expected to get PublicCloud for %q", name)
	}

	name = "AzurePublicCloud"
	if env, _ := EnvironmentFromName(name); env != PublicCloud {
		t.Errorf("Expected to get PublicCloud for %q", name)
	}

	name = "azureusgovernmentcloud"
	if env, _ := EnvironmentFromName(name); env != USGovernmentCloud {
		t.Errorf("Expected to get USGovernmentCloud for %q", name)
	}

	name = "AzureUSGovernmentCloud"
	if env, _ := EnvironmentFromName(name); env != USGovernmentCloud {
		t.Errorf("Expected to get USGovernmentCloud for %q", name)
	}

	name = "thisisnotarealcloudenv"
	if _, err := EnvironmentFromName(name); err == nil {
		t.Errorf("Expected to get an error for %q", name)
	}
}

func TestDeserializeEnvironment(t *testing.T) {
	env := `{
		"name": "--name--",
		"ActiveDirectoryEndpoint": "--active-directory-endpoint--",
		"galleryEndpoint": "--gallery-endpoint--",
		"graphEndpoint": "--graph-endpoint--",
		"keyVaultDNSSuffix": "--key-vault-dns-suffix--",
		"keyVaultEndpoint": "--key-vault-endpoint--",
		"managementPortalURL": "--management-portal-url--",
		"publishSettingsURL": "--publish-settings-url--",
		"resourceManagerEndpoint": "--resource-manager-endpoint--",
		"serviceBusEndpointSuffix": "--service-bus-endpoint-suffix--",
		"serviceManagementEndpoint": "--service-management-endpoint--",
		"sqlDatabaseDNSSuffix": "--sql-database-dns-suffix--",
		"storageEndpointSuffix": "--storage-endpoint-suffix--",
		"trafficManagerDNSSuffix": "--traffic-manager-dns-suffix--",
		"serviceManagementVMDNSSuffix": "--asm-vm-dns-suffix--",
		"resourceManagerVMDNSSuffix": "--arm-vm-dns-suffix--",
		"containerRegistryDNSSuffix": "--container-registry-dns-suffix--"
	}`

	testSubject := Environment{}
	err := json.Unmarshal([]byte(env), &testSubject)
	if err != nil {
		t.Fatalf("failed to unmarshal: %s", err)
	}

	if "--name--" != testSubject.Name {
		t.Errorf("Expected Name to be \"--name--\", but got %q", testSubject.Name)
	}
	if "--management-portal-url--" != testSubject.ManagementPortalURL {
		t.Errorf("Expected ManagementPortalURL to be \"--management-portal-url--\", but got %q", testSubject.ManagementPortalURL)
	}
	if "--publish-settings-url--" != testSubject.PublishSettingsURL {
		t.Errorf("Expected PublishSettingsURL to be \"--publish-settings-url--\", but got %q", testSubject.PublishSettingsURL)
	}
	if "--service-management-endpoint--" != testSubject.ServiceManagementEndpoint {
		t.Errorf("Expected ServiceManagementEndpoint to be \"--service-management-endpoint--\", but got %q", testSubject.ServiceManagementEndpoint)
	}
	if "--resource-manager-endpoint--" != testSubject.ResourceManagerEndpoint {
		t.Errorf("Expected ResourceManagerEndpoint to be \"--resource-manager-endpoint--\", but got %q", testSubject.ResourceManagerEndpoint)
	}
	if "--active-directory-endpoint--" != testSubject.ActiveDirectoryEndpoint {
		t.Errorf("Expected ActiveDirectoryEndpoint to be \"--active-directory-endpoint--\", but got %q", testSubject.ActiveDirectoryEndpoint)
	}
	if "--gallery-endpoint--" != testSubject.GalleryEndpoint {
		t.Errorf("Expected GalleryEndpoint to be \"--gallery-endpoint--\", but got %q", testSubject.GalleryEndpoint)
	}
	if "--key-vault-endpoint--" != testSubject.KeyVaultEndpoint {
		t.Errorf("Expected KeyVaultEndpoint to be \"--key-vault-endpoint--\", but got %q", testSubject.KeyVaultEndpoint)
	}
	if "--graph-endpoint--" != testSubject.GraphEndpoint {
		t.Errorf("Expected GraphEndpoint to be \"--graph-endpoint--\", but got %q", testSubject.GraphEndpoint)
	}
	if "--storage-endpoint-suffix--" != testSubject.StorageEndpointSuffix {
		t.Errorf("Expected StorageEndpointSuffix to be \"--storage-endpoint-suffix--\", but got %q", testSubject.StorageEndpointSuffix)
	}
	if "--sql-database-dns-suffix--" != testSubject.SQLDatabaseDNSSuffix {
		t.Errorf("Expected sql-database-dns-suffix to be \"--sql-database-dns-suffix--\", but got %q", testSubject.SQLDatabaseDNSSuffix)
	}
	if "--key-vault-dns-suffix--" != testSubject.KeyVaultDNSSuffix {
		t.Errorf("Expected StorageEndpointSuffix to be \"--key-vault-dns-suffix--\", but got %q", testSubject.KeyVaultDNSSuffix)
	}
	if "--service-bus-endpoint-suffix--" != testSubject.ServiceBusEndpointSuffix {
		t.Errorf("Expected StorageEndpointSuffix to be \"--service-bus-endpoint-suffix--\", but got %q", testSubject.ServiceBusEndpointSuffix)
	}
	if "--asm-vm-dns-suffix--" != testSubject.ServiceManagementVMDNSSuffix {
		t.Errorf("Expected ServiceManagementVMDNSSuffix to be \"--asm-vm-dns-suffix--\", but got %q", testSubject.ServiceManagementVMDNSSuffix)
	}
	if "--arm-vm-dns-suffix--" != testSubject.ResourceManagerVMDNSSuffix {
		t.Errorf("Expected ResourceManagerVMDNSSuffix to be \"--arm-vm-dns-suffix--\", but got %q", testSubject.ResourceManagerVMDNSSuffix)
	}
	if "--container-registry-dns-suffix--" != testSubject.ContainerRegistryDNSSuffix {
		t.Errorf("Expected ContainerRegistryDNSSuffix to be \"--container-registry-dns-suffix--\", but got %q", testSubject.ContainerRegistryDNSSuffix)
	}
}

func TestRoundTripSerialization(t *testing.T) {
	env := Environment{
		Name:                         "--unit-test--",
		ManagementPortalURL:          "--management-portal-url",
		PublishSettingsURL:           "--publish-settings-url--",
		ServiceManagementEndpoint:    "--service-management-endpoint--",
		ResourceManagerEndpoint:      "--resource-management-endpoint--",
		ActiveDirectoryEndpoint:      "--active-directory-endpoint--",
		GalleryEndpoint:              "--gallery-endpoint--",
		KeyVaultEndpoint:             "--key-vault--endpoint--",
		GraphEndpoint:                "--graph-endpoint--",
		StorageEndpointSuffix:        "--storage-endpoint-suffix--",
		SQLDatabaseDNSSuffix:         "--sql-database-dns-suffix--",
		TrafficManagerDNSSuffix:      "--traffic-manager-dns-suffix--",
		KeyVaultDNSSuffix:            "--key-vault-dns-suffix--",
		ServiceBusEndpointSuffix:     "--service-bus-endpoint-suffix--",
		ServiceManagementVMDNSSuffix: "--asm-vm-dns-suffix--",
		ResourceManagerVMDNSSuffix:   "--arm-vm-dns-suffix--",
		ContainerRegistryDNSSuffix:   "--container-registry-dns-suffix--",
	}

	bytes, err := json.Marshal(env)
	if err != nil {
		t.Fatalf("failed to marshal: %s", err)
	}

	testSubject := Environment{}
	err = json.Unmarshal(bytes, &testSubject)
	if err != nil {
		t.Fatalf("failed to unmarshal: %s", err)
	}

	if env.Name != testSubject.Name {
		t.Errorf("Expected Name to be %q, but got %q", env.Name, testSubject.Name)
	}
	if env.ManagementPortalURL != testSubject.ManagementPortalURL {
		t.Errorf("Expected ManagementPortalURL to be %q, but got %q", env.ManagementPortalURL, testSubject.ManagementPortalURL)
	}
	if env.PublishSettingsURL != testSubject.PublishSettingsURL {
		t.Errorf("Expected PublishSettingsURL to be %q, but got %q", env.PublishSettingsURL, testSubject.PublishSettingsURL)
	}
	if env.ServiceManagementEndpoint != testSubject.ServiceManagementEndpoint {
		t.Errorf("Expected ServiceManagementEndpoint to be %q, but got %q", env.ServiceManagementEndpoint, testSubject.ServiceManagementEndpoint)
	}
	if env.ResourceManagerEndpoint != testSubject.ResourceManagerEndpoint {
		t.Errorf("Expected ResourceManagerEndpoint to be %q, but got %q", env.ResourceManagerEndpoint, testSubject.ResourceManagerEndpoint)
	}
	if env.ActiveDirectoryEndpoint != testSubject.ActiveDirectoryEndpoint {
		t.Errorf("Expected ActiveDirectoryEndpoint to be %q, but got %q", env.ActiveDirectoryEndpoint, testSubject.ActiveDirectoryEndpoint)
	}
	if env.GalleryEndpoint != testSubject.GalleryEndpoint {
		t.Errorf("Expected GalleryEndpoint to be %q, but got %q", env.GalleryEndpoint, testSubject.GalleryEndpoint)
	}
	if env.KeyVaultEndpoint != testSubject.KeyVaultEndpoint {
		t.Errorf("Expected KeyVaultEndpoint to be %q, but got %q", env.KeyVaultEndpoint, testSubject.KeyVaultEndpoint)
	}
	if env.GraphEndpoint != testSubject.GraphEndpoint {
		t.Errorf("Expected GraphEndpoint to be %q, but got %q", env.GraphEndpoint, testSubject.GraphEndpoint)
	}
	if env.StorageEndpointSuffix != testSubject.StorageEndpointSuffix {
		t.Errorf("Expected StorageEndpointSuffix to be %q, but got %q", env.StorageEndpointSuffix, testSubject.StorageEndpointSuffix)
	}
	if env.SQLDatabaseDNSSuffix != testSubject.SQLDatabaseDNSSuffix {
		t.Errorf("Expected SQLDatabaseDNSSuffix to be %q, but got %q", env.SQLDatabaseDNSSuffix, testSubject.SQLDatabaseDNSSuffix)
	}
	if env.TrafficManagerDNSSuffix != testSubject.TrafficManagerDNSSuffix {
		t.Errorf("Expected TrafficManagerDNSSuffix to be %q, but got %q", env.TrafficManagerDNSSuffix, testSubject.TrafficManagerDNSSuffix)
	}
	if env.KeyVaultDNSSuffix != testSubject.KeyVaultDNSSuffix {
		t.Errorf("Expected KeyVaultDNSSuffix to be %q, but got %q", env.KeyVaultDNSSuffix, testSubject.KeyVaultDNSSuffix)
	}
	if env.ServiceBusEndpointSuffix != testSubject.ServiceBusEndpointSuffix {
		t.Errorf("Expected ServiceBusEndpointSuffix to be %q, but got %q", env.ServiceBusEndpointSuffix, testSubject.ServiceBusEndpointSuffix)
	}
	if env.ServiceManagementVMDNSSuffix != testSubject.ServiceManagementVMDNSSuffix {
		t.Errorf("Expected ServiceManagementVMDNSSuffix to be %q, but got %q", env.ServiceManagementVMDNSSuffix, testSubject.ServiceManagementVMDNSSuffix)
	}
	if env.ResourceManagerVMDNSSuffix != testSubject.ResourceManagerVMDNSSuffix {
		t.Errorf("Expected ResourceManagerVMDNSSuffix to be %q, but got %q", env.ResourceManagerVMDNSSuffix, testSubject.ResourceManagerVMDNSSuffix)
	}
	if env.ContainerRegistryDNSSuffix != testSubject.ContainerRegistryDNSSuffix {
		t.Errorf("Expected ContainerRegistryDNSSuffix to be %q, but got %q", env.ContainerRegistryDNSSuffix, testSubject.ContainerRegistryDNSSuffix)
	}
}
