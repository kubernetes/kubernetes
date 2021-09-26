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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"testing"
)

const (
	batchResourceID      = "--batch-resource-id--"
	datalakeResourceID   = "--datalake-resource-id--"
	graphResourceID      = "--graph-resource-id--"
	keyvaultResourceID   = "--keyvault-resource-id--"
	opInsightsResourceID = "--operational-insights-resource-id--"
)

// This correlates to the expected contents of ./testdata/test_environment_1.json
var testEnvironment1 = Environment{
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
	TokenAudience:                "--token-audience",
	ResourceIdentifiers: ResourceIdentifier{
		Batch:               batchResourceID,
		Datalake:            datalakeResourceID,
		Graph:               graphResourceID,
		KeyVault:            keyvaultResourceID,
		OperationalInsights: opInsightsResourceID,
	},
}

func TestEnvironment_EnvironmentFromURL_NoOverride_Success(t *testing.T) {
	fileContents, _ := ioutil.ReadFile(filepath.Join("testdata", "test_metadata_environment_1.json"))
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fileContents))
	}))
	defer ts.Close()

	got, err := EnvironmentFromURL(ts.URL)

	if err != nil {
		t.Error(err)
	}
	if got.Name != "HybridEnvironment" {
		t.Logf("got: %v want: HybridEnvironment", got.Name)
		t.Fail()
	}
}

func TestEnvironment_EnvironmentFromURL_OverrideStorageSuffix_Success(t *testing.T) {
	fileContents, _ := ioutil.ReadFile(filepath.Join("testdata", "test_metadata_environment_1.json"))
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fileContents))
	}))
	defer ts.Close()
	overrideProperty := OverrideProperty{
		Key:   EnvironmentStorageEndpointSuffix,
		Value: "fakeStorageSuffix",
	}
	got, err := EnvironmentFromURL(ts.URL, overrideProperty)

	if err != nil {
		t.Error(err)
	}
	if got.StorageEndpointSuffix != "fakeStorageSuffix" {
		t.Logf("got: %v want: fakeStorageSuffix", got.StorageEndpointSuffix)
		t.Fail()
	}
}

func TestEnvironment_EnvironmentFromURL_EmptyEndpoint_Failure(t *testing.T) {
	_, err := EnvironmentFromURL("")

	if err == nil {
		t.Fail()
	}
	if err.Error() != "Metadata resource manager endpoint is empty" {
		t.Fail()
	}
}

func TestEnvironment_EnvironmentFromFile(t *testing.T) {
	got, err := EnvironmentFromFile(filepath.Join("testdata", "test_environment_1.json"))
	if err != nil {
		t.Error(err)
	}

	if got != testEnvironment1 {
		t.Logf("got: %v want: %v", got, testEnvironment1)
		t.Fail()
	}
}

func TestEnvironment_EnvironmentFromName_Stack(t *testing.T) {
	_, currentFile, _, _ := runtime.Caller(0)
	prevEnvFilepathValue := os.Getenv(EnvironmentFilepathName)
	os.Setenv(EnvironmentFilepathName, filepath.Join(path.Dir(currentFile), "testdata", "test_environment_1.json"))
	defer os.Setenv(EnvironmentFilepathName, prevEnvFilepathValue)

	got, err := EnvironmentFromName("AZURESTACKCLOUD")
	if err != nil {
		t.Error(err)
	}

	if got != testEnvironment1 {
		t.Logf("got: %v want: %v", got, testEnvironment1)
		t.Fail()
	}
}

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
		"serviceBusEndpoint": "--service-bus-endpoint--",
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
		"containerRegistryDNSSuffix": "--container-registry-dns-suffix--",
		"resourceIdentifiers": {
			"batch": "` + batchResourceID + `",
			"datalake": "` + datalakeResourceID + `",
			"graph": "` + graphResourceID + `",
			"keyVault": "` + keyvaultResourceID + `",
			"operationalInsights": "` + opInsightsResourceID + `"
		}
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
	if "--service-bus-endpoint--" != testSubject.ServiceBusEndpoint {
		t.Errorf("Expected ServiceBusEndpoint to be \"--service-bus-endpoint--\", but goet %q", testSubject.ServiceBusEndpoint)
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
	if batchResourceID != testSubject.ResourceIdentifiers.Batch {
		t.Errorf("Expected ResourceIdentifiers.Batch to be "+batchResourceID+", but got %q", testSubject.ResourceIdentifiers.Batch)
	}
	if datalakeResourceID != testSubject.ResourceIdentifiers.Datalake {
		t.Errorf("Expected ResourceIdentifiers.Datalake to be "+datalakeResourceID+", but got %q", testSubject.ResourceIdentifiers.Datalake)
	}
	if graphResourceID != testSubject.ResourceIdentifiers.Graph {
		t.Errorf("Expected ResourceIdentifiers.Graph to be "+graphResourceID+", but got %q", testSubject.ResourceIdentifiers.Graph)
	}
	if keyvaultResourceID != testSubject.ResourceIdentifiers.KeyVault {
		t.Errorf("Expected ResourceIdentifiers.KeyVault to be "+keyvaultResourceID+", but got %q", testSubject.ResourceIdentifiers.KeyVault)
	}
	if opInsightsResourceID != testSubject.ResourceIdentifiers.OperationalInsights {
		t.Errorf("Expected ResourceIdentifiers.OperationalInsights to be "+opInsightsResourceID+", but got %q", testSubject.ResourceIdentifiers.OperationalInsights)
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
		ServiceBusEndpoint:           "--service-bus-endpoint--",
		StorageEndpointSuffix:        "--storage-endpoint-suffix--",
		SQLDatabaseDNSSuffix:         "--sql-database-dns-suffix--",
		TrafficManagerDNSSuffix:      "--traffic-manager-dns-suffix--",
		KeyVaultDNSSuffix:            "--key-vault-dns-suffix--",
		ServiceBusEndpointSuffix:     "--service-bus-endpoint-suffix--",
		ServiceManagementVMDNSSuffix: "--asm-vm-dns-suffix--",
		ResourceManagerVMDNSSuffix:   "--arm-vm-dns-suffix--",
		ContainerRegistryDNSSuffix:   "--container-registry-dns-suffix--",
		ResourceIdentifiers: ResourceIdentifier{
			Batch:               batchResourceID,
			Datalake:            datalakeResourceID,
			Graph:               graphResourceID,
			KeyVault:            keyvaultResourceID,
			OperationalInsights: opInsightsResourceID,
		},
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
	if env.ServiceBusEndpoint != testSubject.ServiceBusEndpoint {
		t.Errorf("Expected ServiceBusEnpoint to be %q, but got %q", env.ServiceBusEndpoint, testSubject.ServiceBusEndpoint)
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
	if env.ResourceIdentifiers.Batch != testSubject.ResourceIdentifiers.Batch {
		t.Errorf("Expected ResourceIdentifiers.Batch to be %q, but got %q", env.ResourceIdentifiers.Batch, testSubject.ResourceIdentifiers.Batch)
	}
	if env.ResourceIdentifiers.Datalake != testSubject.ResourceIdentifiers.Datalake {
		t.Errorf("Expected ResourceIdentifiers.Datalake to be %q, but got %q", env.ResourceIdentifiers.Datalake, testSubject.ResourceIdentifiers.Datalake)
	}
	if env.ResourceIdentifiers.Graph != testSubject.ResourceIdentifiers.Graph {
		t.Errorf("Expected ResourceIdentifiers.Graph to be %q, but got %q", env.ResourceIdentifiers.Graph, testSubject.ResourceIdentifiers.Graph)
	}
	if env.ResourceIdentifiers.KeyVault != testSubject.ResourceIdentifiers.KeyVault {
		t.Errorf("Expected ResourceIdentifiers.KeyVault to be %q, but got %q", env.ResourceIdentifiers.KeyVault, testSubject.ResourceIdentifiers.KeyVault)
	}
	if env.ResourceIdentifiers.OperationalInsights != testSubject.ResourceIdentifiers.OperationalInsights {
		t.Errorf("Expected ResourceIdentifiers.OperationalInsights to be %q, but got %q", env.ResourceIdentifiers.OperationalInsights, testSubject.ResourceIdentifiers.OperationalInsights)
	}
}

func TestSetEnvironment(t *testing.T) {
	const testEnvName = "testenvironment"
	if _, err := EnvironmentFromName(testEnvName); err == nil {
		t.Fatal("expected non-nil error")
	}
	testEnv := Environment{Name: testEnvName}
	SetEnvironment(testEnvName, testEnv)
	result, err := EnvironmentFromName(testEnvName)
	if err != nil {
		t.Fatalf("failed to get custom environment: %v", err)
	}
	if testEnv != result {
		t.Fatalf("expected %v, got %v", testEnv, result)
	}
}
