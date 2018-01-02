package main

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
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/arm/examples/helpers"
	"github.com/Azure/azure-sdk-for-go/arm/storage"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"
)

func main() {
	resourceGroup := "resourceGroupName"
	name := "gosdktestname01"

	c := map[string]string{
		"AZURE_CLIENT_ID":       os.Getenv("AZURE_CLIENT_ID"),
		"AZURE_CLIENT_SECRET":   os.Getenv("AZURE_CLIENT_SECRET"),
		"AZURE_SUBSCRIPTION_ID": os.Getenv("AZURE_SUBSCRIPTION_ID"),
		"AZURE_TENANT_ID":       os.Getenv("AZURE_TENANT_ID")}
	if err := checkEnvVar(&c); err != nil {
		log.Fatalf("Error: %v", err)
		return
	}
	spt, err := helpers.NewServicePrincipalTokenFromCredentials(c, azure.PublicCloud.ResourceManagerEndpoint)
	if err != nil {
		log.Fatalf("Error: %v", err)
		return
	}

	ac := storage.NewAccountsClient(c["AZURE_SUBSCRIPTION_ID"])
	ac.Authorizer = autorest.NewBearerAuthorizer(spt)

	cna, err := ac.CheckNameAvailability(
		storage.AccountCheckNameAvailabilityParameters{
			Name: to.StringPtr(name),
			Type: to.StringPtr("Microsoft.Storage/storageAccounts")})
	if err != nil {
		log.Fatalf("Error: %v", err)
		return
	}
	if !to.Bool(cna.NameAvailable) {
		fmt.Printf("%s is unavailable -- try with another name\n", name)
		return
	}
	fmt.Printf("%s is available\n\n", name)

	cp := storage.AccountCreateParameters{
		Sku: &storage.Sku{
			Name: storage.StandardLRS,
			Tier: storage.Standard},
		Location: to.StringPtr("westus")}
	cancel := make(chan struct{})

	_, errchan := ac.Create(resourceGroup, name, cp, cancel)
	err = <-errchan
	if err != nil {
		fmt.Printf("Create '%s' storage account failed: %v\n", name, err)
		return
	}
	fmt.Printf("Successfully created '%s' storage account in '%s' resource group\n\n", name, resourceGroup)

	r, err := ac.Delete(resourceGroup, name)
	if err != nil {
		fmt.Printf("Delete of '%s' failed with status %s\n...%v\n", name, r.Status, err)
		return
	}
	fmt.Printf("Deletion of '%s' storage account in '%s' resource group succeeded -- %s\n", name, resourceGroup, r.Status)
}

func checkEnvVar(envVars *map[string]string) error {
	var missingVars []string
	for varName, value := range *envVars {
		if value == "" {
			missingVars = append(missingVars, varName)
		}
	}
	if len(missingVars) > 0 {
		return fmt.Errorf("Missing environment variables %v", missingVars)
	}
	return nil
}
