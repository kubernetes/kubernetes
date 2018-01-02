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

	"github.com/Azure/azure-sdk-for-go/arm/dns"
	"github.com/Azure/azure-sdk-for-go/arm/examples/helpers"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"
)

func main() {
	resourceGroup := "delete-dns"

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

	dc := dns.NewZonesClient(c["AZURE_SUBSCRIPTION_ID"])
	dc.Authorizer = autorest.NewBearerAuthorizer(spt)
	rc := dns.NewRecordSetsClient(c["AZURE_SUBSCRIPTION_ID"])
	rc.Authorizer = autorest.NewBearerAuthorizer(spt)

	newZoneName := "xtoph-test.local"

	zoneParam := &dns.Zone{
		Location: to.StringPtr("global"),
		ZoneProperties: &dns.ZoneProperties{
			MaxNumberOfRecordSets: to.Int64Ptr(1),
			NumberOfRecordSets:    to.Int64Ptr(1),
			//			NameServers: [ "my-nameserver.xtoph.com" ]
		},
	}

	newZone, err := dc.CreateOrUpdate(resourceGroup, newZoneName, *zoneParam, "", "")
	if err != nil {
		log.Fatalf("Couldn't create Zone %s Error: %v\n", newZoneName, err)
		return
	}
	fmt.Printf("Status: %s\n", newZone.Status)

	fmt.Printf("New Zone Created\n")

	rrparams := &dns.RecordSet{
		RecordSetProperties: &dns.RecordSetProperties{
			TTL: to.Int64Ptr(1000),
			ARecords: &[]dns.ARecord{
				{Ipv4Address: to.StringPtr("2.2.2.2")},
			},
		},
	}

	fmt.Printf("Creating A Rec\n")

	aRecName := "newTestARecordFromAPI"
	_, err = rc.CreateOrUpdate(resourceGroup, newZoneName, aRecName, dns.A, *rrparams, "", "")
	if err != nil {
		log.Fatalf("Error creating Cname: %s, %v", newZoneName, err)
		return
	}

	cnameparams := &dns.RecordSet{
		RecordSetProperties: &dns.RecordSetProperties{
			TTL: to.Int64Ptr(1000),
			CnameRecord: &dns.CnameRecord{
				Cname: to.StringPtr("my-new-cname"),
			},
		},
	}

	fmt.Printf("Creating CName\n")
	cnameRecordName := "newTestCNameRecordFromAPI"
	_, err = rc.CreateOrUpdate(resourceGroup, newZoneName, cnameRecordName, dns.CNAME, *cnameparams, "", "")
	if err != nil {
		log.Fatalf("Error creating Cname: %s, %v", newZoneName, err)
		return
	}

	zone, err := dc.Get(resourceGroup, newZoneName)

	if err != nil {
		log.Fatalf("Error getting zone: %s, %v", newZoneName, err)
		return
	}

	fmt.Printf("Nameservers for %s \n", *zone.Name)

	for _, ns := range *zone.NameServers {
		fmt.Printf("%s\n", ns)
	}

	fmt.Printf("Recordsets for %s \n", *zone.Name)

	var top int32
	top = 10
	rrsets, err := rc.ListByDNSZone(resourceGroup, newZoneName, &top, "")
	if err != nil {
		log.Fatalf("Error: %v", err)
		return
	}

	for _, rrset := range *rrsets.Value {
		fmt.Printf("Recordset: %s Type: %s\n", *rrset.Name, *rrset.Type)

		switch *rrset.Type {
		case "Microsoft.Network/dnszones/A":
			printARecords(rrset)
		case "Microsoft.Network/dnszones/CNAME":
			printCNames(rrset)
		case "Microsoft.Network/dnszones/NS":
			printNS(rrset)
		case "Microsoft.Network/dnszones/SOA":
			printSOA(rrset)
		}
	}

	fmt.Printf("*** Cleaning up *** ")
	rc.Delete(resourceGroup, newZoneName, cnameRecordName, dns.CNAME, "")
	rc.Delete(resourceGroup, newZoneName, aRecName, dns.A, "")
	dc.Delete(resourceGroup, newZoneName, "", nil)
	fmt.Printf("done\n")

}

func printNS(rrset dns.RecordSet) {
	fmt.Printf("*** NS Record ***\n")
	if rrset.NsRecords != nil {
		for _, ns := range *rrset.NsRecords {
			fmt.Printf("Nameserver: %s\n", *ns.Nsdname)
		}
	} else {
		fmt.Printf("*** None ***\n")
	}
}

func printSOA(rrset dns.RecordSet) {
	fmt.Printf("*** SOA Record ***\n")

	if rrset.SoaRecord != nil {
		fmt.Printf("Email: %s\n", *rrset.SoaRecord.Email)
		fmt.Printf("Host: %s\n", *rrset.SoaRecord.Host)
	} else {
		fmt.Printf("*** None ***\n")
	}

}

func printCNames(rrset dns.RecordSet) {
	fmt.Printf("*** CNAME Record ***\n")
	if rrset.CnameRecord != nil {
		fmt.Printf("Cname %s\n", *rrset.CnameRecord.Cname)
	} else {
		fmt.Printf("*** None ***\n")
	}

}
func printARecords(rrset dns.RecordSet) {
	fmt.Printf("*** A Records ***\n")
	if rrset.ARecords != nil {
		for _, arec := range *rrset.ARecords {
			fmt.Printf("record %s\n", *arec.Ipv4Address)
		}
	} else {
		fmt.Printf("*** None ***\n")
	}

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
