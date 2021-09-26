// +build go1.7

package sql

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
)

// DatabaseServerCreateParams represents the set of possible parameters
// when issuing a database server creation request to Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505699.aspx
type DatabaseServerCreateParams struct {
	XMLName                    xml.Name `xml:"http://schemas.microsoft.com/sqlazure/2010/12/ Server"`
	AdministratorLogin         string
	AdministratorLoginPassword string
	Location                   string
	Version                    string
}

// DatabaseServerCreateResponse represents the response following the creation of
// a database server on Azure.
type DatabaseServerCreateResponse struct {
	ServerName string
}

const (
	DatabaseServerVersion11 = "2.0"
	DatabaseServerVersion12 = "12.0"
)

// DatabaseServer represents the set of data received from
// a database server list operation.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505702.aspx
type DatabaseServer struct {
	Name                     string
	AdministratorLogin       string
	Location                 string
	FullyQualifiedDomainName string
	Version                  string
	State                    string
}

type ListServersResponse struct {
	DatabaseServers []DatabaseServer `xml:"Server"`
}

// FirewallRuleCreateParams represents the set of possible
// parameters when creating a firewall rule on an Azure database server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505712.aspx
type FirewallRuleCreateParams struct {
	XMLName        xml.Name `xml:"http://schemas.microsoft.com/windowsazure ServiceResource"`
	Name           string
	StartIPAddress string
	EndIPAddress   string
}

// FirewallRuleResponse represents the set of data received from
// an Azure database server firewall rule get response.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505698.aspx
type FirewallRuleResponse struct {
	Name           string
	StartIPAddress string
	EndIPAddress   string
}

type ListFirewallRulesResponse struct {
	FirewallRules []FirewallRuleResponse `xml:"ServiceResource"`
}

// FirewallRuleUpdateParams represents the set of possible
// parameters when issuing an update of a database server firewall rule.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505707.aspx
type FirewallRuleUpdateParams struct {
	XMLName        xml.Name `xml:"http://schemas.microsoft.com/windowsazure ServiceResource"`
	Name           string
	StartIPAddress string
	EndIPAddress   string
}

// DatabaseCreateParams represents the set of possible parameters when issuing
// a database creation to Azure, and reading a list response from Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505701.aspx
type DatabaseCreateParams struct {
	XMLName            xml.Name `xml:"http://schemas.microsoft.com/windowsazure ServiceResource"`
	Name               string
	Edition            string `xml:",omitempty"`
	CollationName      string `xml:",omitempty"`
	MaxSizeBytes       int64  `xml:",omitempty"`
	ServiceObjectiveID string `xml:"ServiceObjectiveId,omitempty"`
}

// ServiceResource represents the set of parameters obtained from a database
// get or list call.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505708.aspx
type ServiceResource struct {
	Name               string
	State              string
	SelfLink           string
	Edition            string
	CollationName      string
	MaxSizeBytes       int64
	ServiceObjectiveID string `xml:"ServiceObjectiveId,omitempty"`
}

type ListDatabasesResponse struct {
	ServiceResources []ServiceResource `xml:"ServiceResource"`
}

// ServiceResourceUpdateParams represents the set of parameters available
// for a database service update operation.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505718.aspx
type ServiceResourceUpdateParams struct {
	XMLName            xml.Name `xml:"http://schemas.microsoft.com/windowsazure ServiceResource"`
	Name               string
	Edition            string `xml:",omitempty"`
	MaxSizeBytes       int64  `xml:",omitempty"`
	ServiceObjectiveID string `xml:"ServiceObjectiveId,omitempty"`
}
