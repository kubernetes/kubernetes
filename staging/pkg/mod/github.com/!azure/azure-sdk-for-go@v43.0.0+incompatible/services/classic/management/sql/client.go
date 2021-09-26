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
	"fmt"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

// Definitions of numerous constants representing API endpoints.
const (
	azureCreateDatabaseServerURL = "services/sqlservers/servers"
	azureListDatabaseServersURL  = "services/sqlservers/servers"
	azureDeleteDatabaseServerURL = "services/sqlservers/servers/%s"

	azureCreateFirewallRuleURL = "services/sqlservers/servers/%s/firewallrules"
	azureGetFirewallRuleURL    = "services/sqlservers/servers/%s/firewallrules/%s"
	azureListFirewallRulesURL  = "services/sqlservers/servers/%s/firewallrules"
	azureUpdateFirewallRuleURL = "services/sqlservers/servers/%s/firewallrules/%s"
	azureDeleteFirewallRuleURL = "services/sqlservers/servers/%s/firewallrules/%s"

	azureCreateDatabaseURL = "services/sqlservers/servers/%s/databases"
	azureGetDatabaseURL    = "services/sqlservers/servers/%s/databases/%s"
	azureListDatabasesURL  = "services/sqlservers/servers/%s/databases?contentview=generic"
	azureUpdateDatabaseURL = "services/sqlservers/servers/%s/databases/%s"
	azureDeleteDatabaseURL = "services/sqlservers/servers/%s/databases/%s"

	errParamNotSpecified = "Parameter %s was not specified."

	DatabaseStateCreating = "Creating"
)

// SQLDatabaseClient defines various database CRUD operations.
// It contains a management.Client for making the actual http calls.
type SQLDatabaseClient struct {
	mgmtClient management.Client
}

// NewClient returns a new SQLDatabaseClient struct with the provided
// management.Client as the underlying client.
func NewClient(mgmtClient management.Client) SQLDatabaseClient {
	return SQLDatabaseClient{mgmtClient}
}

// CreateServer creates a new Azure SQL Database server and return its name.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505699.aspx
func (c SQLDatabaseClient) CreateServer(params DatabaseServerCreateParams) (string, error) {
	req, err := xml.Marshal(params)
	if err != nil {
		return "", err
	}

	resp, err := c.mgmtClient.SendAzurePostRequestWithReturnedResponse(azureCreateDatabaseServerURL, req)
	if err != nil {
		return "", err
	}

	var name string
	err = xml.Unmarshal(resp, &name)

	return name, err
}

// ListServers retrieves the Azure SQL Database servers for this subscription.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505702.aspx
func (c SQLDatabaseClient) ListServers() (ListServersResponse, error) {
	var resp ListServersResponse

	data, err := c.mgmtClient.SendAzureGetRequest(azureListDatabaseServersURL)
	if err != nil {
		return resp, err
	}

	err = xml.Unmarshal(data, &resp)
	return resp, err
}

// DeleteServer deletes an Azure SQL Database server (including all its databases).
//
// https://msdn.microsoft.com/en-us/library/azure/dn505695.aspx
func (c SQLDatabaseClient) DeleteServer(name string) error {
	if name == "" {
		return fmt.Errorf(errParamNotSpecified, "name")
	}

	url := fmt.Sprintf(azureDeleteDatabaseServerURL, name)
	_, err := c.mgmtClient.SendAzureDeleteRequest(url)
	return err
}

// CreateFirewallRule creates an Azure SQL Database server
// firewall rule.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505712.aspx
func (c SQLDatabaseClient) CreateFirewallRule(server string, params FirewallRuleCreateParams) error {
	if server == "" {
		return fmt.Errorf(errParamNotSpecified, "server")
	}

	req, err := xml.Marshal(params)
	if err != nil {
		return err
	}

	url := fmt.Sprintf(azureCreateFirewallRuleURL, server)

	_, err = c.mgmtClient.SendAzurePostRequest(url, req)
	return err
}

// GetFirewallRule gets the details of an Azure SQL Database Server firewall rule.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505698.aspx
func (c SQLDatabaseClient) GetFirewallRule(server, ruleName string) (FirewallRuleResponse, error) {
	var rule FirewallRuleResponse

	if server == "" {
		return rule, fmt.Errorf(errParamNotSpecified, "server")
	}
	if ruleName == "" {
		return rule, fmt.Errorf(errParamNotSpecified, "ruleName")
	}

	url := fmt.Sprintf(azureGetFirewallRuleURL, server, ruleName)
	resp, err := c.mgmtClient.SendAzureGetRequest(url)
	if err != nil {
		return rule, err
	}

	err = xml.Unmarshal(resp, &rule)
	return rule, err
}

// ListFirewallRules retrieves the set of firewall rules for an Azure SQL
// Database Server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505715.aspx
func (c SQLDatabaseClient) ListFirewallRules(server string) (ListFirewallRulesResponse, error) {
	var rules ListFirewallRulesResponse

	if server == "" {
		return rules, fmt.Errorf(errParamNotSpecified, "server")
	}

	url := fmt.Sprintf(azureListFirewallRulesURL, server)
	resp, err := c.mgmtClient.SendAzureGetRequest(url)
	if err != nil {
		return rules, err
	}

	err = xml.Unmarshal(resp, &rules)
	return rules, err
}

// UpdateFirewallRule update a firewall rule for an Azure SQL Database server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505707.aspx
func (c SQLDatabaseClient) UpdateFirewallRule(server, ruleName string, params FirewallRuleUpdateParams) error {
	if server == "" {
		return fmt.Errorf(errParamNotSpecified, "server")
	}
	if ruleName == "" {
		return fmt.Errorf(errParamNotSpecified, "ruleName")
	}

	req, err := xml.Marshal(params)
	if err != nil {
		return err
	}

	url := fmt.Sprintf(azureUpdateFirewallRuleURL, server, ruleName)
	_, err = c.mgmtClient.SendAzurePutRequest(url, "text/xml", req)
	return err
}

// DeleteFirewallRule deletes an Azure SQL Database server firewall rule.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505706.aspx
func (c SQLDatabaseClient) DeleteFirewallRule(server, ruleName string) error {
	if server == "" {
		return fmt.Errorf(errParamNotSpecified, "server")
	}
	if ruleName == "" {
		return fmt.Errorf(errParamNotSpecified, "ruleName")
	}

	url := fmt.Sprintf(azureDeleteFirewallRuleURL, server, ruleName)

	_, err := c.mgmtClient.SendAzureDeleteRequest(url)
	return err
}

// CreateDatabase creates a new Microsoft Azure SQL Database on the given database server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505701.aspx
func (c SQLDatabaseClient) CreateDatabase(server string, params DatabaseCreateParams) error {
	if server == "" {
		return fmt.Errorf(errParamNotSpecified, "server")
	}

	req, err := xml.Marshal(params)
	if err != nil {
		return err
	}

	target := fmt.Sprintf(azureCreateDatabaseURL, server)
	_, err = c.mgmtClient.SendAzurePostRequest(target, req)
	return err
}

// WaitForDatabaseCreation is a helper method which waits
// for the creation of the database on the given server.
func (c SQLDatabaseClient) WaitForDatabaseCreation(
	server, database string,
	cancel chan struct{}) error {
	for {
		stat, err := c.GetDatabase(server, database)
		if err != nil {
			return err
		}
		if stat.State != DatabaseStateCreating {
			return nil
		}

		select {
		case <-time.After(management.DefaultOperationPollInterval):
		case <-cancel:
			return management.ErrOperationCancelled
		}
	}
}

// GetDatabase gets the details for an Azure SQL Database.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505708.aspx
func (c SQLDatabaseClient) GetDatabase(server, database string) (ServiceResource, error) {
	var db ServiceResource

	if database == "" {
		return db, fmt.Errorf(errParamNotSpecified, "database")
	}
	if server == "" {
		return db, fmt.Errorf(errParamNotSpecified, "server")
	}

	url := fmt.Sprintf(azureGetDatabaseURL, server, database)
	resp, err := c.mgmtClient.SendAzureGetRequest(url)
	if err != nil {
		return db, err
	}

	err = xml.Unmarshal(resp, &db)
	return db, err
}

// ListDatabases returns a list of Azure SQL Databases on the given server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505711.aspx
func (c SQLDatabaseClient) ListDatabases(server string) (ListDatabasesResponse, error) {
	var databases ListDatabasesResponse
	if server == "" {
		return databases, fmt.Errorf(errParamNotSpecified, "server name")
	}

	url := fmt.Sprintf(azureListDatabasesURL, server)
	resp, err := c.mgmtClient.SendAzureGetRequest(url)
	if err != nil {
		return databases, err
	}

	err = xml.Unmarshal(resp, &databases)
	return databases, err
}

// UpdateDatabase updates the details of the given Database off the given server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505718.aspx
func (c SQLDatabaseClient) UpdateDatabase(
	server, database string,
	params ServiceResourceUpdateParams) (management.OperationID, error) {
	if database == "" {
		return "", fmt.Errorf(errParamNotSpecified, "database")
	}
	if server == "" {
		return "", fmt.Errorf(errParamNotSpecified, "server")
	}

	url := fmt.Sprintf(azureUpdateDatabaseURL, server, database)
	req, err := xml.Marshal(params)
	if err != nil {
		return "", err
	}

	return c.mgmtClient.SendAzurePutRequest(url, "text/xml", req)
}

// DeleteDatabase deletes the Azure SQL Database off the given server.
//
// https://msdn.microsoft.com/en-us/library/azure/dn505705.aspx
func (c SQLDatabaseClient) DeleteDatabase(server, database string) error {
	if database == "" {
		return fmt.Errorf(errParamNotSpecified, "database")
	}
	if server == "" {
		return fmt.Errorf(errParamNotSpecified, "server")
	}

	url := fmt.Sprintf(azureDeleteDatabaseURL, server, database)

	_, err := c.mgmtClient.SendAzureDeleteRequest(url)

	return err
}
