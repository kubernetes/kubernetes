// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"flag"
	"fmt"
	"io/ioutil"
	"strings"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/jwt"
	bigquery "google.golang.org/api/bigquery/v2"
)

var (
	// TODO(jnagal): Condense all flags to an identity file and a pem key file.
	clientId       = flag.String("bq_id", "", "Client ID")
	clientSecret   = flag.String("bq_secret", "notasecret", "Client Secret")
	projectId      = flag.String("bq_project_id", "", "Bigquery project ID")
	serviceAccount = flag.String("bq_account", "", "Service account email")
	pemFile        = flag.String("bq_credentials_file", "", "Credential Key file (pem)")
)

const (
	errAlreadyExists string = "Error 409: Already Exists"
)

type Client struct {
	service   *bigquery.Service
	token     *oauth2.Token
	datasetId string
	tableId   string
}

// Helper method to create an authenticated connection.
func connect() (*oauth2.Token, *bigquery.Service, error) {
	if *clientId == "" {
		return nil, nil, fmt.Errorf("no client id specified")
	}
	if *serviceAccount == "" {
		return nil, nil, fmt.Errorf("no service account specified")
	}
	if *projectId == "" {
		return nil, nil, fmt.Errorf("no project id specified")
	}
	authScope := bigquery.BigqueryScope
	if *pemFile == "" {
		return nil, nil, fmt.Errorf("no credentials specified")
	}
	pemBytes, err := ioutil.ReadFile(*pemFile)
	if err != nil {
		return nil, nil, fmt.Errorf("could not access credential file %v - %v", pemFile, err)
	}

	jwtConfig := &jwt.Config{
		Email:      *serviceAccount,
		Scopes:     []string{authScope},
		PrivateKey: pemBytes,
		TokenURL:   "https://accounts.google.com/o/oauth2/token",
	}
	token, err := jwtConfig.TokenSource(oauth2.NoContext).Token()
	if err != nil {
		return nil, nil, err
	}
	if !token.Valid() {
		return nil, nil, fmt.Errorf("invalid token for BigQuery oauth")
	}

	config := &oauth2.Config{
		ClientID:     *clientId,
		ClientSecret: *clientSecret,
		Scopes:       []string{authScope},
		Endpoint: oauth2.Endpoint{
			AuthURL:  "https://accounts.google.com/o/oauth2/auth",
			TokenURL: "https://accounts.google.com/o/oauth2/token",
		},
	}
	client := config.Client(oauth2.NoContext, token)

	service, err := bigquery.New(client)
	if err != nil {
		fmt.Printf("Failed to create new service: %v\n", err)
		return nil, nil, err
	}

	return token, service, nil
}

// Creates a new client instance with an authenticated connection to bigquery.
func NewClient() (*Client, error) {
	token, service, err := connect()
	if err != nil {
		return nil, err
	}
	c := &Client{
		token:   token,
		service: service,
	}
	return c, nil
}

func (c *Client) Close() error {
	c.service = nil
	return nil
}

// Helper method to return the bigquery service connection.
// Expired connection is refreshed.
func (c *Client) getService() (*bigquery.Service, error) {
	if c.token == nil || c.service == nil {
		return nil, fmt.Errorf("service not initialized")
	}

	// Refresh expired token.
	if !c.token.Valid() {
		token, service, err := connect()
		if err != nil {
			return nil, err
		}
		c.token = token
		c.service = service
		return service, nil
	}
	return c.service, nil
}

func (c *Client) PrintDatasets() error {
	datasetList, err := c.service.Datasets.List(*projectId).Do()
	if err != nil {
		fmt.Printf("Failed to get list of datasets\n")
		return err
	} else {
		fmt.Printf("Successfully retrieved datasets. Retrieved: %d\n", len(datasetList.Datasets))
	}

	for _, d := range datasetList.Datasets {
		fmt.Printf("%s %s\n", d.Id, d.FriendlyName)
	}
	return nil
}

func (c *Client) CreateDataset(datasetId string) error {
	if c.service == nil {
		return fmt.Errorf("no service created")
	}
	_, err := c.service.Datasets.Insert(*projectId, &bigquery.Dataset{
		DatasetReference: &bigquery.DatasetReference{
			DatasetId: datasetId,
			ProjectId: *projectId,
		},
	}).Do()
	// TODO(jnagal): Do a Get() to verify dataset already exists.
	if err != nil && !strings.Contains(err.Error(), errAlreadyExists) {
		return err
	}
	c.datasetId = datasetId
	return nil
}

// Create a table with provided table ID and schema.
// Schema is currently not updated if the table already exists.
func (c *Client) CreateTable(tableId string, schema *bigquery.TableSchema) error {
	if c.service == nil || c.datasetId == "" {
		return fmt.Errorf("no dataset created")
	}
	_, err := c.service.Tables.Get(*projectId, c.datasetId, tableId).Do()
	if err != nil {
		// Create a new table.
		_, err := c.service.Tables.Insert(*projectId, c.datasetId, &bigquery.Table{
			Schema: schema,
			TableReference: &bigquery.TableReference{
				DatasetId: c.datasetId,
				ProjectId: *projectId,
				TableId:   tableId,
			},
		}).Do()
		if err != nil {
			return err
		}
	}
	// TODO(jnagal): Update schema if it has changed. We can only extend existing schema.
	c.tableId = tableId
	return nil
}

// Add a row to the connected table.
func (c *Client) InsertRow(rowData map[string]interface{}) error {
	service, _ := c.getService()
	if service == nil || c.datasetId == "" || c.tableId == "" {
		return fmt.Errorf("table not setup to add rows")
	}
	jsonRows := make(map[string]bigquery.JsonValue)
	for key, value := range rowData {
		jsonRows[key] = bigquery.JsonValue(value)
	}
	rows := []*bigquery.TableDataInsertAllRequestRows{
		{
			Json: jsonRows,
		},
	}

	// TODO(jnagal): Batch insert requests.
	insertRequest := &bigquery.TableDataInsertAllRequest{Rows: rows}

	result, err := service.Tabledata.InsertAll(*projectId, c.datasetId, c.tableId, insertRequest).Do()
	if err != nil {
		return fmt.Errorf("error inserting row: %v", err)
	}

	if len(result.InsertErrors) > 0 {
		errstr := fmt.Sprintf("Insertion for %d rows failed\n", len(result.InsertErrors))
		for _, errors := range result.InsertErrors {
			for _, errorproto := range errors.Errors {
				errstr += fmt.Sprintf("Error inserting row %d: %+v\n", errors.Index, errorproto)
			}
		}
		return fmt.Errorf(errstr)
	}
	return nil
}
