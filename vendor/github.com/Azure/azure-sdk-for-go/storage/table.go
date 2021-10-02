package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	tablesURIPath                  = "/Tables"
	nextTableQueryParameter        = "NextTableName"
	headerNextPartitionKey         = "x-ms-continuation-NextPartitionKey"
	headerNextRowKey               = "x-ms-continuation-NextRowKey"
	nextPartitionKeyQueryParameter = "NextPartitionKey"
	nextRowKeyQueryParameter       = "NextRowKey"
)

// TableAccessPolicy are used for SETTING table policies
type TableAccessPolicy struct {
	ID         string
	StartTime  time.Time
	ExpiryTime time.Time
	CanRead    bool
	CanAppend  bool
	CanUpdate  bool
	CanDelete  bool
}

// Table represents an Azure table.
type Table struct {
	tsc           *TableServiceClient
	Name          string `json:"TableName"`
	OdataEditLink string `json:"odata.editLink"`
	OdataID       string `json:"odata.id"`
	OdataMetadata string `json:"odata.metadata"`
	OdataType     string `json:"odata.type"`
}

// EntityQueryResult contains the response from
// ExecuteQuery and ExecuteQueryNextResults functions.
type EntityQueryResult struct {
	OdataMetadata string    `json:"odata.metadata"`
	Entities      []*Entity `json:"value"`
	QueryNextLink
	table *Table
}

type continuationToken struct {
	NextPartitionKey string
	NextRowKey       string
}

func (t *Table) buildPath() string {
	return fmt.Sprintf("/%s", t.Name)
}

func (t *Table) buildSpecificPath() string {
	return fmt.Sprintf("%s('%s')", tablesURIPath, t.Name)
}

// Get gets the referenced table.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/querying-tables-and-entities
func (t *Table) Get(timeout uint, ml MetadataLevel) error {
	if ml == EmptyPayload {
		return errEmptyPayload
	}

	query := url.Values{
		"timeout": {strconv.FormatUint(uint64(timeout), 10)},
	}
	headers := t.tsc.client.getStandardHeaders()
	headers[headerAccept] = string(ml)

	uri := t.tsc.client.getEndpoint(tableServiceName, t.buildSpecificPath(), query)
	resp, err := t.tsc.client.exec(http.MethodGet, uri, headers, nil, t.tsc.auth)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err = checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return err
	}

	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	err = json.Unmarshal(respBody, t)
	if err != nil {
		return err
	}
	return nil
}

// Create creates the referenced table.
// This function fails if the name is not compliant
// with the specification or the tables already exists.
// ml determines the level of detail of metadata in the operation response,
// or no data at all.
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/create-table
func (t *Table) Create(timeout uint, ml MetadataLevel, options *TableOptions) error {
	uri := t.tsc.client.getEndpoint(tableServiceName, tablesURIPath, url.Values{
		"timeout": {strconv.FormatUint(uint64(timeout), 10)},
	})

	type createTableRequest struct {
		TableName string `json:"TableName"`
	}
	req := createTableRequest{TableName: t.Name}
	buf := new(bytes.Buffer)
	if err := json.NewEncoder(buf).Encode(req); err != nil {
		return err
	}

	headers := t.tsc.client.getStandardHeaders()
	headers = addReturnContentHeaders(headers, ml)
	headers = addBodyRelatedHeaders(headers, buf.Len())
	headers = options.addToHeaders(headers)

	resp, err := t.tsc.client.exec(http.MethodPost, uri, headers, buf, t.tsc.auth)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if ml == EmptyPayload {
		if err := checkRespCode(resp, []int{http.StatusNoContent}); err != nil {
			return err
		}
	} else {
		if err := checkRespCode(resp, []int{http.StatusCreated}); err != nil {
			return err
		}
	}

	if ml != EmptyPayload {
		data, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		err = json.Unmarshal(data, t)
		if err != nil {
			return err
		}
	}

	return nil
}

// Delete deletes the referenced table.
// This function fails if the table is not present.
// Be advised: Delete deletes all the entries that may be present.
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/delete-table
func (t *Table) Delete(timeout uint, options *TableOptions) error {
	uri := t.tsc.client.getEndpoint(tableServiceName, t.buildSpecificPath(), url.Values{
		"timeout": {strconv.Itoa(int(timeout))},
	})

	headers := t.tsc.client.getStandardHeaders()
	headers = addReturnContentHeaders(headers, EmptyPayload)
	headers = options.addToHeaders(headers)

	resp, err := t.tsc.client.exec(http.MethodDelete, uri, headers, nil, t.tsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)

	return checkRespCode(resp, []int{http.StatusNoContent})
}

// QueryOptions includes options for a query entities operation.
// Top, filter and select are OData query options.
type QueryOptions struct {
	Top       uint
	Filter    string
	Select    []string
	RequestID string
}

func (options *QueryOptions) getParameters() (url.Values, map[string]string) {
	query := url.Values{}
	headers := map[string]string{}
	if options != nil {
		if options.Top > 0 {
			query.Add(OdataTop, strconv.FormatUint(uint64(options.Top), 10))
		}
		if options.Filter != "" {
			query.Add(OdataFilter, options.Filter)
		}
		if len(options.Select) > 0 {
			query.Add(OdataSelect, strings.Join(options.Select, ","))
		}
		headers = addToHeaders(headers, "x-ms-client-request-id", options.RequestID)
	}
	return query, headers
}

// QueryEntities returns the entities in the table.
// You can use query options defined by the OData Protocol specification.
//
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/query-entities
func (t *Table) QueryEntities(timeout uint, ml MetadataLevel, options *QueryOptions) (*EntityQueryResult, error) {
	if ml == EmptyPayload {
		return nil, errEmptyPayload
	}
	query, headers := options.getParameters()
	query = addTimeout(query, timeout)
	uri := t.tsc.client.getEndpoint(tableServiceName, t.buildPath(), query)
	return t.queryEntities(uri, headers, ml)
}

// NextResults returns the next page of results
// from a QueryEntities or NextResults operation.
//
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/query-entities
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/query-timeout-and-pagination
func (eqr *EntityQueryResult) NextResults(options *TableOptions) (*EntityQueryResult, error) {
	if eqr == nil {
		return nil, errNilPreviousResult
	}
	if eqr.NextLink == nil {
		return nil, errNilNextLink
	}
	headers := options.addToHeaders(map[string]string{})
	return eqr.table.queryEntities(*eqr.NextLink, headers, eqr.ml)
}

// SetPermissions sets up table ACL permissions
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/Set-Table-ACL
func (t *Table) SetPermissions(tap []TableAccessPolicy, timeout uint, options *TableOptions) error {
	params := url.Values{"comp": {"acl"},
		"timeout": {strconv.Itoa(int(timeout))},
	}

	uri := t.tsc.client.getEndpoint(tableServiceName, t.Name, params)
	headers := t.tsc.client.getStandardHeaders()
	headers = options.addToHeaders(headers)

	body, length, err := generateTableACLPayload(tap)
	if err != nil {
		return err
	}
	headers["Content-Length"] = strconv.Itoa(length)

	resp, err := t.tsc.client.exec(http.MethodPut, uri, headers, body, t.tsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)

	return checkRespCode(resp, []int{http.StatusNoContent})
}

func generateTableACLPayload(policies []TableAccessPolicy) (io.Reader, int, error) {
	sil := SignedIdentifiers{
		SignedIdentifiers: []SignedIdentifier{},
	}
	for _, tap := range policies {
		permission := generateTablePermissions(&tap)
		signedIdentifier := convertAccessPolicyToXMLStructs(tap.ID, tap.StartTime, tap.ExpiryTime, permission)
		sil.SignedIdentifiers = append(sil.SignedIdentifiers, signedIdentifier)
	}
	return xmlMarshal(sil)
}

// GetPermissions gets the table ACL permissions
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/get-table-acl
func (t *Table) GetPermissions(timeout int, options *TableOptions) ([]TableAccessPolicy, error) {
	params := url.Values{"comp": {"acl"},
		"timeout": {strconv.Itoa(int(timeout))},
	}

	uri := t.tsc.client.getEndpoint(tableServiceName, t.Name, params)
	headers := t.tsc.client.getStandardHeaders()
	headers = options.addToHeaders(headers)

	resp, err := t.tsc.client.exec(http.MethodGet, uri, headers, nil, t.tsc.auth)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if err = checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	var ap AccessPolicy
	err = xmlUnmarshal(resp.Body, &ap.SignedIdentifiersList)
	if err != nil {
		return nil, err
	}
	return updateTableAccessPolicy(ap), nil
}

func (t *Table) queryEntities(uri string, headers map[string]string, ml MetadataLevel) (*EntityQueryResult, error) {
	headers = mergeHeaders(headers, t.tsc.client.getStandardHeaders())
	if ml != EmptyPayload {
		headers[headerAccept] = string(ml)
	}

	resp, err := t.tsc.client.exec(http.MethodGet, uri, headers, nil, t.tsc.auth)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if err = checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var entities EntityQueryResult
	err = json.Unmarshal(data, &entities)
	if err != nil {
		return nil, err
	}

	for i := range entities.Entities {
		entities.Entities[i].Table = t
	}
	entities.table = t

	contToken := extractContinuationTokenFromHeaders(resp.Header)
	if contToken == nil {
		entities.NextLink = nil
	} else {
		originalURI, err := url.Parse(uri)
		if err != nil {
			return nil, err
		}
		v := originalURI.Query()
		if contToken.NextPartitionKey != "" {
			v.Set(nextPartitionKeyQueryParameter, contToken.NextPartitionKey)
		}
		if contToken.NextRowKey != "" {
			v.Set(nextRowKeyQueryParameter, contToken.NextRowKey)
		}
		newURI := t.tsc.client.getEndpoint(tableServiceName, t.buildPath(), v)
		entities.NextLink = &newURI
		entities.ml = ml
	}

	return &entities, nil
}

func extractContinuationTokenFromHeaders(h http.Header) *continuationToken {
	ct := continuationToken{
		NextPartitionKey: h.Get(headerNextPartitionKey),
		NextRowKey:       h.Get(headerNextRowKey),
	}

	if ct.NextPartitionKey != "" || ct.NextRowKey != "" {
		return &ct
	}
	return nil
}

func updateTableAccessPolicy(ap AccessPolicy) []TableAccessPolicy {
	taps := []TableAccessPolicy{}
	for _, policy := range ap.SignedIdentifiersList.SignedIdentifiers {
		tap := TableAccessPolicy{
			ID:         policy.ID,
			StartTime:  policy.AccessPolicy.StartTime,
			ExpiryTime: policy.AccessPolicy.ExpiryTime,
		}
		tap.CanRead = updatePermissions(policy.AccessPolicy.Permission, "r")
		tap.CanAppend = updatePermissions(policy.AccessPolicy.Permission, "a")
		tap.CanUpdate = updatePermissions(policy.AccessPolicy.Permission, "u")
		tap.CanDelete = updatePermissions(policy.AccessPolicy.Permission, "d")

		taps = append(taps, tap)
	}
	return taps
}

func generateTablePermissions(tap *TableAccessPolicy) (permissions string) {
	// generate the permissions string (raud).
	// still want the end user API to have bool flags.
	permissions = ""

	if tap.CanRead {
		permissions += "r"
	}

	if tap.CanAppend {
		permissions += "a"
	}

	if tap.CanUpdate {
		permissions += "u"
	}

	if tap.CanDelete {
		permissions += "d"
	}
	return permissions
}
