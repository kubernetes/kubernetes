package storage

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
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
)

const (
	headerAccept          = "Accept"
	headerEtag            = "Etag"
	headerPrefer          = "Prefer"
	headerXmsContinuation = "x-ms-Continuation-NextTableName"
)

// TableServiceClient contains operations for Microsoft Azure Table Storage
// Service.
type TableServiceClient struct {
	client Client
	auth   authentication
}

// TableOptions includes options for some table operations
type TableOptions struct {
	RequestID string
}

func (options *TableOptions) addToHeaders(h map[string]string) map[string]string {
	if options != nil {
		h = addToHeaders(h, "x-ms-client-request-id", options.RequestID)
	}
	return h
}

// QueryNextLink includes information for getting the next page of
// results in query operations
type QueryNextLink struct {
	NextLink *string
	ml       MetadataLevel
}

// GetServiceProperties gets the properties of your storage account's table service.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-table-service-properties
func (t *TableServiceClient) GetServiceProperties() (*ServiceProperties, error) {
	return t.client.getServiceProperties(tableServiceName, t.auth)
}

// SetServiceProperties sets the properties of your storage account's table service.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/set-table-service-properties
func (t *TableServiceClient) SetServiceProperties(props ServiceProperties) error {
	return t.client.setServiceProperties(props, tableServiceName, t.auth)
}

// GetTableReference returns a Table object for the specified table name.
func (t *TableServiceClient) GetTableReference(name string) *Table {
	return &Table{
		tsc:  t,
		Name: name,
	}
}

// QueryTablesOptions includes options for some table operations
type QueryTablesOptions struct {
	Top       uint
	Filter    string
	RequestID string
}

func (options *QueryTablesOptions) getParameters() (url.Values, map[string]string) {
	query := url.Values{}
	headers := map[string]string{}
	if options != nil {
		if options.Top > 0 {
			query.Add(OdataTop, strconv.FormatUint(uint64(options.Top), 10))
		}
		if options.Filter != "" {
			query.Add(OdataFilter, options.Filter)
		}
		headers = addToHeaders(headers, "x-ms-client-request-id", options.RequestID)
	}
	return query, headers
}

// QueryTables returns the tables in the storage account.
// You can use query options defined by the OData Protocol specification.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/query-tables
func (t *TableServiceClient) QueryTables(ml MetadataLevel, options *QueryTablesOptions) (*TableQueryResult, error) {
	query, headers := options.getParameters()
	uri := t.client.getEndpoint(tableServiceName, tablesURIPath, query)
	return t.queryTables(uri, headers, ml)
}

// NextResults returns the next page of results
// from a QueryTables or a NextResults operation.
//
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/query-tables
// See https://docs.microsoft.com/rest/api/storageservices/fileservices/query-timeout-and-pagination
func (tqr *TableQueryResult) NextResults(options *TableOptions) (*TableQueryResult, error) {
	if tqr == nil {
		return nil, errNilPreviousResult
	}
	if tqr.NextLink == nil {
		return nil, errNilNextLink
	}
	headers := options.addToHeaders(map[string]string{})

	return tqr.tsc.queryTables(*tqr.NextLink, headers, tqr.ml)
}

// TableQueryResult contains the response from
// QueryTables and QueryTablesNextResults functions.
type TableQueryResult struct {
	OdataMetadata string  `json:"odata.metadata"`
	Tables        []Table `json:"value"`
	QueryNextLink
	tsc *TableServiceClient
}

func (t *TableServiceClient) queryTables(uri string, headers map[string]string, ml MetadataLevel) (*TableQueryResult, error) {
	if ml == EmptyPayload {
		return nil, errEmptyPayload
	}
	headers = mergeHeaders(headers, t.client.getStandardHeaders())
	headers[headerAccept] = string(ml)

	resp, err := t.client.exec(http.MethodGet, uri, headers, nil, t.auth)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if err := checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out TableQueryResult
	err = json.Unmarshal(respBody, &out)
	if err != nil {
		return nil, err
	}

	for i := range out.Tables {
		out.Tables[i].tsc = t
	}
	out.tsc = t

	nextLink := resp.Header.Get(http.CanonicalHeaderKey(headerXmsContinuation))
	if nextLink == "" {
		out.NextLink = nil
	} else {
		originalURI, err := url.Parse(uri)
		if err != nil {
			return nil, err
		}
		v := originalURI.Query()
		v.Set(nextTableQueryParameter, nextLink)
		newURI := t.client.getEndpoint(tableServiceName, tablesURIPath, v)
		out.NextLink = &newURI
		out.ml = ml
	}

	return &out, nil
}

func addBodyRelatedHeaders(h map[string]string, length int) map[string]string {
	h[headerContentType] = "application/json"
	h[headerContentLength] = fmt.Sprintf("%v", length)
	h[headerAcceptCharset] = "UTF-8"
	return h
}

func addReturnContentHeaders(h map[string]string, ml MetadataLevel) map[string]string {
	if ml != EmptyPayload {
		h[headerPrefer] = "return-content"
		h[headerAccept] = string(ml)
	} else {
		h[headerPrefer] = "return-no-content"
		// From API version 2015-12-11 onwards, Accept header is required
		h[headerAccept] = string(NoMetadata)
	}
	return h
}
