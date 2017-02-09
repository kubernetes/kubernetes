package storage

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"reflect"
)

// Annotating as secure for gas scanning
/* #nosec */
const (
	partitionKeyNode                    = "PartitionKey"
	rowKeyNode                          = "RowKey"
	tag                                 = "table"
	tagIgnore                           = "-"
	continuationTokenPartitionKeyHeader = "X-Ms-Continuation-Nextpartitionkey"
	continuationTokenRowHeader          = "X-Ms-Continuation-Nextrowkey"
	maxTopParameter                     = 1000
)

type queryTablesResponse struct {
	TableName []struct {
		TableName string `json:"TableName"`
	} `json:"value"`
}

const (
	tableOperationTypeInsert          = iota
	tableOperationTypeUpdate          = iota
	tableOperationTypeMerge           = iota
	tableOperationTypeInsertOrReplace = iota
	tableOperationTypeInsertOrMerge   = iota
)

type tableOperation int

// TableEntity interface specifies
// the functions needed to support
// marshaling and unmarshaling into
// Azure Tables. The struct must only contain
// simple types because Azure Tables do not
// support hierarchy.
type TableEntity interface {
	PartitionKey() string
	RowKey() string
	SetPartitionKey(string) error
	SetRowKey(string) error
}

// ContinuationToken is an opaque (ie not useful to inspect)
// struct that Get... methods can return if there are more
// entries to be returned than the ones already
// returned. Just pass it to the same function to continue
// receiving the remaining entries.
type ContinuationToken struct {
	NextPartitionKey string
	NextRowKey       string
}

type getTableEntriesResponse struct {
	Elements []map[string]interface{} `json:"value"`
}

// QueryTableEntities queries the specified table and returns the unmarshaled
// entities of type retType.
// top parameter limits the returned entries up to top. Maximum top
// allowed by Azure API is 1000. In case there are more than top entries to be
// returned the function will return a non nil *ContinuationToken. You can call the
// same function again passing the received ContinuationToken as previousContToken
// parameter in order to get the following entries. The query parameter
// is the odata query. To retrieve all the entries pass the empty string.
// The function returns a pointer to a TableEntity slice, the *ContinuationToken
// if there are more entries to be returned and an error in case something went
// wrong.
//
// Example:
// 		entities, cToken, err = tSvc.QueryTableEntities("table", cToken, reflect.TypeOf(entity), 20, "")
func (c *TableServiceClient) QueryTableEntities(tableName AzureTable, previousContToken *ContinuationToken, retType reflect.Type, top int, query string) ([]TableEntity, *ContinuationToken, error) {
	if top > maxTopParameter {
		return nil, nil, fmt.Errorf("top accepts at maximum %d elements. Requested %d instead", maxTopParameter, top)
	}

	uri := c.client.getEndpoint(tableServiceName, pathForTable(tableName), url.Values{})
	uri += fmt.Sprintf("?$top=%d", top)
	if query != "" {
		uri += fmt.Sprintf("&$filter=%s", url.QueryEscape(query))
	}

	if previousContToken != nil {
		uri += fmt.Sprintf("&NextPartitionKey=%s&NextRowKey=%s", previousContToken.NextPartitionKey, previousContToken.NextRowKey)
	}

	headers := c.getStandardHeaders()

	headers["Content-Length"] = "0"

	resp, err := c.client.execInternalJSON(http.MethodGet, uri, headers, nil, c.auth)

	if err != nil {
		return nil, nil, err
	}

	contToken := extractContinuationTokenFromHeaders(resp.headers)

	defer resp.body.Close()

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return nil, contToken, err
	}

	retEntries, err := deserializeEntity(retType, resp.body)
	if err != nil {
		return nil, contToken, err
	}

	return retEntries, contToken, nil
}

// InsertEntity inserts an entity in the specified table.
// The function fails if there is an entity with the same
// PartitionKey and RowKey in the table.
func (c *TableServiceClient) InsertEntity(table AzureTable, entity TableEntity) error {
	if sc, err := c.execTable(table, entity, false, http.MethodPost); err != nil {
		return checkRespCode(sc, []int{http.StatusCreated})
	}

	return nil
}

func (c *TableServiceClient) execTable(table AzureTable, entity TableEntity, specifyKeysInURL bool, method string) (int, error) {
	uri := c.client.getEndpoint(tableServiceName, pathForTable(table), url.Values{})
	if specifyKeysInURL {
		uri += fmt.Sprintf("(PartitionKey='%s',RowKey='%s')", url.QueryEscape(entity.PartitionKey()), url.QueryEscape(entity.RowKey()))
	}

	headers := c.getStandardHeaders()

	var buf bytes.Buffer

	if err := injectPartitionAndRowKeys(entity, &buf); err != nil {
		return 0, err
	}

	headers["Content-Length"] = fmt.Sprintf("%d", buf.Len())

	resp, err := c.client.execInternalJSON(method, uri, headers, &buf, c.auth)

	if err != nil {
		return 0, err
	}

	defer resp.body.Close()

	return resp.statusCode, nil
}

// UpdateEntity updates the contents of an entity with the
// one passed as parameter. The function fails if there is no entity
// with the same PartitionKey and RowKey in the table.
func (c *TableServiceClient) UpdateEntity(table AzureTable, entity TableEntity) error {
	if sc, err := c.execTable(table, entity, true, http.MethodPut); err != nil {
		return checkRespCode(sc, []int{http.StatusNoContent})
	}
	return nil
}

// MergeEntity merges the contents of an entity with the
// one passed as parameter.
// The function fails if there is no entity
// with the same PartitionKey and RowKey in the table.
func (c *TableServiceClient) MergeEntity(table AzureTable, entity TableEntity) error {
	if sc, err := c.execTable(table, entity, true, "MERGE"); err != nil {
		return checkRespCode(sc, []int{http.StatusNoContent})
	}
	return nil
}

// DeleteEntityWithoutCheck deletes the entity matching by
// PartitionKey and RowKey. There is no check on IfMatch
// parameter so the entity is always deleted.
// The function fails if there is no entity
// with the same PartitionKey and RowKey in the table.
func (c *TableServiceClient) DeleteEntityWithoutCheck(table AzureTable, entity TableEntity) error {
	return c.DeleteEntity(table, entity, "*")
}

// DeleteEntity deletes the entity matching by
// PartitionKey, RowKey and ifMatch field.
// The function fails if there is no entity
// with the same PartitionKey and RowKey in the table or
// the ifMatch is different.
func (c *TableServiceClient) DeleteEntity(table AzureTable, entity TableEntity, ifMatch string) error {
	uri := c.client.getEndpoint(tableServiceName, pathForTable(table), url.Values{})
	uri += fmt.Sprintf("(PartitionKey='%s',RowKey='%s')", url.QueryEscape(entity.PartitionKey()), url.QueryEscape(entity.RowKey()))

	headers := c.getStandardHeaders()

	headers["Content-Length"] = "0"
	headers["If-Match"] = ifMatch

	resp, err := c.client.execInternalJSON(http.MethodDelete, uri, headers, nil, c.auth)

	if err != nil {
		return err
	}
	defer resp.body.Close()

	if err := checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
		return err
	}

	return nil
}

// InsertOrReplaceEntity inserts an entity in the specified table
// or replaced the existing one.
func (c *TableServiceClient) InsertOrReplaceEntity(table AzureTable, entity TableEntity) error {
	if sc, err := c.execTable(table, entity, true, http.MethodPut); err != nil {
		return checkRespCode(sc, []int{http.StatusNoContent})
	}
	return nil
}

// InsertOrMergeEntity inserts an entity in the specified table
// or merges the existing one.
func (c *TableServiceClient) InsertOrMergeEntity(table AzureTable, entity TableEntity) error {
	if sc, err := c.execTable(table, entity, true, "MERGE"); err != nil {
		return checkRespCode(sc, []int{http.StatusNoContent})
	}
	return nil
}

func injectPartitionAndRowKeys(entity TableEntity, buf *bytes.Buffer) error {
	if err := json.NewEncoder(buf).Encode(entity); err != nil {
		return err
	}

	dec := make(map[string]interface{})
	if err := json.NewDecoder(buf).Decode(&dec); err != nil {
		return err
	}

	// Inject PartitionKey and RowKey
	dec[partitionKeyNode] = entity.PartitionKey()
	dec[rowKeyNode] = entity.RowKey()

	// Remove tagged fields
	// The tag is defined in the const section
	// This is useful to avoid storing the PartitionKey and RowKey twice.
	numFields := reflect.ValueOf(entity).Elem().NumField()
	for i := 0; i < numFields; i++ {
		f := reflect.ValueOf(entity).Elem().Type().Field(i)

		if f.Tag.Get(tag) == tagIgnore {
			// we must look for its JSON name in the dictionary
			// as the user can rename it using a tag
			jsonName := f.Name
			if f.Tag.Get("json") != "" {
				jsonName = f.Tag.Get("json")
			}
			delete(dec, jsonName)
		}
	}

	buf.Reset()

	if err := json.NewEncoder(buf).Encode(&dec); err != nil {
		return err
	}

	return nil
}

func deserializeEntity(retType reflect.Type, reader io.Reader) ([]TableEntity, error) {
	buf := new(bytes.Buffer)

	var ret getTableEntriesResponse
	if err := json.NewDecoder(reader).Decode(&ret); err != nil {
		return nil, err
	}

	tEntries := make([]TableEntity, len(ret.Elements))

	for i, entry := range ret.Elements {

		buf.Reset()
		if err := json.NewEncoder(buf).Encode(entry); err != nil {
			return nil, err
		}

		dec := make(map[string]interface{})
		if err := json.NewDecoder(buf).Decode(&dec); err != nil {
			return nil, err
		}

		var pKey, rKey string
		// strip pk and rk
		for key, val := range dec {
			switch key {
			case partitionKeyNode:
				pKey = val.(string)
			case rowKeyNode:
				rKey = val.(string)
			}
		}

		delete(dec, partitionKeyNode)
		delete(dec, rowKeyNode)

		buf.Reset()
		if err := json.NewEncoder(buf).Encode(dec); err != nil {
			return nil, err
		}

		// Create a empty retType instance
		tEntries[i] = reflect.New(retType.Elem()).Interface().(TableEntity)
		// Popolate it with the values
		if err := json.NewDecoder(buf).Decode(&tEntries[i]); err != nil {
			return nil, err
		}

		// Reset PartitionKey and RowKey
		if err := tEntries[i].SetPartitionKey(pKey); err != nil {
			return nil, err
		}
		if err := tEntries[i].SetRowKey(rKey); err != nil {
			return nil, err
		}
	}

	return tEntries, nil
}

func extractContinuationTokenFromHeaders(h http.Header) *ContinuationToken {
	ct := ContinuationToken{h.Get(continuationTokenPartitionKeyHeader), h.Get(continuationTokenRowHeader)}

	if ct.NextPartitionKey != "" && ct.NextRowKey != "" {
		return &ct
	}
	return nil
}
