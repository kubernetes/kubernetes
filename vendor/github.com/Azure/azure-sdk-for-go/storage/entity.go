package storage

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/satori/uuid"
)

// Annotating as secure for gas scanning
/* #nosec */
const (
	partitionKeyNode  = "PartitionKey"
	rowKeyNode        = "RowKey"
	etagErrorTemplate = "Etag didn't match: %v"
)

var (
	errEmptyPayload      = errors.New("Empty payload is not a valid metadata level for this operation")
	errNilPreviousResult = errors.New("The previous results page is nil")
	errNilNextLink       = errors.New("There are no more pages in this query results")
)

// Entity represents an entity inside an Azure table.
type Entity struct {
	Table         *Table
	PartitionKey  string
	RowKey        string
	TimeStamp     time.Time
	OdataMetadata string
	OdataType     string
	OdataID       string
	OdataEtag     string
	OdataEditLink string
	Properties    map[string]interface{}
}

// GetEntityReference returns an Entity object with the specified
// partition key and row key.
func (t *Table) GetEntityReference(partitionKey, rowKey string) *Entity {
	return &Entity{
		PartitionKey: partitionKey,
		RowKey:       rowKey,
		Table:        t,
	}
}

// EntityOptions includes options for entity operations.
type EntityOptions struct {
	Timeout   uint
	RequestID string `header:"x-ms-client-request-id"`
}

// GetEntityOptions includes options for a get entity operation
type GetEntityOptions struct {
	Select    []string
	RequestID string `header:"x-ms-client-request-id"`
}

// Get gets the referenced entity. Which properties to get can be
// specified using the select option.
// See:
// https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/query-entities
// https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/querying-tables-and-entities
func (e *Entity) Get(timeout uint, ml MetadataLevel, options *GetEntityOptions) error {
	if ml == EmptyPayload {
		return errEmptyPayload
	}
	// RowKey and PartitionKey could be lost if not included in the query
	// As those are the entity identifiers, it is best if they are not lost
	rk := e.RowKey
	pk := e.PartitionKey

	query := url.Values{
		"timeout": {strconv.FormatUint(uint64(timeout), 10)},
	}
	headers := e.Table.tsc.client.getStandardHeaders()
	headers[headerAccept] = string(ml)

	if options != nil {
		if len(options.Select) > 0 {
			query.Add("$select", strings.Join(options.Select, ","))
		}
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}

	uri := e.Table.tsc.client.getEndpoint(tableServiceName, e.buildPath(), query)
	resp, err := e.Table.tsc.client.exec(http.MethodGet, uri, headers, nil, e.Table.tsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return err
	}

	respBody, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return err
	}
	err = json.Unmarshal(respBody, e)
	if err != nil {
		return err
	}
	e.PartitionKey = pk
	e.RowKey = rk

	return nil
}

// Insert inserts the referenced entity in its table.
// The function fails if there is an entity with the same
// PartitionKey and RowKey in the table.
// ml determines the level of detail of metadata in the operation response,
// or no data at all.
// See: https://docs.microsoft.com/rest/api/storageservices/fileservices/insert-entity
func (e *Entity) Insert(ml MetadataLevel, options *EntityOptions) error {
	query, headers := options.getParameters()
	headers = mergeHeaders(headers, e.Table.tsc.client.getStandardHeaders())

	body, err := json.Marshal(e)
	if err != nil {
		return err
	}
	headers = addBodyRelatedHeaders(headers, len(body))
	headers = addReturnContentHeaders(headers, ml)

	uri := e.Table.tsc.client.getEndpoint(tableServiceName, e.Table.buildPath(), query)
	resp, err := e.Table.tsc.client.exec(http.MethodPost, uri, headers, bytes.NewReader(body), e.Table.tsc.auth)
	if err != nil {
		return err
	}
	defer resp.body.Close()

	data, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return err
	}

	if ml != EmptyPayload {
		if err = checkRespCode(resp.statusCode, []int{http.StatusCreated}); err != nil {
			return err
		}
		if err = e.UnmarshalJSON(data); err != nil {
			return err
		}
	} else {
		if err = checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
			return err
		}
	}

	return nil
}

// Update updates the contents of an entity. The function fails if there is no entity
// with the same PartitionKey and RowKey in the table or if the ETag is different
// than the one in Azure.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/update-entity2
func (e *Entity) Update(force bool, options *EntityOptions) error {
	return e.updateMerge(force, http.MethodPut, options)
}

// Merge merges the contents of entity specified with PartitionKey and RowKey
// with the content specified in Properties.
// The function fails if there is no entity with the same PartitionKey and
// RowKey in the table or if the ETag is different than the one in Azure.
// Read more: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/merge-entity
func (e *Entity) Merge(force bool, options *EntityOptions) error {
	return e.updateMerge(force, "MERGE", options)
}

// Delete deletes the entity.
// The function fails if there is no entity with the same PartitionKey and
// RowKey in the table or if the ETag is different than the one in Azure.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/delete-entity1
func (e *Entity) Delete(force bool, options *EntityOptions) error {
	query, headers := options.getParameters()
	headers = mergeHeaders(headers, e.Table.tsc.client.getStandardHeaders())

	headers = addIfMatchHeader(headers, force, e.OdataEtag)
	headers = addReturnContentHeaders(headers, EmptyPayload)

	uri := e.Table.tsc.client.getEndpoint(tableServiceName, e.buildPath(), query)
	resp, err := e.Table.tsc.client.exec(http.MethodDelete, uri, headers, nil, e.Table.tsc.auth)
	if err != nil {
		if resp.statusCode == http.StatusPreconditionFailed {
			return fmt.Errorf(etagErrorTemplate, err)
		}
		return err
	}
	defer readAndCloseBody(resp.body)

	if err = checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
		return err
	}

	return e.updateTimestamp(resp.headers)
}

// InsertOrReplace inserts an entity or replaces the existing one.
// Read more: https://docs.microsoft.com/rest/api/storageservices/fileservices/insert-or-replace-entity
func (e *Entity) InsertOrReplace(options *EntityOptions) error {
	return e.insertOr(http.MethodPut, options)
}

// InsertOrMerge inserts an entity or merges the existing one.
// Read more: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/insert-or-merge-entity
func (e *Entity) InsertOrMerge(options *EntityOptions) error {
	return e.insertOr("MERGE", options)
}

func (e *Entity) buildPath() string {
	return fmt.Sprintf("%s(PartitionKey='%s', RowKey='%s')", e.Table.buildPath(), e.PartitionKey, e.RowKey)
}

// MarshalJSON is a custom marshaller for entity
func (e *Entity) MarshalJSON() ([]byte, error) {
	completeMap := map[string]interface{}{}
	completeMap[partitionKeyNode] = e.PartitionKey
	completeMap[rowKeyNode] = e.RowKey
	for k, v := range e.Properties {
		typeKey := strings.Join([]string{k, OdataTypeSuffix}, "")
		switch t := v.(type) {
		case []byte:
			completeMap[typeKey] = OdataBinary
			completeMap[k] = string(t)
		case time.Time:
			completeMap[typeKey] = OdataDateTime
			completeMap[k] = t.Format(time.RFC3339Nano)
		case uuid.UUID:
			completeMap[typeKey] = OdataGUID
			completeMap[k] = t.String()
		case int64:
			completeMap[typeKey] = OdataInt64
			completeMap[k] = fmt.Sprintf("%v", v)
		default:
			completeMap[k] = v
		}
		if strings.HasSuffix(k, OdataTypeSuffix) {
			if !(completeMap[k] == OdataBinary ||
				completeMap[k] == OdataDateTime ||
				completeMap[k] == OdataGUID ||
				completeMap[k] == OdataInt64) {
				return nil, fmt.Errorf("Odata.type annotation %v value is not valid", k)
			}
			valueKey := strings.TrimSuffix(k, OdataTypeSuffix)
			if _, ok := completeMap[valueKey]; !ok {
				return nil, fmt.Errorf("Odata.type annotation %v defined without value defined", k)
			}
		}
	}
	return json.Marshal(completeMap)
}

// UnmarshalJSON is a custom unmarshaller for entities
func (e *Entity) UnmarshalJSON(data []byte) error {
	errorTemplate := "Deserializing error: %v"

	props := map[string]interface{}{}
	err := json.Unmarshal(data, &props)
	if err != nil {
		return err
	}

	// deselialize metadata
	e.OdataMetadata = stringFromMap(props, "odata.metadata")
	e.OdataType = stringFromMap(props, "odata.type")
	e.OdataID = stringFromMap(props, "odata.id")
	e.OdataEtag = stringFromMap(props, "odata.etag")
	e.OdataEditLink = stringFromMap(props, "odata.editLink")
	e.PartitionKey = stringFromMap(props, partitionKeyNode)
	e.RowKey = stringFromMap(props, rowKeyNode)

	// deserialize timestamp
	timeStamp, ok := props["Timestamp"]
	if ok {
		str, ok := timeStamp.(string)
		if !ok {
			return fmt.Errorf(errorTemplate, "Timestamp casting error")
		}
		t, err := time.Parse(time.RFC3339Nano, str)
		if err != nil {
			return fmt.Errorf(errorTemplate, err)
		}
		e.TimeStamp = t
	}
	delete(props, "Timestamp")
	delete(props, "Timestamp@odata.type")

	// deserialize entity (user defined fields)
	for k, v := range props {
		if strings.HasSuffix(k, OdataTypeSuffix) {
			valueKey := strings.TrimSuffix(k, OdataTypeSuffix)
			str, ok := props[valueKey].(string)
			if !ok {
				return fmt.Errorf(errorTemplate, fmt.Sprintf("%v casting error", v))
			}
			switch v {
			case OdataBinary:
				props[valueKey] = []byte(str)
			case OdataDateTime:
				t, err := time.Parse("2006-01-02T15:04:05Z", str)
				if err != nil {
					return fmt.Errorf(errorTemplate, err)
				}
				props[valueKey] = t
			case OdataGUID:
				props[valueKey] = uuid.FromStringOrNil(str)
			case OdataInt64:
				i, err := strconv.ParseInt(str, 10, 64)
				if err != nil {
					return fmt.Errorf(errorTemplate, err)
				}
				props[valueKey] = i
			default:
				return fmt.Errorf(errorTemplate, fmt.Sprintf("%v is not supported", v))
			}
			delete(props, k)
		}
	}

	e.Properties = props
	return nil
}

func getAndDelete(props map[string]interface{}, key string) interface{} {
	if value, ok := props[key]; ok {
		delete(props, key)
		return value
	}
	return nil
}

func addIfMatchHeader(h map[string]string, force bool, etag string) map[string]string {
	if force {
		h[headerIfMatch] = "*"
	} else {
		h[headerIfMatch] = etag
	}
	return h
}

// updates Etag and timestamp
func (e *Entity) updateEtagAndTimestamp(headers http.Header) error {
	e.OdataEtag = headers.Get(headerEtag)
	return e.updateTimestamp(headers)
}

func (e *Entity) updateTimestamp(headers http.Header) error {
	str := headers.Get(headerDate)
	t, err := time.Parse(time.RFC1123, str)
	if err != nil {
		return fmt.Errorf("Update timestamp error: %v", err)
	}
	e.TimeStamp = t
	return nil
}

func (e *Entity) insertOr(verb string, options *EntityOptions) error {
	query, headers := options.getParameters()
	headers = mergeHeaders(headers, e.Table.tsc.client.getStandardHeaders())

	body, err := json.Marshal(e)
	if err != nil {
		return err
	}
	headers = addBodyRelatedHeaders(headers, len(body))
	headers = addReturnContentHeaders(headers, EmptyPayload)

	uri := e.Table.tsc.client.getEndpoint(tableServiceName, e.buildPath(), query)
	resp, err := e.Table.tsc.client.exec(verb, uri, headers, bytes.NewReader(body), e.Table.tsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	if err = checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
		return err
	}

	return e.updateEtagAndTimestamp(resp.headers)
}

func (e *Entity) updateMerge(force bool, verb string, options *EntityOptions) error {
	query, headers := options.getParameters()
	headers = mergeHeaders(headers, e.Table.tsc.client.getStandardHeaders())

	body, err := json.Marshal(e)
	if err != nil {
		return err
	}
	headers = addBodyRelatedHeaders(headers, len(body))
	headers = addIfMatchHeader(headers, force, e.OdataEtag)
	headers = addReturnContentHeaders(headers, EmptyPayload)

	uri := e.Table.tsc.client.getEndpoint(tableServiceName, e.buildPath(), query)
	resp, err := e.Table.tsc.client.exec(verb, uri, headers, bytes.NewReader(body), e.Table.tsc.auth)
	if err != nil {
		if resp.statusCode == http.StatusPreconditionFailed {
			return fmt.Errorf(etagErrorTemplate, err)
		}
		return err
	}
	defer readAndCloseBody(resp.body)

	if err = checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
		return err
	}

	return e.updateEtagAndTimestamp(resp.headers)
}

func stringFromMap(props map[string]interface{}, key string) string {
	value := getAndDelete(props, key)
	if value != nil {
		return value.(string)
	}
	return ""
}

func (options *EntityOptions) getParameters() (url.Values, map[string]string) {
	query := url.Values{}
	headers := map[string]string{}
	if options != nil {
		query = addTimeout(query, options.Timeout)
		headers = headersFromStruct(*options)
	}
	return query, headers
}
