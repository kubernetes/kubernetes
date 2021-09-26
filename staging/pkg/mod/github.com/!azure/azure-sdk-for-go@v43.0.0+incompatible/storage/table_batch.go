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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"sort"
	"strings"
)

// Operation type. Insert, Delete, Replace etc.
type Operation int

// consts for batch operations.
const (
	InsertOp          = Operation(1)
	DeleteOp          = Operation(2)
	ReplaceOp         = Operation(3)
	MergeOp           = Operation(4)
	InsertOrReplaceOp = Operation(5)
	InsertOrMergeOp   = Operation(6)
)

// BatchEntity used for tracking Entities to operate on and
// whether operations (replace/merge etc) should be forced.
// Wrapper for regular Entity with additional data specific for the entity.
type BatchEntity struct {
	*Entity
	Force bool
	Op    Operation
}

// TableBatch stores all the entities that will be operated on during a batch process.
// Entities can be inserted, replaced or deleted.
type TableBatch struct {
	BatchEntitySlice []BatchEntity

	// reference to table we're operating on.
	Table *Table
}

// defaultChangesetHeaders for changeSets
var defaultChangesetHeaders = map[string]string{
	"Accept":       "application/json;odata=minimalmetadata",
	"Content-Type": "application/json",
	"Prefer":       "return-no-content",
}

// NewBatch return new TableBatch for populating.
func (t *Table) NewBatch() *TableBatch {
	return &TableBatch{
		Table: t,
	}
}

// InsertEntity adds an entity in preparation for a batch insert.
func (t *TableBatch) InsertEntity(entity *Entity) {
	be := BatchEntity{Entity: entity, Force: false, Op: InsertOp}
	t.BatchEntitySlice = append(t.BatchEntitySlice, be)
}

// InsertOrReplaceEntity adds an entity in preparation for a batch insert or replace.
func (t *TableBatch) InsertOrReplaceEntity(entity *Entity, force bool) {
	be := BatchEntity{Entity: entity, Force: false, Op: InsertOrReplaceOp}
	t.BatchEntitySlice = append(t.BatchEntitySlice, be)
}

// InsertOrReplaceEntityByForce adds an entity in preparation for a batch insert or replace. Forces regardless of ETag
func (t *TableBatch) InsertOrReplaceEntityByForce(entity *Entity) {
	t.InsertOrReplaceEntity(entity, true)
}

// InsertOrMergeEntity adds an entity in preparation for a batch insert or merge.
func (t *TableBatch) InsertOrMergeEntity(entity *Entity, force bool) {
	be := BatchEntity{Entity: entity, Force: false, Op: InsertOrMergeOp}
	t.BatchEntitySlice = append(t.BatchEntitySlice, be)
}

// InsertOrMergeEntityByForce adds an entity in preparation for a batch insert or merge. Forces regardless of ETag
func (t *TableBatch) InsertOrMergeEntityByForce(entity *Entity) {
	t.InsertOrMergeEntity(entity, true)
}

// ReplaceEntity adds an entity in preparation for a batch replace.
func (t *TableBatch) ReplaceEntity(entity *Entity) {
	be := BatchEntity{Entity: entity, Force: false, Op: ReplaceOp}
	t.BatchEntitySlice = append(t.BatchEntitySlice, be)
}

// DeleteEntity adds an entity in preparation for a batch delete
func (t *TableBatch) DeleteEntity(entity *Entity, force bool) {
	be := BatchEntity{Entity: entity, Force: false, Op: DeleteOp}
	t.BatchEntitySlice = append(t.BatchEntitySlice, be)
}

// DeleteEntityByForce adds an entity in preparation for a batch delete. Forces regardless of ETag
func (t *TableBatch) DeleteEntityByForce(entity *Entity, force bool) {
	t.DeleteEntity(entity, true)
}

// MergeEntity adds an entity in preparation for a batch merge
func (t *TableBatch) MergeEntity(entity *Entity) {
	be := BatchEntity{Entity: entity, Force: false, Op: MergeOp}
	t.BatchEntitySlice = append(t.BatchEntitySlice, be)
}

// ExecuteBatch executes many table operations in one request to Azure.
// The operations can be combinations of Insert, Delete, Replace and Merge
// Creates the inner changeset body (various operations, Insert, Delete etc) then creates the outer request packet that encompasses
// the changesets.
// As per document https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/performing-entity-group-transactions
func (t *TableBatch) ExecuteBatch() error {

	id, err := newUUID()
	if err != nil {
		return err
	}

	changesetBoundary := fmt.Sprintf("changeset_%s", id.String())
	uri := t.Table.tsc.client.getEndpoint(tableServiceName, "$batch", nil)
	changesetBody, err := t.generateChangesetBody(changesetBoundary)
	if err != nil {
		return err
	}

	id, err = newUUID()
	if err != nil {
		return err
	}

	boundary := fmt.Sprintf("batch_%s", id.String())
	body, err := generateBody(changesetBody, changesetBoundary, boundary)
	if err != nil {
		return err
	}

	headers := t.Table.tsc.client.getStandardHeaders()
	headers[headerContentType] = fmt.Sprintf("multipart/mixed; boundary=%s", boundary)

	resp, err := t.Table.tsc.client.execBatchOperationJSON(http.MethodPost, uri, headers, bytes.NewReader(body.Bytes()), t.Table.tsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp.resp)

	if err = checkRespCode(resp.resp, []int{http.StatusAccepted}); err != nil {

		// check which batch failed.
		operationFailedMessage := t.getFailedOperation(resp.odata.Err.Message.Value)
		requestID, date, version := getDebugHeaders(resp.resp.Header)
		return AzureStorageServiceError{
			StatusCode: resp.resp.StatusCode,
			Code:       resp.odata.Err.Code,
			RequestID:  requestID,
			Date:       date,
			APIVersion: version,
			Message:    operationFailedMessage,
		}
	}

	return nil
}

// getFailedOperation parses the original Azure error string and determines which operation failed
// and generates appropriate message.
func (t *TableBatch) getFailedOperation(errorMessage string) string {
	// errorMessage consists of "number:string" we just need the number.
	sp := strings.Split(errorMessage, ":")
	if len(sp) > 1 {
		msg := fmt.Sprintf("Element %s in the batch returned an unexpected response code.\n%s", sp[0], errorMessage)
		return msg
	}

	// cant parse the message, just return the original message to client
	return errorMessage
}

// generateBody generates the complete body for the batch request.
func generateBody(changeSetBody *bytes.Buffer, changesetBoundary string, boundary string) (*bytes.Buffer, error) {

	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	writer.SetBoundary(boundary)
	h := make(textproto.MIMEHeader)
	h.Set(headerContentType, fmt.Sprintf("multipart/mixed; boundary=%s\r\n", changesetBoundary))
	batchWriter, err := writer.CreatePart(h)
	if err != nil {
		return nil, err
	}
	batchWriter.Write(changeSetBody.Bytes())
	writer.Close()
	return body, nil
}

// generateChangesetBody generates the individual changesets for the various operations within the batch request.
// There is a changeset for Insert, Delete, Merge etc.
func (t *TableBatch) generateChangesetBody(changesetBoundary string) (*bytes.Buffer, error) {

	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	writer.SetBoundary(changesetBoundary)

	for _, be := range t.BatchEntitySlice {
		t.generateEntitySubset(&be, writer)
	}

	writer.Close()
	return body, nil
}

// generateVerb generates the HTTP request VERB required for each changeset.
func generateVerb(op Operation) (string, error) {
	switch op {
	case InsertOp:
		return http.MethodPost, nil
	case DeleteOp:
		return http.MethodDelete, nil
	case ReplaceOp, InsertOrReplaceOp:
		return http.MethodPut, nil
	case MergeOp, InsertOrMergeOp:
		return "MERGE", nil
	default:
		return "", errors.New("Unable to detect operation")
	}
}

// generateQueryPath generates the query path for within the changesets
// For inserts it will just be a table query path (table name)
// but for other operations (modifying an existing entity) then
// the partition/row keys need to be generated.
func (t *TableBatch) generateQueryPath(op Operation, entity *Entity) string {
	if op == InsertOp {
		return entity.Table.buildPath()
	}
	return entity.buildPath()
}

// generateGenericOperationHeaders generates common headers for a given operation.
func generateGenericOperationHeaders(be *BatchEntity) map[string]string {
	retval := map[string]string{}

	for k, v := range defaultChangesetHeaders {
		retval[k] = v
	}

	if be.Op == DeleteOp || be.Op == ReplaceOp || be.Op == MergeOp {
		if be.Force || be.Entity.OdataEtag == "" {
			retval["If-Match"] = "*"
		} else {
			retval["If-Match"] = be.Entity.OdataEtag
		}
	}

	return retval
}

// generateEntitySubset generates body payload for particular batch entity
func (t *TableBatch) generateEntitySubset(batchEntity *BatchEntity, writer *multipart.Writer) error {

	h := make(textproto.MIMEHeader)
	h.Set(headerContentType, "application/http")
	h.Set(headerContentTransferEncoding, "binary")

	verb, err := generateVerb(batchEntity.Op)
	if err != nil {
		return err
	}

	genericOpHeadersMap := generateGenericOperationHeaders(batchEntity)
	queryPath := t.generateQueryPath(batchEntity.Op, batchEntity.Entity)
	uri := t.Table.tsc.client.getEndpoint(tableServiceName, queryPath, nil)

	operationWriter, err := writer.CreatePart(h)
	if err != nil {
		return err
	}

	urlAndVerb := fmt.Sprintf("%s %s HTTP/1.1\r\n", verb, uri)
	operationWriter.Write([]byte(urlAndVerb))
	writeHeaders(genericOpHeadersMap, &operationWriter)
	operationWriter.Write([]byte("\r\n")) // additional \r\n is needed per changeset separating the "headers" and the body.

	// delete operation doesn't need a body.
	if batchEntity.Op != DeleteOp {
		//var e Entity = batchEntity.Entity
		body, err := json.Marshal(batchEntity.Entity)
		if err != nil {
			return err
		}
		operationWriter.Write(body)
	}

	return nil
}

func writeHeaders(h map[string]string, writer *io.Writer) {
	// This way it is guaranteed the headers will be written in a sorted order
	var keys []string
	for k := range h {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		(*writer).Write([]byte(fmt.Sprintf("%s: %s\r\n", k, h[k])))
	}
}
