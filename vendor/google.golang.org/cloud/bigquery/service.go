// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import (
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"golang.org/x/net/context"
	bq "google.golang.org/api/bigquery/v2"
)

// service provides an internal abstraction to isolate the generated
// BigQuery API; most of this package uses this interface instead.
// The single implementation, *bigqueryService, contains all the knowledge
// of the generated BigQuery API.
type service interface {
	// Jobs
	insertJob(ctx context.Context, job *bq.Job, projectId string) (*Job, error)
	getJobType(ctx context.Context, projectId, jobID string) (jobType, error)
	jobStatus(ctx context.Context, projectId, jobID string) (*JobStatus, error)

	// Queries

	// readQuery reads data resulting from a query job. If the job is not
	// yet complete, an errIncompleteJob is returned. readQuery may be
	// called repeatedly to wait for results indefinitely.
	readQuery(ctx context.Context, conf *readQueryConf, pageToken string) (*readDataResult, error)

	readTabledata(ctx context.Context, conf *readTableConf, pageToken string) (*readDataResult, error)

	// Tables
	createTable(ctx context.Context, conf *createTableConf) error
	getTableMetadata(ctx context.Context, projectID, datasetID, tableID string) (*TableMetadata, error)
	deleteTable(ctx context.Context, projectID, datasetID, tableID string) error
	listTables(ctx context.Context, projectID, datasetID, pageToken string) ([]*Table, string, error)
	patchTable(ctx context.Context, projectID, datasetID, tableID string, conf *patchTableConf) (*TableMetadata, error)
}

type bigqueryService struct {
	s *bq.Service
}

func newBigqueryService(client *http.Client) (*bigqueryService, error) {
	s, err := bq.New(client)
	if err != nil {
		return nil, fmt.Errorf("constructing bigquery client: %v", err)
	}

	return &bigqueryService{s: s}, nil
}

// getPages calls the supplied getPage function repeatedly until there are no pages left to get.
// token is the token of the initial page to start from.  Use an empty string to start from the beginning.
func getPages(token string, getPage func(token string) (nextToken string, err error)) error {
	for {
		var err error
		token, err = getPage(token)
		if err != nil {
			return err
		}
		if token == "" {
			return nil
		}
	}
}

func (s *bigqueryService) insertJob(ctx context.Context, job *bq.Job, projectID string) (*Job, error) {
	res, err := s.s.Jobs.Insert(projectID, job).Context(ctx).Do()
	if err != nil {
		return nil, err
	}
	return &Job{service: s, projectID: projectID, jobID: res.JobReference.JobId}, nil
}

type pagingConf struct {
	recordsPerRequest    int64
	setRecordsPerRequest bool

	startIndex uint64
}

type readTableConf struct {
	projectID, datasetID, tableID string
	paging                        pagingConf
	schema                        Schema // lazily initialized when the first page of data is fetched.
}

type readDataResult struct {
	pageToken string
	rows      [][]Value
	totalRows uint64
	schema    Schema
}

type readQueryConf struct {
	projectID, jobID string
	paging           pagingConf
}

func (s *bigqueryService) readTabledata(ctx context.Context, conf *readTableConf, pageToken string) (*readDataResult, error) {
	// Prepare request to fetch one page of table data.
	req := s.s.Tabledata.List(conf.projectID, conf.datasetID, conf.tableID)

	if pageToken != "" {
		req.PageToken(pageToken)
	} else {
		req.StartIndex(conf.paging.startIndex)
	}

	if conf.paging.setRecordsPerRequest {
		req.MaxResults(conf.paging.recordsPerRequest)
	}

	// Fetch the table schema in the background, if necessary.
	var schemaErr error
	var schemaFetch sync.WaitGroup
	if conf.schema == nil {
		schemaFetch.Add(1)
		go func() {
			defer schemaFetch.Done()
			var t *bq.Table
			t, schemaErr = s.s.Tables.Get(conf.projectID, conf.datasetID, conf.tableID).
				Fields("schema").
				Context(ctx).
				Do()
			if schemaErr == nil && t.Schema != nil {
				conf.schema = convertTableSchema(t.Schema)
			}
		}()
	}

	res, err := req.Context(ctx).Do()
	if err != nil {
		return nil, err
	}

	schemaFetch.Wait()
	if schemaErr != nil {
		return nil, schemaErr
	}

	result := &readDataResult{
		pageToken: res.PageToken,
		totalRows: uint64(res.TotalRows),
		schema:    conf.schema,
	}
	result.rows, err = convertRows(res.Rows, conf.schema)
	if err != nil {
		return nil, err
	}
	return result, nil
}

var errIncompleteJob = errors.New("internal error: query results not available because job is not complete")

// getQueryResultsTimeout controls the maximum duration of a request to the
// BigQuery GetQueryResults endpoint.  Setting a long timeout here does not
// cause increased overall latency, as results are returned as soon as they are
// available.
const getQueryResultsTimeout = time.Minute

func (s *bigqueryService) readQuery(ctx context.Context, conf *readQueryConf, pageToken string) (*readDataResult, error) {
	req := s.s.Jobs.GetQueryResults(conf.projectID, conf.jobID).
		TimeoutMs(getQueryResultsTimeout.Nanoseconds() / 1e6)

	if pageToken != "" {
		req.PageToken(pageToken)
	} else {
		req.StartIndex(conf.paging.startIndex)
	}

	if conf.paging.setRecordsPerRequest {
		req.MaxResults(conf.paging.recordsPerRequest)
	}

	res, err := req.Context(ctx).Do()
	if err != nil {
		return nil, err
	}

	if !res.JobComplete {
		return nil, errIncompleteJob
	}
	schema := convertTableSchema(res.Schema)
	result := &readDataResult{
		pageToken: res.PageToken,
		totalRows: res.TotalRows,
		schema:    schema,
	}
	result.rows, err = convertRows(res.Rows, schema)
	if err != nil {
		return nil, err
	}
	return result, nil
}

type jobType int

const (
	copyJobType jobType = iota
	extractJobType
	loadJobType
	queryJobType
)

func (s *bigqueryService) getJobType(ctx context.Context, projectID, jobID string) (jobType, error) {
	res, err := s.s.Jobs.Get(projectID, jobID).
		Fields("configuration").
		Context(ctx).
		Do()

	if err != nil {
		return 0, err
	}

	switch {
	case res.Configuration.Copy != nil:
		return copyJobType, nil
	case res.Configuration.Extract != nil:
		return extractJobType, nil
	case res.Configuration.Load != nil:
		return loadJobType, nil
	case res.Configuration.Query != nil:
		return queryJobType, nil
	default:
		return 0, errors.New("unknown job type")
	}
}

func (s *bigqueryService) jobStatus(ctx context.Context, projectID, jobID string) (*JobStatus, error) {
	res, err := s.s.Jobs.Get(projectID, jobID).
		Fields("status"). // Only fetch what we need.
		Context(ctx).
		Do()
	if err != nil {
		return nil, err
	}
	return jobStatusFromProto(res.Status)
}

var stateMap = map[string]State{"PENDING": Pending, "RUNNING": Running, "DONE": Done}

func jobStatusFromProto(status *bq.JobStatus) (*JobStatus, error) {
	state, ok := stateMap[status.State]
	if !ok {
		return nil, fmt.Errorf("unexpected job state: %v", status.State)
	}

	newStatus := &JobStatus{
		State: state,
		err:   nil,
	}
	if err := errorFromErrorProto(status.ErrorResult); state == Done && err != nil {
		newStatus.err = err
	}

	for _, ep := range status.Errors {
		newStatus.Errors = append(newStatus.Errors, errorFromErrorProto(ep))
	}
	return newStatus, nil
}

// listTables returns a subset of tables that belong to a dataset, and a token for fetching the next subset.
func (s *bigqueryService) listTables(ctx context.Context, projectID, datasetID, pageToken string) ([]*Table, string, error) {
	var tables []*Table
	res, err := s.s.Tables.List(projectID, datasetID).
		PageToken(pageToken).
		Context(ctx).
		Do()
	if err != nil {
		return nil, "", err
	}
	for _, t := range res.Tables {
		tables = append(tables, convertListedTable(t))
	}
	return tables, res.NextPageToken, nil
}

type createTableConf struct {
	projectID, datasetID, tableID string
	expiration                    time.Time
	viewQuery                     string
}

// createTable creates a table in the BigQuery service.
// expiration is an optional time after which the table will be deleted and its storage reclaimed.
// If viewQuery is non-empty, the created table will be of type VIEW.
// Note: expiration can only be set during table creation.
// Note: after table creation, a view can be modified only if its table was initially created with a view.
func (s *bigqueryService) createTable(ctx context.Context, conf *createTableConf) error {
	table := &bq.Table{
		TableReference: &bq.TableReference{
			ProjectId: conf.projectID,
			DatasetId: conf.datasetID,
			TableId:   conf.tableID,
		},
	}
	if !conf.expiration.IsZero() {
		table.ExpirationTime = conf.expiration.UnixNano() / 1000
	}
	if conf.viewQuery != "" {
		table.View = &bq.ViewDefinition{
			Query: conf.viewQuery,
		}
	}

	_, err := s.s.Tables.Insert(conf.projectID, conf.datasetID, table).Context(ctx).Do()
	return err
}

func (s *bigqueryService) getTableMetadata(ctx context.Context, projectID, datasetID, tableID string) (*TableMetadata, error) {
	table, err := s.s.Tables.Get(projectID, datasetID, tableID).Context(ctx).Do()
	if err != nil {
		return nil, err
	}
	return bqTableToMetadata(table), nil
}

func (s *bigqueryService) deleteTable(ctx context.Context, projectID, datasetID, tableID string) error {
	return s.s.Tables.Delete(projectID, datasetID, tableID).Context(ctx).Do()
}

func bqTableToMetadata(t *bq.Table) *TableMetadata {
	md := &TableMetadata{
		Description: t.Description,
		Name:        t.FriendlyName,
		Type:        TableType(t.Type),
		ID:          t.Id,
		NumBytes:    t.NumBytes,
		NumRows:     t.NumRows,
	}
	if t.ExpirationTime != 0 {
		md.ExpirationTime = time.Unix(0, t.ExpirationTime*1e6)
	}
	if t.CreationTime != 0 {
		md.CreationTime = time.Unix(0, t.CreationTime*1e6)
	}
	if t.LastModifiedTime != 0 {
		md.LastModifiedTime = time.Unix(0, int64(t.LastModifiedTime*1e6))
	}
	if t.Schema != nil {
		md.Schema = convertTableSchema(t.Schema)
	}
	if t.View != nil {
		md.View = t.View.Query
	}

	return md
}

func convertListedTable(t *bq.TableListTables) *Table {
	return &Table{
		ProjectID: t.TableReference.ProjectId,
		DatasetID: t.TableReference.DatasetId,
		TableID:   t.TableReference.TableId,
	}
}

// patchTableConf contains fields to be patched.
type patchTableConf struct {
	// These fields are omitted from the patch operation if nil.
	Description *string
	Name        *string
}

func (s *bigqueryService) patchTable(ctx context.Context, projectID, datasetID, tableID string, conf *patchTableConf) (*TableMetadata, error) {
	t := &bq.Table{}
	forceSend := func(field string) {
		t.ForceSendFields = append(t.ForceSendFields, field)
	}

	if conf.Description != nil {
		t.Description = *conf.Description
		forceSend("Description")
	}
	if conf.Name != nil {
		t.FriendlyName = *conf.Name
		forceSend("FriendlyName")
	}
	table, err := s.s.Tables.Patch(projectID, datasetID, tableID, t).
		Context(ctx).
		Do()
	if err != nil {
		return nil, err
	}
	return bqTableToMetadata(table), nil
}
