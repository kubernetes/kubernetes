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
	"fmt"
	"time"

	"golang.org/x/net/context"

	bq "google.golang.org/api/bigquery/v2"
)

// A Table is a reference to a BigQuery table.
type Table struct {
	// ProjectID, DatasetID and TableID may be omitted if the Table is the destination for a query.
	// In this case the result will be stored in an ephemeral table.
	ProjectID string
	DatasetID string
	// TableID must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_).
	// The maximum length is 1,024 characters.
	TableID string

	service service
}

// TableMetadata contains information about a BigQuery table.
type TableMetadata struct {
	Description string // The user-friendly description of this table.
	Name        string // The user-friendly name for this table.
	Schema      Schema
	View        string

	ID   string // An opaque ID uniquely identifying the table.
	Type TableType

	// The time when this table expires. If not set, the table will persist
	// indefinitely. Expired tables will be deleted and their storage reclaimed.
	ExpirationTime time.Time

	CreationTime     time.Time
	LastModifiedTime time.Time

	// The size of the table in bytes.
	// This does not include data that is being buffered during a streaming insert.
	NumBytes int64

	// The number of rows of data in this table.
	// This does not include data that is being buffered during a streaming insert.
	NumRows uint64
}

// Tables is a group of tables. The tables may belong to differing projects or datasets.
type Tables []*Table

// CreateDisposition specifies the circumstances under which destination table will be created.
// Default is CreateIfNeeded.
type TableCreateDisposition string

const (
	// The table will be created if it does not already exist.  Tables are created atomically on successful completion of a job.
	CreateIfNeeded TableCreateDisposition = "CREATE_IF_NEEDED"

	// The table must already exist and will not be automatically created.
	CreateNever TableCreateDisposition = "CREATE_NEVER"
)

func CreateDisposition(disp TableCreateDisposition) Option { return disp }

func (opt TableCreateDisposition) implementsOption() {}

func (opt TableCreateDisposition) customizeLoad(conf *bq.JobConfigurationLoad, projectID string) {
	conf.CreateDisposition = string(opt)
}

func (opt TableCreateDisposition) customizeCopy(conf *bq.JobConfigurationTableCopy, projectID string) {
	conf.CreateDisposition = string(opt)
}

func (opt TableCreateDisposition) customizeQuery(conf *bq.JobConfigurationQuery, projectID string) {
	conf.CreateDisposition = string(opt)
}

// TableWriteDisposition specifies how existing data in a destination table is treated.
// Default is WriteAppend.
type TableWriteDisposition string

const (
	// Data will be appended to any existing data in the destination table.
	// Data is appended atomically on successful completion of a job.
	WriteAppend TableWriteDisposition = "WRITE_APPEND"

	// Existing data in the destination table will be overwritten.
	// Data is overwritten atomically on successful completion of a job.
	WriteTruncate TableWriteDisposition = "WRITE_TRUNCATE"

	// Writes will fail if the destination table already contains data.
	WriteEmpty TableWriteDisposition = "WRITE_EMPTY"
)

func WriteDisposition(disp TableWriteDisposition) Option { return disp }

func (opt TableWriteDisposition) implementsOption() {}

func (opt TableWriteDisposition) customizeLoad(conf *bq.JobConfigurationLoad, projectID string) {
	conf.WriteDisposition = string(opt)
}

func (opt TableWriteDisposition) customizeCopy(conf *bq.JobConfigurationTableCopy, projectID string) {
	conf.WriteDisposition = string(opt)
}

func (opt TableWriteDisposition) customizeQuery(conf *bq.JobConfigurationQuery, projectID string) {
	conf.WriteDisposition = string(opt)
}

// TableType is the type of table.
type TableType string

const (
	RegularTable TableType = "TABLE"
	ViewTable    TableType = "VIEW"
)

func (t *Table) implementsSource()      {}
func (t *Table) implementsReadSource()  {}
func (t *Table) implementsDestination() {}
func (ts Tables) implementsSource()     {}

func (t *Table) tableRefProto() *bq.TableReference {
	return &bq.TableReference{
		ProjectId: t.ProjectID,
		DatasetId: t.DatasetID,
		TableId:   t.TableID,
	}
}

// FullyQualifiedName returns the ID of the table in projectID:datasetID.tableID format.
func (t *Table) FullyQualifiedName() string {
	return fmt.Sprintf("%s:%s.%s", t.ProjectID, t.DatasetID, t.TableID)
}

// implicitTable reports whether Table is an empty placeholder, which signifies that a new table should be created with an auto-generated Table ID.
func (t *Table) implicitTable() bool {
	return t.ProjectID == "" && t.DatasetID == "" && t.TableID == ""
}

func (t *Table) customizeLoadDst(conf *bq.JobConfigurationLoad, projectID string) {
	conf.DestinationTable = t.tableRefProto()
}

func (t *Table) customizeExtractSrc(conf *bq.JobConfigurationExtract, projectID string) {
	conf.SourceTable = t.tableRefProto()
}

func (t *Table) customizeCopyDst(conf *bq.JobConfigurationTableCopy, projectID string) {
	conf.DestinationTable = t.tableRefProto()
}

func (ts Tables) customizeCopySrc(conf *bq.JobConfigurationTableCopy, projectID string) {
	for _, t := range ts {
		conf.SourceTables = append(conf.SourceTables, t.tableRefProto())
	}
}

func (t *Table) customizeQueryDst(conf *bq.JobConfigurationQuery, projectID string) {
	if !t.implicitTable() {
		conf.DestinationTable = t.tableRefProto()
	}
}

func (t *Table) customizeReadSrc(cursor *readTableConf) {
	cursor.projectID = t.ProjectID
	cursor.datasetID = t.DatasetID
	cursor.tableID = t.TableID
}

// OpenTable creates a handle to an existing BigQuery table.  If the table does not already exist, subsequent uses of the *Table will fail.
func (c *Client) OpenTable(projectID, datasetID, tableID string) *Table {
	return &Table{ProjectID: projectID, DatasetID: datasetID, TableID: tableID, service: c.service}
}

// CreateTable creates a table in the BigQuery service and returns a handle to it.
func (c *Client) CreateTable(ctx context.Context, projectID, datasetID, tableID string, options ...CreateTableOption) (*Table, error) {
	conf := &createTableConf{
		projectID: projectID,
		datasetID: datasetID,
		tableID:   tableID,
	}
	for _, o := range options {
		o.customizeCreateTable(conf)
	}
	if err := c.service.createTable(ctx, conf); err != nil {
		return nil, err
	}
	return &Table{ProjectID: projectID, DatasetID: datasetID, TableID: tableID, service: c.service}, nil
}

// Metadata fetches the metadata for the table.
func (t *Table) Metadata(ctx context.Context) (*TableMetadata, error) {
	return t.service.getTableMetadata(ctx, t.ProjectID, t.DatasetID, t.TableID)
}

// Delete deletes the table.
func (t *Table) Delete(ctx context.Context) error {
	return t.service.deleteTable(ctx, t.ProjectID, t.DatasetID, t.TableID)
}

// A CreateTableOption is an optional argument to CreateTable.
type CreateTableOption interface {
	customizeCreateTable(*createTableConf)
}

type tableExpiration time.Time

// TableExpiration returns a CreateTableOption which will cause the created table to be deleted after the expiration time.
func TableExpiration(exp time.Time) CreateTableOption { return tableExpiration(exp) }

func (opt tableExpiration) customizeCreateTable(conf *createTableConf) {
	conf.expiration = time.Time(opt)
}

type viewQuery string

// ViewQuery returns a CreateTableOption that causes the created table to be a virtual table defined by the supplied query.
// For more information see: https://cloud.google.com/bigquery/querying-data#views
func ViewQuery(query string) CreateTableOption { return viewQuery(query) }

func (opt viewQuery) customizeCreateTable(conf *createTableConf) {
	conf.viewQuery = string(opt)
}

// TableMetadataPatch represents a set of changes to a table's metadata.
type TableMetadataPatch struct {
	s                             service
	projectID, datasetID, tableID string
	conf                          patchTableConf
}

// Patch returns a *TableMetadataPatch, which can be used to modify specific Table metadata fields.
// In order to apply the changes, the TableMetadataPatch's Apply method must be called.
func (t *Table) Patch() *TableMetadataPatch {
	return &TableMetadataPatch{
		s:         t.service,
		projectID: t.ProjectID,
		datasetID: t.DatasetID,
		tableID:   t.TableID,
	}
}

// Description sets the table description.
func (p *TableMetadataPatch) Description(desc string) {
	p.conf.Description = &desc
}

// Name sets the table name.
func (p *TableMetadataPatch) Name(name string) {
	p.conf.Name = &name
}

// TODO(mcgreevy): support patching the schema.

// Apply applies the patch operation.
func (p *TableMetadataPatch) Apply(ctx context.Context) (*TableMetadata, error) {
	return p.s.patchTable(ctx, p.projectID, p.datasetID, p.tableID, &p.conf)
}
