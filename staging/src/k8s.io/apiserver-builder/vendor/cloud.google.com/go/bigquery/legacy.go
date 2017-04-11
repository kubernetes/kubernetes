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

	"golang.org/x/net/context"
)

// OpenTable creates a handle to an existing BigQuery table. If the table does
// not already exist, subsequent uses of the *Table will fail.
//
// Deprecated: use Client.DatasetInProject.Table instead.
func (c *Client) OpenTable(projectID, datasetID, tableID string) *Table {
	return c.Table(projectID, datasetID, tableID)
}

// Table creates a handle to a BigQuery table.
//
// Use this method to reference a table in a project other than that of the
// Client.
//
// Deprecated: use Client.DatasetInProject.Table instead.
func (c *Client) Table(projectID, datasetID, tableID string) *Table {
	return &Table{ProjectID: projectID, DatasetID: datasetID, TableID: tableID, service: c.service}
}

// CreateTable creates a table in the BigQuery service and returns a handle to it.
//
// Deprecated: use Table.Create instead.
func (c *Client) CreateTable(ctx context.Context, projectID, datasetID, tableID string, options ...CreateTableOption) (*Table, error) {
	t := c.Table(projectID, datasetID, tableID)
	if err := t.Create(ctx, options...); err != nil {
		return nil, err
	}
	return t, nil
}

// Read fetches data from a ReadSource and returns the data via an Iterator.
//
// Deprecated: use Query.Read, Job.Read or Table.Read instead.
func (c *Client) Read(ctx context.Context, src ReadSource, options ...ReadOption) (*Iterator, error) {
	switch src := src.(type) {
	case *Job:
		return src.Read(ctx, options...)
	case *Query:
		// For compatibility, support Query values created by literal, rather
		// than Client.Query.
		if src.client == nil {
			src.client = c
		}
		return src.Read(ctx, options...)
	case *Table:
		return src.Read(ctx, options...)
	}
	return nil, fmt.Errorf("src (%T) does not support the Read operation", src)
}
