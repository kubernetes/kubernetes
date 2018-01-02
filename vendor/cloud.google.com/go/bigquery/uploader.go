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
	"reflect"

	"golang.org/x/net/context"
)

// An UploadOption is an optional argument to NewUploader.
type UploadOption interface {
	customizeInsertRows(conf *insertRowsConf)
}

// An Uploader does streaming inserts into a BigQuery table.
// It is safe for concurrent use.
type Uploader struct {
	conf insertRowsConf
	t    *Table
}

// SkipInvalidRows returns an UploadOption that causes rows containing invalid data to be silently ignored.
// The default value is false, which causes the entire request to fail, if there is an attempt to insert an invalid row.
func SkipInvalidRows() UploadOption { return skipInvalidRows{} }

type skipInvalidRows struct{}

func (opt skipInvalidRows) customizeInsertRows(conf *insertRowsConf) {
	conf.skipInvalidRows = true
}

// UploadIgnoreUnknownValues returns an UploadOption that causes values not matching the schema to be ignored.
// If this option is not used, records containing such values are treated as invalid records.
func UploadIgnoreUnknownValues() UploadOption { return uploadIgnoreUnknownValues{} }

type uploadIgnoreUnknownValues struct{}

func (opt uploadIgnoreUnknownValues) customizeInsertRows(conf *insertRowsConf) {
	conf.ignoreUnknownValues = true
}

// A TableTemplateSuffix allows Uploaders to create tables automatically.
//
// Experimental: this option is experimental and may be modified or removed in future versions,
// regardless of any other documented package stability guarantees.
//
// When you specify a suffix, the table you upload data to
// will be used as a template for creating a new table, with the same schema,
// called <table> + <suffix>.
//
// More information is available at
// https://cloud.google.com/bigquery/streaming-data-into-bigquery#template-tables
func TableTemplateSuffix(suffix string) UploadOption { return tableTemplateSuffix(suffix) }

type tableTemplateSuffix string

func (opt tableTemplateSuffix) customizeInsertRows(conf *insertRowsConf) {
	conf.templateSuffix = string(opt)
}

// Put uploads one or more rows to the BigQuery service.  src must implement ValueSaver or be a slice of ValueSavers.
// Put returns a PutMultiError if one or more rows failed to be uploaded.
// The PutMultiError contains a RowInsertionError for each failed row.
func (u *Uploader) Put(ctx context.Context, src interface{}) error {
	// TODO(mcgreevy): Support structs which do not implement ValueSaver as src, a la Datastore.

	if saver, ok := src.(ValueSaver); ok {
		return u.putMulti(ctx, []ValueSaver{saver})
	}

	srcVal := reflect.ValueOf(src)
	if srcVal.Kind() != reflect.Slice {
		return fmt.Errorf("%T is not a ValueSaver or slice of ValueSavers", src)
	}

	var savers []ValueSaver
	for i := 0; i < srcVal.Len(); i++ {
		s := srcVal.Index(i).Interface()
		saver, ok := s.(ValueSaver)
		if !ok {
			return fmt.Errorf("element %d of src is of type %T, which is not a ValueSaver", i, s)
		}
		savers = append(savers, saver)
	}
	return u.putMulti(ctx, savers)
}

func (u *Uploader) putMulti(ctx context.Context, src []ValueSaver) error {
	var rows []*insertionRow
	for _, saver := range src {
		row, insertID, err := saver.Save()
		if err != nil {
			return err
		}
		rows = append(rows, &insertionRow{InsertID: insertID, Row: row})
	}
	return u.t.service.insertRows(ctx, u.t.ProjectID, u.t.DatasetID, u.t.TableID, rows, &u.conf)
}

// An insertionRow represents a row of data to be inserted into a table.
type insertionRow struct {
	// If InsertID is non-empty, BigQuery will use it to de-duplicate insertions of
	// this row on a best-effort basis.
	InsertID string
	// The data to be inserted, represented as a map from field name to Value.
	Row map[string]Value
}
