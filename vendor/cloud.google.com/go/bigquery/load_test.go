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
	"reflect"
	"testing"

	"golang.org/x/net/context"

	bq "google.golang.org/api/bigquery/v2"
)

func defaultLoadJob() *bq.Job {
	return &bq.Job{
		Configuration: &bq.JobConfiguration{
			Load: &bq.JobConfigurationLoad{
				DestinationTable: &bq.TableReference{
					ProjectId: "project-id",
					DatasetId: "dataset-id",
					TableId:   "table-id",
				},
				SourceUris: []string{"uri"},
			},
		},
	}
}

func stringFieldSchema() *FieldSchema {
	return &FieldSchema{Name: "fieldname", Type: StringFieldType}
}

func nestedFieldSchema() *FieldSchema {
	return &FieldSchema{
		Name:   "nested",
		Type:   RecordFieldType,
		Schema: Schema{stringFieldSchema()},
	}
}

func bqStringFieldSchema() *bq.TableFieldSchema {
	return &bq.TableFieldSchema{
		Name: "fieldname",
		Type: "STRING",
	}
}

func bqNestedFieldSchema() *bq.TableFieldSchema {
	return &bq.TableFieldSchema{
		Name:   "nested",
		Type:   "RECORD",
		Fields: []*bq.TableFieldSchema{bqStringFieldSchema()},
	}
}

func TestLoad(t *testing.T) {
	testCases := []struct {
		dst     *Table
		src     *GCSReference
		options []Option
		want    *bq.Job
	}{
		{
			dst:  defaultTable(nil),
			src:  defaultGCS,
			want: defaultLoadJob(),
		},
		{
			dst: defaultTable(nil),
			src: defaultGCS,
			options: []Option{
				MaxBadRecords(1),
				AllowJaggedRows(),
				AllowQuotedNewlines(),
				IgnoreUnknownValues(),
			},
			want: func() *bq.Job {
				j := defaultLoadJob()
				j.Configuration.Load.MaxBadRecords = 1
				j.Configuration.Load.AllowJaggedRows = true
				j.Configuration.Load.AllowQuotedNewlines = true
				j.Configuration.Load.IgnoreUnknownValues = true
				return j
			}(),
		},
		{
			dst: &Table{
				ProjectID: "project-id",
				DatasetID: "dataset-id",
				TableID:   "table-id",
			},
			options: []Option{CreateNever, WriteTruncate},
			src:     defaultGCS,
			want: func() *bq.Job {
				j := defaultLoadJob()
				j.Configuration.Load.CreateDisposition = "CREATE_NEVER"
				j.Configuration.Load.WriteDisposition = "WRITE_TRUNCATE"
				return j
			}(),
		},
		{
			dst: &Table{
				ProjectID: "project-id",
				DatasetID: "dataset-id",
				TableID:   "table-id",
			},
			src: defaultGCS,
			options: []Option{
				DestinationSchema(Schema{
					stringFieldSchema(),
					nestedFieldSchema(),
				}),
			},
			want: func() *bq.Job {
				j := defaultLoadJob()
				j.Configuration.Load.Schema = &bq.TableSchema{
					Fields: []*bq.TableFieldSchema{
						bqStringFieldSchema(),
						bqNestedFieldSchema(),
					}}
				return j
			}(),
		},
		{
			dst: defaultTable(nil),
			src: &GCSReference{
				uris:            []string{"uri"},
				SkipLeadingRows: 1,
				SourceFormat:    JSON,
				Encoding:        UTF_8,
				FieldDelimiter:  "\t",
				Quote:           "-",
			},
			want: func() *bq.Job {
				j := defaultLoadJob()
				j.Configuration.Load.SkipLeadingRows = 1
				j.Configuration.Load.SourceFormat = "NEWLINE_DELIMITED_JSON"
				j.Configuration.Load.Encoding = "UTF-8"
				j.Configuration.Load.FieldDelimiter = "\t"
				hyphen := "-"
				j.Configuration.Load.Quote = &hyphen
				return j
			}(),
		},
		{
			dst: defaultTable(nil),
			src: &GCSReference{
				uris:  []string{"uri"},
				Quote: "",
			},
			want: func() *bq.Job {
				j := defaultLoadJob()
				j.Configuration.Load.Quote = nil
				return j
			}(),
		},
		{
			dst: defaultTable(nil),
			src: &GCSReference{
				uris:           []string{"uri"},
				Quote:          "",
				ForceZeroQuote: true,
			},
			want: func() *bq.Job {
				j := defaultLoadJob()
				empty := ""
				j.Configuration.Load.Quote = &empty
				return j
			}(),
		},
	}

	for _, tc := range testCases {
		s := &testService{}
		c := &Client{
			service: s,
		}
		if _, err := c.Copy(context.Background(), tc.dst, tc.src, tc.options...); err != nil {
			t.Errorf("err calling load: %v", err)
			continue
		}
		if !reflect.DeepEqual(s.Job, tc.want) {
			t.Errorf("loading: got:\n%v\nwant:\n%v", s.Job, tc.want)
		}
	}
}
