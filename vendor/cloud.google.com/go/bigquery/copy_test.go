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

func defaultCopyJob() *bq.Job {
	return &bq.Job{
		Configuration: &bq.JobConfiguration{
			Copy: &bq.JobConfigurationTableCopy{
				DestinationTable: &bq.TableReference{
					ProjectId: "d-project-id",
					DatasetId: "d-dataset-id",
					TableId:   "d-table-id",
				},
				SourceTables: []*bq.TableReference{
					{
						ProjectId: "s-project-id",
						DatasetId: "s-dataset-id",
						TableId:   "s-table-id",
					},
				},
			},
		},
	}
}

func TestCopy(t *testing.T) {
	testCases := []struct {
		dst     *Table
		src     Tables
		options []Option
		want    *bq.Job
	}{
		{
			dst: &Table{
				ProjectID: "d-project-id",
				DatasetID: "d-dataset-id",
				TableID:   "d-table-id",
			},
			src: Tables{
				{
					ProjectID: "s-project-id",
					DatasetID: "s-dataset-id",
					TableID:   "s-table-id",
				},
			},
			want: defaultCopyJob(),
		},
		{
			dst: &Table{
				ProjectID: "d-project-id",
				DatasetID: "d-dataset-id",
				TableID:   "d-table-id",
			},
			src: Tables{
				{
					ProjectID: "s-project-id",
					DatasetID: "s-dataset-id",
					TableID:   "s-table-id",
				},
			},
			options: []Option{CreateNever, WriteTruncate},
			want: func() *bq.Job {
				j := defaultCopyJob()
				j.Configuration.Copy.CreateDisposition = "CREATE_NEVER"
				j.Configuration.Copy.WriteDisposition = "WRITE_TRUNCATE"
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
			t.Errorf("err calling cp: %v", err)
			continue
		}
		if !reflect.DeepEqual(s.Job, tc.want) {
			t.Errorf("copying: got:\n%v\nwant:\n%v", s.Job, tc.want)
		}
	}
}
