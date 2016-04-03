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

import "golang.org/x/net/context"

// Dataset is a reference to a BigQuery dataset.
type Dataset struct {
	id     string
	client *Client
}

// ListTables returns a list of all the tables contained in the Dataset.
func (d *Dataset) ListTables(ctx context.Context) ([]*Table, error) {
	var tables []*Table

	err := getPages("", func(pageToken string) (string, error) {
		ts, tok, err := d.client.service.listTables(ctx, d.client.projectID, d.id, pageToken)
		if err == nil {
			tables = append(tables, ts...)
		}
		return tok, err
	})

	if err != nil {
		return nil, err
	}
	return tables, nil
}
