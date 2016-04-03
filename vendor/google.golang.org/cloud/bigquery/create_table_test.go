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
	"time"

	"golang.org/x/net/context"
)

type createTableRecorder struct {
	conf *createTableConf
	service
}

func (rec *createTableRecorder) createTable(ctx context.Context, conf *createTableConf) error {
	rec.conf = conf
	return nil
}

func TestCreateTableOptions(t *testing.T) {
	s := &createTableRecorder{}
	c := &Client{
		service: s,
	}
	exp := time.Now()
	q := "query"
	if _, err := c.CreateTable(context.Background(), "p", "d", "t", TableExpiration(exp), ViewQuery(q)); err != nil {
		t.Fatalf("err calling CreateTable: %v", err)
	}
	want := createTableConf{
		projectID:  "p",
		datasetID:  "d",
		tableID:    "t",
		expiration: exp,
		viewQuery:  q,
	}
	if !reflect.DeepEqual(*s.conf, want) {
		t.Errorf("createTableConf: got:\n%v\nwant:\n%v", *s.conf, want)
	}
}
