// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package influxdb

import (
	"strings"
	"time"

	influxdb "github.com/influxdata/influxdb/client"
)

type PointSavedToInfluxdb struct {
	Ponit influxdb.Point
}

type FakeInfluxDBClient struct {
	Pnts []PointSavedToInfluxdb
}

func NewFakeInfluxDBClient() *FakeInfluxDBClient {
	return &FakeInfluxDBClient{[]PointSavedToInfluxdb{}}
}

func (client *FakeInfluxDBClient) Write(bps influxdb.BatchPoints) (*influxdb.Response, error) {
	for _, pnt := range bps.Points {
		client.Pnts = append(client.Pnts, PointSavedToInfluxdb{pnt})
	}
	return nil, nil
}

func (client *FakeInfluxDBClient) Query(q influxdb.Query) (*influxdb.Response, error) {
	numQueries := strings.Count(q.Command, ";")

	// return an empty result for each separate query
	return &influxdb.Response{
		Results: make([]influxdb.Result, numQueries),
	}, nil
}

func (client *FakeInfluxDBClient) Ping() (time.Duration, string, error) {
	return 0, "", nil
}

var Client = NewFakeInfluxDBClient()

var Config = InfluxdbConfig{
	User:     "root",
	Password: "root",
	Host:     "localhost:8086",
	DbName:   "k8s",
	Secure:   false,
}
