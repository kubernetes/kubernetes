// Copyright 2014 Google Inc. All Rights Reserved.
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

package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/SeanDolphin/bqschema"
	"github.com/google/cadvisor/storage/bigquery/client"
)

type container struct {
	Name         string    `json:"name"`
	CpuUsage     uint64    `json:"cpuusage,omitempty"`
	MemoryUsage  uint64    `json:"memoryusage,omitempty"`
	NetworkUsage uint64    `json:"networkusage,omitempty"`
	Timestamp    time.Time `json:"timestamp"`
}

func main() {
	flag.Parse()
	c, err := client.NewClient()
	if err != nil {
		fmt.Printf("Failed to connect to bigquery\n")
		panic(err)
	}

	c.PrintDatasets()

	// Create a new dataset.
	err = c.CreateDataset("sampledataset")
	if err != nil {
		fmt.Printf("Failed to create dataset %v\n", err)
		panic(err)
	}

	// Create a new table
	containerData := container{
		Name:         "test_container",
		CpuUsage:     123456,
		MemoryUsage:  1024,
		NetworkUsage: 9046,
		Timestamp:    time.Now(),
	}
	schema, err := bqschema.ToSchema(containerData)
	if err != nil {
		fmt.Printf("Failed to create schema")
		panic(err)
	}

	err = c.CreateTable("sampletable", schema)
	if err != nil {
		fmt.Printf("Failed to create table")
		panic(err)
	}

	// Add Data
	m := make(map[string]interface{})
	t := time.Now()
	for i := 0; i < 10; i++ {
		m["Name"] = containerData.Name
		m["CpuUsage"] = containerData.CpuUsage + uint64(i*100)
		m["MemoryUsage"] = containerData.MemoryUsage - uint64(i*10)
		m["NetworkUsage"] = containerData.NetworkUsage + uint64(i*10)
		m["Timestamp"] = t.Add(time.Duration(i) * time.Second)

		err = c.InsertRow(m)
		if err != nil {
			fmt.Printf("Failed to insert row")
			panic(err)
		}
	}
}
