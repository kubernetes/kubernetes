// +build go1.7

package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"time"

	"github.com/satori/go.uuid"
	chk "gopkg.in/check.v1"
)

type TableBatchSuite struct{}

var _ = chk.Suite(&TableBatchSuite{})

func (s *TableBatchSuite) Test_BatchInsertMultipleEntities(c *chk.C) {
	cli := getBasicClient(c).GetTableService()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c, "me"))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	props := map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	entity.Properties = props

	entity2 := table.GetEntityReference("mypartitionkey", "myrowkey2")
	props2 := map[string]interface{}{
		"AmountDue":      111.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	entity2.Properties = props2

	batch := table.NewBatch()
	batch.InsertOrReplaceEntity(entity, false)
	batch.InsertOrReplaceEntity(entity2, false)

	err = batch.ExecuteBatch()
	c.Assert(err, chk.IsNil)

	options := QueryOptions{
		Top: 2,
	}

	results, err := table.QueryEntities(30, FullMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 2)
}

func (s *TableBatchSuite) Test_BatchInsertSameEntryMultipleTimes(c *chk.C) {
	cli := getBasicClient(c).GetTableService()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	props := map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	entity.Properties = props

	batch := table.NewBatch()
	batch.InsertOrReplaceEntity(entity, false)
	batch.InsertOrReplaceEntity(entity, false)

	err = batch.ExecuteBatch()
	c.Assert(err, chk.NotNil)
	v, ok := err.(AzureStorageServiceError)
	if ok {
		c.Assert(v.Code, chk.Equals, "InvalidDuplicateRow")
	}
}

func (s *TableBatchSuite) Test_BatchInsertDeleteSameEntity(c *chk.C) {
	cli := getBasicClient(c).GetTableService()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	props := map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	entity.Properties = props

	batch := table.NewBatch()
	batch.InsertOrReplaceEntity(entity, false)
	batch.DeleteEntity(entity, true)

	err = batch.ExecuteBatch()
	c.Assert(err, chk.NotNil)

	v, ok := err.(AzureStorageServiceError)
	if ok {
		c.Assert(v.Code, chk.Equals, "InvalidDuplicateRow")
	}
}

func (s *TableBatchSuite) Test_BatchInsertThenDeleteDifferentBatches(c *chk.C) {
	cli := getBasicClient(c).GetTableService()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	props := map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	entity.Properties = props

	batch := table.NewBatch()
	batch.InsertOrReplaceEntity(entity, false)
	err = batch.ExecuteBatch()
	c.Assert(err, chk.IsNil)

	options := QueryOptions{
		Top: 2,
	}

	results, err := table.QueryEntities(30, FullMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 1)

	batch = table.NewBatch()
	batch.DeleteEntity(entity, true)
	err = batch.ExecuteBatch()
	c.Assert(err, chk.IsNil)

	// Timeout set to 15 for this test to work propwrly with the recordings
	results, err = table.QueryEntities(15, FullMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 0)
}

func (s *TableBatchSuite) Test_BatchInsertThenMergeDifferentBatches(c *chk.C) {
	cli := getBasicClient(c).GetTableService()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	props := map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	entity.Properties = props

	batch := table.NewBatch()
	batch.InsertOrReplaceEntity(entity, false)
	err = batch.ExecuteBatch()
	c.Assert(err, chk.IsNil)

	entity2 := table.GetEntityReference("mypartitionkey", "myrowkey")
	props2 := map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"DifferentField": 123,
		"NumberOfOrders": int64(255),
	}
	entity2.Properties = props2

	batch = table.NewBatch()
	batch.InsertOrReplaceEntity(entity2, false)
	err = batch.ExecuteBatch()
	c.Assert(err, chk.IsNil)

	options := QueryOptions{
		Top: 2,
	}

	results, err := table.QueryEntities(30, FullMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 1)
}
