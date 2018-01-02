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
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/satori/uuid"
	chk "gopkg.in/check.v1"
)

type StorageEntitySuite struct{}

var _ = chk.Suite(&StorageEntitySuite{})

func (s *StorageEntitySuite) TestGet(c *chk.C) {
	cli := getTableClient(c)
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
	err = entity.Insert(EmptyPayload, nil)
	c.Assert(err, chk.IsNil)

	err = entity.Get(30, FullMetadata, &GetEntityOptions{
		Select: []string{"IsActive"},
	})
	c.Assert(err, chk.IsNil)
	c.Assert(entity.Properties, chk.HasLen, 1)

	err = entity.Get(30, FullMetadata, &GetEntityOptions{
		Select: []string{
			"AmountDue",
			"CustomerCode",
			"CustomerSince",
			"IsActive",
			"NumberOfOrders",
		}})
	c.Assert(err, chk.IsNil)
	c.Assert(entity.Properties, chk.HasLen, 5)

	err = entity.Get(30, FullMetadata, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(entity.Properties, chk.HasLen, 5)
}

const (
	validEtag = "W/\"datetime''2017-04-01T01%3A07%3A23.8881885Z''\""
)

func (s *StorageEntitySuite) TestInsert(c *chk.C) {
	cli := getTableClient(c)
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
	err = entity.Insert(EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	// Did not update
	c.Assert(entity.TimeStamp, chk.Equals, time.Time{})
	c.Assert(entity.OdataMetadata, chk.Equals, "")
	c.Assert(entity.OdataType, chk.Equals, "")
	c.Assert(entity.OdataID, chk.Equals, "")
	c.Assert(entity.OdataEtag, chk.Equals, "")
	c.Assert(entity.OdataEditLink, chk.Equals, "")

	// Update
	entity.PartitionKey = "mypartitionkey2"
	entity.RowKey = "myrowkey2"
	err = entity.Insert(FullMetadata, nil)
	c.Assert(err, chk.IsNil)
	// Check everything was updated...
	c.Assert(entity.TimeStamp, chk.NotNil)
	c.Assert(entity.OdataMetadata, chk.Not(chk.Equals), "")
	c.Assert(entity.OdataType, chk.Not(chk.Equals), "")
	c.Assert(entity.OdataID, chk.Not(chk.Equals), "")
	c.Assert(entity.OdataEtag, chk.Not(chk.Equals), "")
	c.Assert(entity.OdataEditLink, chk.Not(chk.Equals), "")
}

func (s *StorageEntitySuite) TestUpdate(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	entity.Properties = map[string]interface{}{
		"AmountDue":      200.23,
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}
	// Force update
	err = entity.Insert(FullMetadata, nil)
	c.Assert(err, chk.IsNil)

	etag := entity.OdataEtag
	timestamp := entity.TimeStamp

	props := map[string]interface{}{
		"Name":         "Anakin",
		"FamilyName":   "Skywalker",
		"HasEpicTheme": true,
	}
	entity.Properties = props
	// Update providing etag
	err = entity.Update(false, nil)
	c.Assert(err, chk.IsNil)

	c.Assert(entity.Properties, chk.DeepEquals, props)
	c.Assert(entity.OdataEtag, chk.Not(chk.Equals), etag)
	c.Assert(entity.TimeStamp, chk.Not(chk.Equals), timestamp)

	// Try to update with old etag
	entity.OdataEtag = validEtag
	err = entity.Update(false, nil)
	c.Assert(err, chk.NotNil)
	c.Assert(strings.Contains(err.Error(), "Etag didn't match"), chk.Equals, true)

	// Force update
	props = map[string]interface{}{
		"Name":            "Leia",
		"FamilyName":      "Organa",
		"HasAwesomeDress": true,
	}
	entity.Properties = props
	err = entity.Update(true, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(entity.Properties, chk.DeepEquals, props)
}

func (s *StorageEntitySuite) TestMerge(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	entity.Properties = map[string]interface{}{
		"Country":  "Mexico",
		"MalePoet": "Nezahualcoyotl",
	}
	c.Assert(entity.Insert(FullMetadata, nil), chk.IsNil)

	etag := entity.OdataEtag
	timestamp := entity.TimeStamp

	entity.Properties = map[string]interface{}{
		"FemalePoet": "Sor Juana Ines de la Cruz",
	}
	// Merge providing etag
	err = entity.Merge(false, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(entity.OdataEtag, chk.Not(chk.Equals), etag)
	c.Assert(entity.TimeStamp, chk.Not(chk.Equals), timestamp)

	// Try to merge with incorrect etag
	entity.OdataEtag = validEtag
	err = entity.Merge(false, nil)
	c.Assert(err, chk.NotNil)
	c.Assert(strings.Contains(err.Error(), "Etag didn't match"), chk.Equals, true)

	// Force merge
	entity.Properties = map[string]interface{}{
		"MalePainter":   "Diego Rivera",
		"FemalePainter": "Frida Kahlo",
	}
	err = entity.Merge(true, nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageEntitySuite) TestDelete(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	// Delete providing etag
	entity1 := table.GetEntityReference("pkey1", "rowkey1")
	c.Assert(entity1.Insert(FullMetadata, nil), chk.IsNil)

	err = entity1.Delete(false, nil)
	c.Assert(err, chk.IsNil)

	// Try to delete with incorrect etag
	entity2 := table.GetEntityReference("pkey2", "rowkey2")
	c.Assert(entity2.Insert(EmptyPayload, nil), chk.IsNil)
	entity2.OdataEtag = "GolangRocksOnAzure"

	err = entity2.Delete(false, nil)
	c.Assert(err, chk.NotNil)

	// Force delete
	err = entity2.Delete(true, nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageEntitySuite) TestInsertOrReplace(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	entity.Properties = map[string]interface{}{
		"Name":         "Anakin",
		"FamilyName":   "Skywalker",
		"HasEpicTheme": true,
	}

	err = entity.InsertOrReplace(nil)
	c.Assert(err, chk.IsNil)

	entity.Properties = map[string]interface{}{
		"Name":            "Leia",
		"FamilyName":      "Organa",
		"HasAwesomeDress": true,
	}
	err = entity.InsertOrReplace(nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageEntitySuite) TestInsertOrMerge(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "myrowkey")
	entity.Properties = map[string]interface{}{
		"Name":       "Luke",
		"FamilyName": "Skywalker",
	}

	err = entity.InsertOrMerge(nil)
	c.Assert(err, chk.IsNil)

	entity.Properties = map[string]interface{}{
		"Father": "Anakin",
		"Mentor": "Yoda",
	}
	err = entity.InsertOrMerge(nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageEntitySuite) Test_InsertAndGetEntities(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "100")
	entity.Properties = map[string]interface{}{
		"Name":          "Luke",
		"FamilyName":    "Skywalker",
		"HasCoolWeapon": true,
	}
	c.Assert(entity.Insert(EmptyPayload, nil), chk.IsNil)

	entity.RowKey = "200"
	c.Assert(entity.Insert(FullMetadata, nil), chk.IsNil)

	entities, err := table.QueryEntities(30, FullMetadata, nil)
	c.Assert(err, chk.IsNil)

	c.Assert(entities.Entities, chk.HasLen, 2)
	c.Assert(entities.OdataMetadata+"/@Element", chk.Equals, entity.OdataMetadata)

	compareEntities(entities.Entities[1], entity, c)
}

func (s *StorageEntitySuite) Test_InsertAndExecuteQuery(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "100")
	entity.Properties = map[string]interface{}{
		"Name":          "Luke",
		"FamilyName":    "Skywalker",
		"HasCoolWeapon": true,
	}
	c.Assert(entity.Insert(EmptyPayload, nil), chk.IsNil)

	entity.RowKey = "200"
	c.Assert(entity.Insert(EmptyPayload, nil), chk.IsNil)

	queryOptions := QueryOptions{
		Filter: "RowKey eq '200'",
	}

	entities, err := table.QueryEntities(30, FullMetadata, &queryOptions)
	c.Assert(err, chk.IsNil)

	c.Assert(entities.Entities, chk.HasLen, 1)
	c.Assert(entities.Entities[0].RowKey, chk.Equals, entity.RowKey)
}

func (s *StorageEntitySuite) Test_InsertAndDeleteEntities(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	entity := table.GetEntityReference("mypartitionkey", "100")
	entity.Properties = map[string]interface{}{
		"FamilyName": "Skywalker",
		"Name":       "Luke",
		"Number":     3,
	}
	c.Assert(entity.Insert(EmptyPayload, nil), chk.IsNil)

	entity.Properties["Number"] = 1
	entity.RowKey = "200"
	c.Assert(entity.Insert(FullMetadata, nil), chk.IsNil)

	options := QueryOptions{
		Filter: "Number eq 1",
	}

	result, err := table.QueryEntities(30, FullMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(result.Entities, chk.HasLen, 1)
	compareEntities(result.Entities[0], entity, c)

	err = result.Entities[0].Delete(true, nil)
	c.Assert(err, chk.IsNil)

	result, err = table.QueryEntities(30, FullMetadata, nil)
	c.Assert(err, chk.IsNil)

	// only 1 entry must be present
	c.Assert(result.Entities, chk.HasLen, 1)
}

func (s *StorageEntitySuite) TestExecuteQueryNextResults(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	var entityList []*Entity

	for i := 0; i < 5; i++ {
		entity := table.GetEntityReference("pkey", fmt.Sprintf("r%d", i))
		err := entity.Insert(FullMetadata, nil)
		c.Assert(err, chk.IsNil)
		entityList = append(entityList, entity)
	}

	// retrieve using top = 2. Should return 2 entries, 2 entries and finally
	// 1 entry
	options := QueryOptions{
		Top: 2,
	}
	results, err := table.QueryEntities(30, FullMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 2)
	c.Assert(results.NextLink, chk.NotNil)
	compareEntities(results.Entities[0], entityList[0], c)
	compareEntities(results.Entities[1], entityList[1], c)

	results, err = results.NextResults(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 2)
	c.Assert(results.NextLink, chk.NotNil)
	compareEntities(results.Entities[0], entityList[2], c)
	compareEntities(results.Entities[1], entityList[3], c)

	results, err = results.NextResults(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(results.Entities, chk.HasLen, 1)
	c.Assert(results.NextLink, chk.IsNil)
	compareEntities(results.Entities[0], entityList[4], c)
}

func (s *StorageEntitySuite) Test_entityMarshalJSON(c *chk.C) {
	expected := `{"Address":"Mountain View","Age":23,"AmountDue":200.23,"Binary":"abcd","Binary@odata.type":"Edm.Binary","CustomerCode":"c9da6455-213d-42c9-9a79-3e9149a57833","CustomerCode@odata.type":"Edm.Guid","CustomerSince":"1992-12-20T21:55:00Z","CustomerSince@odata.type":"Edm.DateTime","IsActive":true,"NumberOfOrders":"255","NumberOfOrders@odata.type":"Edm.Int64","PartitionKey":"mypartitionkey","RowKey":"myrowkey"}`

	entity := Entity{
		PartitionKey: "mypartitionkey",
		RowKey:       "myrowkey",
		Properties: map[string]interface{}{
			"Address":        "Mountain View",
			"Age":            23,
			"AmountDue":      200.23,
			"Binary":         []byte("abcd"),
			"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
			"CustomerSince":  time.Date(1992, time.December, 20, 21, 55, 0, 0, time.UTC),
			"IsActive":       true,
			"NumberOfOrders": int64(255),
		},
	}
	got, err := json.Marshal(&entity)
	c.Assert(err, chk.IsNil)
	c.Assert(string(got), chk.Equals, expected)

	entity.Properties["Contoso@odata.type"] = "Edm.Trololololol"
	got, err = json.Marshal(&entity)
	c.Assert(got, chk.IsNil)
	c.Assert(err, chk.ErrorMatches, ".*Odata.type annotation Contoso@odata.type value is not valid")

	entity.Properties["Contoso@odata.type"] = OdataGUID
	got, err = json.Marshal(&entity)
	c.Assert(got, chk.IsNil)
	c.Assert(err, chk.ErrorMatches, ".*Odata.type annotation Contoso@odata.type defined without value defined")
}

func (s *StorageEntitySuite) Test_entityUnmarshalJSON(c *chk.C) {
	input := `{
        "odata.metadata":"https://azuregosdkstoragetests.table.core.windows.net/$metadata#SampleTable/@Element",
        "odata.type":"azuregosdkstoragetests.SampleTable",
        "odata.id":"https://azuregosdkstoragetests.table.core.windows.net/SampleTable(PartitionKey=''mypartitionkey'',RowKey=''myrowkey'')",
        "odata.etag":"W/\"datetime''2017-01-27T01%3A01%3A44.151805Z''\"",
        "odata.editLink":"SampleTable(PartitionKey=''mypartitionkey'',RowKey=''myrowkey'')",
        "PartitionKey":"mypartitionkey",
        "RowKey":"myrowkey",
	    "Timestamp":"2017-01-27T01:01:44.151805Z",
        "Timestamp@odata.type":"Edm.DateTime",
        "Address": "Mountain View",
        "Age": 23,
        "AmountDue":200.23,
        "Binary@odata.type": "Edm.Binary",
        "Binary": "abcd",
        "CustomerCode@odata.type":"Edm.Guid",
        "CustomerCode":"c9da6455-213d-42c9-9a79-3e9149a57833",
        "CustomerSince@odata.type":"Edm.DateTime",
        "CustomerSince":"1992-12-20T21:55:00Z",
        "IsActive":true,
        "NumberOfOrders@odata.type":"Edm.Int64",
        "NumberOfOrders":"255"}`

	var entity Entity
	data := []byte(input)
	err := json.Unmarshal(data, &entity)
	c.Assert(err, chk.IsNil)

	expectedProperties := map[string]interface{}{
		"Address":        "Mountain View",
		"Age":            23,
		"AmountDue":      200.23,
		"Binary":         []byte("abcd"),
		"CustomerCode":   uuid.FromStringOrNil("c9da6455-213d-42c9-9a79-3e9149a57833"),
		"CustomerSince":  time.Date(1992, 12, 20, 21, 55, 0, 0, time.UTC),
		"IsActive":       true,
		"NumberOfOrders": int64(255),
	}

	c.Assert(entity.OdataMetadata, chk.Equals, "https://azuregosdkstoragetests.table.core.windows.net/$metadata#SampleTable/@Element")
	c.Assert(entity.OdataType, chk.Equals, "azuregosdkstoragetests.SampleTable")
	c.Assert(entity.OdataID, chk.Equals, "https://azuregosdkstoragetests.table.core.windows.net/SampleTable(PartitionKey=''mypartitionkey'',RowKey=''myrowkey'')")
	c.Assert(entity.OdataEtag, chk.Equals, "W/\"datetime''2017-01-27T01%3A01%3A44.151805Z''\"")
	c.Assert(entity.OdataEditLink, chk.Equals, "SampleTable(PartitionKey=''mypartitionkey'',RowKey=''myrowkey'')")
	c.Assert(entity.PartitionKey, chk.Equals, "mypartitionkey")
	c.Assert(entity.RowKey, chk.Equals, "myrowkey")
	c.Assert(entity.TimeStamp, chk.Equals, time.Date(2017, 1, 27, 1, 1, 44, 151805000, time.UTC))

	c.Assert(entity.Properties, chk.HasLen, len(expectedProperties))
	c.Assert(entity.Properties["Address"], chk.Equals, expectedProperties["Address"])
	// Note on Age assertion... Looks like the json unmarshaller thinks all numbers are float64.
	c.Assert(entity.Properties["Age"], chk.Equals, float64(expectedProperties["Age"].(int)))
	c.Assert(entity.Properties["AmountDue"], chk.Equals, expectedProperties["AmountDue"])
	c.Assert(entity.Properties["Binary"], chk.DeepEquals, expectedProperties["Binary"])
	c.Assert(entity.Properties["CustomerSince"], chk.Equals, expectedProperties["CustomerSince"])
	c.Assert(entity.Properties["IsActive"], chk.Equals, expectedProperties["IsActive"])
	c.Assert(entity.Properties["NumberOfOrders"], chk.Equals, expectedProperties["NumberOfOrders"])

}

func compareEntities(got, expected *Entity, c *chk.C) {
	c.Assert(got.PartitionKey, chk.Equals, expected.PartitionKey)
	c.Assert(got.RowKey, chk.Equals, expected.RowKey)
	c.Assert(got.TimeStamp, chk.Equals, expected.TimeStamp)

	c.Assert(got.OdataEtag, chk.Equals, expected.OdataEtag)
	c.Assert(got.OdataType, chk.Equals, expected.OdataType)
	c.Assert(got.OdataID, chk.Equals, expected.OdataID)
	c.Assert(got.OdataEditLink, chk.Equals, expected.OdataEditLink)

	c.Assert(got.Properties, chk.DeepEquals, expected.Properties)
}
