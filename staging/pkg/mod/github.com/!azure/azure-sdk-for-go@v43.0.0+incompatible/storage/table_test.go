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
	"strconv"
	"time"

	chk "gopkg.in/check.v1"
)

type StorageTableSuite struct{}

var _ = chk.Suite(&StorageTableSuite{})

func getTableClient(c *chk.C) TableServiceClient {
	return getBasicClient(c).GetTableService()
}

func (cli *TableServiceClient) deleteAllTables() {
	if result, _ := cli.QueryTables(MinimalMetadata, nil); result != nil {
		for _, t := range result.Tables {
			t.Delete(30, nil)
		}
	}
}

func (s *StorageTableSuite) Test_CreateAndDeleteTable(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table1 := cli.GetTableReference(tableName(c, "1"))
	err := table1.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)

	// update table metadata
	table2 := cli.GetTableReference(tableName(c, "2"))
	err = table2.Create(30, FullMetadata, nil)
	defer table2.Delete(30, nil)
	c.Assert(err, chk.IsNil)

	// Check not empty values
	c.Assert(table2.OdataEditLink, chk.Not(chk.Equals), "")
	c.Assert(table2.OdataID, chk.Not(chk.Equals), "")
	c.Assert(table2.OdataMetadata, chk.Not(chk.Equals), "")
	c.Assert(table2.OdataType, chk.Not(chk.Equals), "")

	err = table1.Delete(30, nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageTableSuite) Test_CreateTableWithAllResponsePayloadLevels(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	createAndDeleteTable(cli, EmptyPayload, c, "empty")
	createAndDeleteTable(cli, NoMetadata, c, "nm")
	createAndDeleteTable(cli, MinimalMetadata, c, "minimal")
	createAndDeleteTable(cli, FullMetadata, c, "full")
}

func (s *StorageTableSuite) TestGet(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	tn := tableName(c)
	table := cli.GetTableReference(tn)
	err := table.Create(30, EmptyPayload, nil)
	c.Assert(err, chk.IsNil)
	defer table.Delete(30, nil)

	err = table.Get(30, FullMetadata)
	c.Assert(err, chk.IsNil)
	c.Assert(table.Name, chk.Equals, tn)
	c.Assert(table.OdataEditLink, chk.Not(chk.Equals), "")
	c.Assert(table.OdataID, chk.Not(chk.Equals), "")
	c.Assert(table.OdataMetadata, chk.Not(chk.Equals), "")
	c.Assert(table.OdataType, chk.Not(chk.Equals), "")
}

func createAndDeleteTable(cli TableServiceClient, ml MetadataLevel, c *chk.C, extra string) {
	table := cli.GetTableReference(tableName(c, extra))
	c.Assert(table.Create(30, ml, nil), chk.IsNil)
	c.Assert(table.Delete(30, nil), chk.IsNil)
}

func (s *StorageTableSuite) TestQueryTablesNextResults(c *chk.C) {
	cli := getTableClient(c)
	cli.deleteAllTables()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	for i := 0; i < 3; i++ {
		table := cli.GetTableReference(tableName(c, strconv.Itoa(i)))
		err := table.Create(30, EmptyPayload, nil)
		c.Assert(err, chk.IsNil)
		defer table.Delete(30, nil)
	}

	options := QueryTablesOptions{
		Top: 2,
	}
	result, err := cli.QueryTables(MinimalMetadata, &options)
	c.Assert(err, chk.IsNil)
	c.Assert(result.Tables, chk.HasLen, 2)
	c.Assert(result.NextLink, chk.NotNil)

	result, err = result.NextResults(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(result.Tables, chk.HasLen, 1)
	c.Assert(result.NextLink, chk.IsNil)

	result, err = result.NextResults(nil)
	c.Assert(result, chk.IsNil)
	c.Assert(err, chk.NotNil)
}

func appendTablePermission(policies []TableAccessPolicy, ID string,
	canRead bool, canAppend bool, canUpdate bool, canDelete bool,
	startTime time.Time, expiryTime time.Time) []TableAccessPolicy {

	tap := TableAccessPolicy{
		ID:         ID,
		StartTime:  startTime,
		ExpiryTime: expiryTime,
		CanRead:    canRead,
		CanAppend:  canAppend,
		CanUpdate:  canUpdate,
		CanDelete:  canDelete,
	}
	policies = append(policies, tap)
	return policies
}

func (s *StorageTableSuite) TestSetPermissionsSuccessfully(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	c.Assert(table.Create(30, EmptyPayload, nil), chk.IsNil)
	defer table.Delete(30, nil)

	policies := []TableAccessPolicy{}
	policies = appendTablePermission(policies, "GolangRocksOnAzure", true, true, true, true, fixedTime, fixedTime.Add(10*time.Hour))

	err := table.SetPermissions(policies, 30, nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageTableSuite) TestSetPermissionsUnsuccessfully(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference("nonexistingtable")

	policies := []TableAccessPolicy{}
	policies = appendTablePermission(policies, "GolangRocksOnAzure", true, true, true, true, fixedTime, fixedTime.Add(10*time.Hour))

	err := table.SetPermissions(policies, 30, nil)
	c.Assert(err, chk.NotNil)
}

func (s *StorageTableSuite) TestSetThenGetPermissionsSuccessfully(c *chk.C) {
	cli := getTableClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	table := cli.GetTableReference(tableName(c))
	c.Assert(table.Create(30, EmptyPayload, nil), chk.IsNil)
	defer table.Delete(30, nil)

	policies := []TableAccessPolicy{}
	policies = appendTablePermission(policies, "GolangRocksOnAzure", true, true, true, true, fixedTime, fixedTime.Add(10*time.Hour))
	policies = appendTablePermission(policies, "AutoRestIsSuperCool", true, true, false, true, fixedTime.Add(20*time.Hour), fixedTime.Add(30*time.Hour))

	err := table.SetPermissions(policies, 30, nil)
	c.Assert(err, chk.IsNil)

	newPolicies, err := table.GetPermissions(30, nil)
	c.Assert(err, chk.IsNil)

	// fixedTime check policy set.
	c.Assert(newPolicies, chk.HasLen, 2)

	for i := range newPolicies {
		c.Assert(newPolicies[i].ID, chk.Equals, policies[i].ID)

		// test timestamps down the second
		// rounding start/expiry time original perms since the returned perms would have been rounded.
		// so need rounded vs rounded.
		c.Assert(newPolicies[i].StartTime.UTC().Round(time.Second).Format(time.RFC1123),
			chk.Equals, policies[i].StartTime.UTC().Round(time.Second).Format(time.RFC1123))
		c.Assert(newPolicies[i].ExpiryTime.UTC().Round(time.Second).Format(time.RFC1123),
			chk.Equals, policies[i].ExpiryTime.UTC().Round(time.Second).Format(time.RFC1123))

		c.Assert(newPolicies[i].CanRead, chk.Equals, policies[i].CanRead)
		c.Assert(newPolicies[i].CanAppend, chk.Equals, policies[i].CanAppend)
		c.Assert(newPolicies[i].CanUpdate, chk.Equals, policies[i].CanUpdate)
		c.Assert(newPolicies[i].CanDelete, chk.Equals, policies[i].CanDelete)
	}
}

func tableName(c *chk.C, extras ...string) string {
	// 32 is the max len for table names
	return nameGenerator(32, "table", alpha, c, extras)
}
