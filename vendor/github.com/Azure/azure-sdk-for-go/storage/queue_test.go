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

	chk "gopkg.in/check.v1"
)

type StorageQueueSuite struct{}

var _ = chk.Suite(&StorageQueueSuite{})

func getQueueClient(c *chk.C) *QueueServiceClient {
	cli := getBasicClient(c).GetQueueService()
	return &cli
}

func (s *StorageQueueSuite) Test_pathForQueue(c *chk.C) {
	c.Assert(getQueueClient(c).
		GetQueueReference("q").
		buildPath(), chk.Equals, "/q")
}

func (s *StorageQueueSuite) Test_pathForQueueMessages(c *chk.C) {
	c.Assert(getQueueClient(c).
		GetQueueReference("q").
		buildPathMessages(), chk.Equals, "/q/messages")
}

func (s *StorageQueueSuite) TestCreateQueue_DeleteQueue(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	q := cli.GetQueueReference(queueName(c))
	c.Assert(q.Create(nil), chk.IsNil)
	c.Assert(q.Delete(nil), chk.IsNil)
}

func (s *StorageQueueSuite) Test_GetMetadata_GetApproximateCount(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	err := queue1.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(queue1.AproxMessageCount, chk.Equals, uint64(0))

	queue2 := cli.GetQueueReference(queueName(c, "2"))
	c.Assert(queue2.Create(nil), chk.IsNil)
	defer queue2.Delete(nil)
	for ix := 0; ix < 3; ix++ {
		msg := queue2.GetMessageReference("lolrofl")
		err = msg.Put(nil)
		c.Assert(err, chk.IsNil)
	}
	time.Sleep(1 * time.Second)

	err = queue2.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(queue2.AproxMessageCount, chk.Equals, uint64(3))
}

func (s *StorageQueueSuite) Test_SetMetadataGetMetadata_Roundtrips(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	metadata := make(map[string]string)
	metadata["Lol1"] = "rofl1"
	metadata["lolBaz"] = "rofl"
	queue1.Metadata = metadata
	err := queue1.SetMetadata(nil)
	c.Assert(err, chk.IsNil)

	err = queue1.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(queue1.Metadata["lol1"], chk.Equals, metadata["Lol1"])
	c.Assert(queue1.Metadata["lolbaz"], chk.Equals, metadata["lolBaz"])
}

func (s *StorageQueueSuite) TestQueueExists(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "nonexistent"))
	ok, err := queue1.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	queue2 := cli.GetQueueReference(queueName(c, "exisiting"))
	c.Assert(queue2.Create(nil), chk.IsNil)
	defer queue2.Delete(nil)

	ok, err = queue2.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageQueueSuite) TestGetMessages(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue := cli.GetQueueReference(queueName(c))
	c.Assert(queue.Create(nil), chk.IsNil)
	defer queue.Delete(nil)

	msg := queue.GetMessageReference("message")
	n := 4
	for i := 0; i < n; i++ {
		c.Assert(msg.Put(nil), chk.IsNil)
	}

	list, err := queue.GetMessages(&GetMessagesOptions{NumOfMessages: n})
	c.Assert(err, chk.IsNil)
	c.Assert(len(list), chk.Equals, n)
}

func (s *StorageQueueSuite) TestDeleteMessages(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue := cli.GetQueueReference(queueName(c))
	c.Assert(queue.Create(nil), chk.IsNil)
	defer queue.Delete(nil)

	msg := queue.GetMessageReference("message")
	c.Assert(msg.Put(nil), chk.IsNil)
	list, err := queue.GetMessages(&GetMessagesOptions{VisibilityTimeout: 1})
	c.Assert(err, chk.IsNil)
	c.Assert(len(list), chk.Equals, 1)
	msg = &(list[0])
	c.Assert(msg.Delete(nil), chk.IsNil)
}

func queueName(c *chk.C, extras ...string) string {
	// 63 is the max len for shares
	return nameGenerator(63, "queue-", alphanum, c, extras)
}

func (s *StorageQueueSuite) Test_SetPermissionsAllTrueNoTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		CanRead:    true,
		CanAdd:     true,
		CanUpdate:  true,
		CanProcess: true,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)
	err := queue1.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageQueueSuite) Test_SetPermissionsAllTrueWithTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		CanRead:    true,
		CanAdd:     true,
		CanUpdate:  true,
		CanProcess: true,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)

	options := SetQueuePermissionOptions{Timeout: 30}
	err := queue1.SetPermissions(perms, &options)
	c.Assert(err, chk.IsNil)

}

func (s *StorageQueueSuite) Test_SetPermissionsAlternateTrueNoTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		CanRead:    true,
		CanAdd:     false,
		CanUpdate:  true,
		CanProcess: false,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)
	err := queue1.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)
}

func (s *StorageQueueSuite) Test_SetPermissionsAlternateTrueWithTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6)),
		CanRead:    true,
		CanAdd:     false,
		CanUpdate:  true,
		CanProcess: false,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)

	options := SetQueuePermissionOptions{Timeout: 30}
	err := queue1.SetPermissions(perms, &options)
	c.Assert(err, chk.IsNil)
}

func (s *StorageQueueSuite) Test_GetPermissionsAllTrueNoTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.UTC),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.UTC),
		CanRead:    true,
		CanAdd:     true,
		CanUpdate:  true,
		CanProcess: true,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)
	err := queue1.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)

	returnedPerms, err := queue1.GetPermissions(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(returnedPerms.AccessPolicies, chk.HasLen, 1)

	c.Assert(returnedPerms.AccessPolicies[0].CanRead, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanAdd, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanUpdate, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanProcess, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].ID, chk.Equals, "GolangRocksOnAzure")
	c.Assert(returnedPerms.AccessPolicies[0].StartTime, chk.Equals, qapd.StartTime)
	c.Assert(returnedPerms.AccessPolicies[0].ExpiryTime, chk.Equals, qapd.ExpiryTime)
}

func (s *StorageQueueSuite) Test_GetPermissionsAllTrueWithTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.UTC),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.UTC),
		CanRead:    true,
		CanAdd:     true,
		CanUpdate:  true,
		CanProcess: true,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)
	err := queue1.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)

	options := GetQueuePermissionOptions{Timeout: 30}
	returnedPerms, err := queue1.GetPermissions(&options)
	c.Assert(err, chk.IsNil)
	c.Assert(returnedPerms.AccessPolicies, chk.HasLen, 1)

	c.Assert(returnedPerms.AccessPolicies[0].CanRead, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanAdd, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanUpdate, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanProcess, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].ID, chk.Equals, "GolangRocksOnAzure")
	c.Assert(returnedPerms.AccessPolicies[0].StartTime, chk.Equals, qapd.StartTime)
	c.Assert(returnedPerms.AccessPolicies[0].ExpiryTime, chk.Equals, qapd.ExpiryTime)

}

func (s *StorageQueueSuite) Test_GetPermissionsAlternateTrueNoTimeout(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue1 := cli.GetQueueReference(queueName(c, "1"))
	c.Assert(queue1.Create(nil), chk.IsNil)
	defer queue1.Delete(nil)

	perms := QueuePermissions{}
	qapd := QueueAccessPolicy{
		ID:         "GolangRocksOnAzure",
		StartTime:  time.Date(2050, time.December, 20, 21, 55, 0, 0, time.UTC),
		ExpiryTime: time.Date(2051, time.December, 20, 21, 55, 0, 0, time.UTC),
		CanRead:    true,
		CanAdd:     false,
		CanUpdate:  true,
		CanProcess: false,
	}
	perms.AccessPolicies = append(perms.AccessPolicies, qapd)
	err := queue1.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)

	returnedPerms, err := queue1.GetPermissions(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(returnedPerms.AccessPolicies, chk.HasLen, 1)

	c.Assert(returnedPerms.AccessPolicies[0].CanRead, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanAdd, chk.Equals, false)
	c.Assert(returnedPerms.AccessPolicies[0].CanUpdate, chk.Equals, true)
	c.Assert(returnedPerms.AccessPolicies[0].CanProcess, chk.Equals, false)
	c.Assert(returnedPerms.AccessPolicies[0].ID, chk.Equals, "GolangRocksOnAzure")
	c.Assert(returnedPerms.AccessPolicies[0].StartTime, chk.Equals, qapd.StartTime)
	c.Assert(returnedPerms.AccessPolicies[0].ExpiryTime, chk.Equals, qapd.ExpiryTime)
}
