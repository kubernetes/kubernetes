package storage

import (
	"time"

	chk "gopkg.in/check.v1"
)

type StorageQueueSuite struct{}

var _ = chk.Suite(&StorageQueueSuite{})

func getQueueClient(c *chk.C) QueueServiceClient {
	return getBasicClient(c).GetQueueService()
}

func (s *StorageQueueSuite) Test_pathForQueue(c *chk.C) {
	c.Assert(pathForQueue("q"), chk.Equals, "/q")
}

func (s *StorageQueueSuite) Test_pathForQueueMessages(c *chk.C) {
	c.Assert(pathForQueueMessages("q"), chk.Equals, "/q/messages")
}

func (s *StorageQueueSuite) Test_pathForMessage(c *chk.C) {
	c.Assert(pathForMessage("q", "m"), chk.Equals, "/q/messages/m")
}

func (s *StorageQueueSuite) TestCreateQueue_DeleteQueue(c *chk.C) {
	cli := getQueueClient(c)
	name := randString(20)
	c.Assert(cli.CreateQueue(name), chk.IsNil)
	c.Assert(cli.DeleteQueue(name), chk.IsNil)
}

func (s *StorageQueueSuite) Test_GetMetadata_GetApproximateCount(c *chk.C) {
	cli := getQueueClient(c)
	name := randString(20)
	c.Assert(cli.CreateQueue(name), chk.IsNil)
	defer cli.DeleteQueue(name)

	qm, err := cli.GetMetadata(name)
	c.Assert(err, chk.IsNil)
	c.Assert(qm.ApproximateMessageCount, chk.Equals, 0)

	for ix := 0; ix < 3; ix++ {
		err = cli.PutMessage(name, "foobar", PutMessageParameters{})
		c.Assert(err, chk.IsNil)
	}
	time.Sleep(1 * time.Second)

	qm, err = cli.GetMetadata(name)
	c.Assert(err, chk.IsNil)
	c.Assert(qm.ApproximateMessageCount, chk.Equals, 3)
}

func (s *StorageQueueSuite) Test_SetMetadataGetMetadata_Roundtrips(c *chk.C) {
	cli := getQueueClient(c)
	name := randString(20)
	c.Assert(cli.CreateQueue(name), chk.IsNil)
	defer cli.DeleteQueue(name)

	metadata := make(map[string]string)
	metadata["Foo1"] = "bar1"
	metadata["fooBaz"] = "bar"
	err := cli.SetMetadata(name, metadata)
	c.Assert(err, chk.IsNil)

	qm, err := cli.GetMetadata(name)
	c.Assert(err, chk.IsNil)
	c.Assert(qm.UserDefinedMetadata["foo1"], chk.Equals, "bar1")
	c.Assert(qm.UserDefinedMetadata["foobaz"], chk.Equals, "bar")
}

func (s *StorageQueueSuite) TestQueueExists(c *chk.C) {
	cli := getQueueClient(c)
	ok, err := cli.QueueExists("nonexistent-queue")
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	name := randString(20)
	c.Assert(cli.CreateQueue(name), chk.IsNil)
	defer cli.DeleteQueue(name)

	ok, err = cli.QueueExists(name)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageQueueSuite) TestPostMessage_PeekMessage_DeleteMessage(c *chk.C) {
	q := randString(20)
	cli := getQueueClient(c)
	c.Assert(cli.CreateQueue(q), chk.IsNil)
	defer cli.DeleteQueue(q)

	msg := randString(64 * 1024) // exercise max length
	c.Assert(cli.PutMessage(q, msg, PutMessageParameters{}), chk.IsNil)
	r, err := cli.PeekMessages(q, PeekMessagesParameters{})
	c.Assert(err, chk.IsNil)
	c.Assert(len(r.QueueMessagesList), chk.Equals, 1)
	c.Assert(r.QueueMessagesList[0].MessageText, chk.Equals, msg)
}

func (s *StorageQueueSuite) TestGetMessages(c *chk.C) {
	q := randString(20)
	cli := getQueueClient(c)
	c.Assert(cli.CreateQueue(q), chk.IsNil)
	defer cli.DeleteQueue(q)

	n := 4
	for i := 0; i < n; i++ {
		c.Assert(cli.PutMessage(q, randString(10), PutMessageParameters{}), chk.IsNil)
	}

	r, err := cli.GetMessages(q, GetMessagesParameters{NumOfMessages: n})
	c.Assert(err, chk.IsNil)
	c.Assert(len(r.QueueMessagesList), chk.Equals, n)
}

func (s *StorageQueueSuite) TestDeleteMessages(c *chk.C) {
	q := randString(20)
	cli := getQueueClient(c)
	c.Assert(cli.CreateQueue(q), chk.IsNil)
	defer cli.DeleteQueue(q)

	c.Assert(cli.PutMessage(q, "message", PutMessageParameters{}), chk.IsNil)
	r, err := cli.GetMessages(q, GetMessagesParameters{VisibilityTimeout: 1})
	c.Assert(err, chk.IsNil)
	c.Assert(len(r.QueueMessagesList), chk.Equals, 1)
	m := r.QueueMessagesList[0]
	c.Assert(cli.DeleteMessage(q, m.MessageID, m.PopReceipt), chk.IsNil)
}
