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

import chk "gopkg.in/check.v1"

type StorageMessageSuite struct{}

var _ = chk.Suite(&StorageMessageSuite{})

func (s *StorageMessageSuite) Test_pathForMessage(c *chk.C) {
	m := getQueueClient(c).GetQueueReference("q").GetMessageReference("m")
	m.ID = "ID"
	c.Assert(m.buildPath(), chk.Equals, "/q/messages/ID")
}

func (s *StorageMessageSuite) TestDeleteMessages(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	q := cli.GetQueueReference(queueName(c))
	c.Assert(q.Create(nil), chk.IsNil)
	defer q.Delete(nil)

	m := q.GetMessageReference("message")
	c.Assert(m.Put(nil), chk.IsNil)

	options := GetMessagesOptions{
		VisibilityTimeout: 1,
	}
	list, err := q.GetMessages(&options)
	c.Assert(err, chk.IsNil)
	c.Assert(list, chk.HasLen, 1)

	c.Assert(list[0].Delete(nil), chk.IsNil)
}

func (s *StorageMessageSuite) TestPutMessage_Peek(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue := cli.GetQueueReference(queueName(c))
	c.Assert(queue.Create(nil), chk.IsNil)
	defer queue.Delete(nil)

	msg := queue.GetMessageReference(string(content(64 * 1024))) // exercise max length
	c.Assert(msg.Put(nil), chk.IsNil)

	list, err := queue.PeekMessages(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(list), chk.Equals, 1)
	c.Assert(list[0].Text, chk.Equals, msg.Text)
}

func (s *StorageMessageSuite) TestPutMessage_Peek_Update_Delete(c *chk.C) {
	cli := getQueueClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	queue := cli.GetQueueReference(queueName(c))
	c.Assert(queue.Create(nil), chk.IsNil)
	defer queue.Delete(nil)

	msg1 := queue.GetMessageReference(string(content(64 * 1024))) // exercise max length
	msg2 := queue.GetMessageReference("and other message")
	c.Assert(msg1.Put(nil), chk.IsNil)
	c.Assert(msg2.Put(nil), chk.IsNil)

	list, err := queue.GetMessages(&GetMessagesOptions{NumOfMessages: 2, VisibilityTimeout: 2})
	c.Assert(err, chk.IsNil)
	c.Assert(len(list), chk.Equals, 2)
	c.Assert(list[0].Text, chk.Equals, msg1.Text)
	c.Assert(list[1].Text, chk.Equals, msg2.Text)

	list[0].Text = "updated message"
	c.Assert(list[0].Update(&UpdateMessageOptions{VisibilityTimeout: 2}), chk.IsNil)

	c.Assert(list[1].Delete(nil), chk.IsNil)
}
