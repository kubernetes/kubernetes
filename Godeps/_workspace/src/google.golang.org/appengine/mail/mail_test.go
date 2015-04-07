// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package mail

import (
	"testing"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal/aetesting"
	basepb "google.golang.org/appengine/internal/base"
	pb "google.golang.org/appengine/internal/mail"
)

func TestMessageConstruction(t *testing.T) {
	var got *pb.MailMessage
	c := aetesting.FakeSingleContext(t, "mail", "Send", func(in *pb.MailMessage, out *basepb.VoidProto) error {
		got = in
		return nil
	})

	msg := &Message{
		Sender: "dsymonds@example.com",
		To:     []string{"nigeltao@example.com"},
		Body:   "Hey, lunch time?",
		Attachments: []Attachment{
			// Regression test for a prod bug. The address of a range variable was used when
			// constructing the outgoing proto, so multiple attachments used the same name.
			{
				Name:      "att1.txt",
				Data:      []byte("data1"),
				ContentID: "<att1>",
			},
			{
				Name: "att2.txt",
				Data: []byte("data2"),
			},
		},
	}
	if err := Send(c, msg); err != nil {
		t.Fatalf("Send: %v", err)
	}
	want := &pb.MailMessage{
		Sender:   proto.String("dsymonds@example.com"),
		To:       []string{"nigeltao@example.com"},
		Subject:  proto.String(""),
		TextBody: proto.String("Hey, lunch time?"),
		Attachment: []*pb.MailAttachment{
			{
				FileName:  proto.String("att1.txt"),
				Data:      []byte("data1"),
				ContentID: proto.String("<att1>"),
			},
			{
				FileName: proto.String("att2.txt"),
				Data:     []byte("data2"),
			},
		},
	}
	if !proto.Equal(got, want) {
		t.Errorf("Bad proto for %+v\n got %v\nwant %v", msg, got, want)
	}
}
