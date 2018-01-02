// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package mail provides the means of sending email from an
App Engine application.

Example:
	msg := &mail.Message{
		Sender:  "romeo@montague.com",
		To:      []string{"Juliet <juliet@capulet.org>"},
		Subject: "See you tonight",
		Body:    "Don't forget our plans. Hark, 'til later.",
	}
	if err := mail.Send(c, msg); err != nil {
		log.Errorf(c, "Alas, my user, the email failed to sendeth: %v", err)
	}
*/
package mail // import "google.golang.org/appengine/mail"

import (
	"net/mail"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
	bpb "google.golang.org/appengine/internal/base"
	pb "google.golang.org/appengine/internal/mail"
)

// A Message represents an email message.
// Addresses may be of any form permitted by RFC 822.
type Message struct {
	// Sender must be set, and must be either an application admin
	// or the currently signed-in user.
	Sender  string
	ReplyTo string // may be empty

	// At least one of these slices must have a non-zero length,
	// except when calling SendToAdmins.
	To, Cc, Bcc []string

	Subject string

	// At least one of Body or HTMLBody must be non-empty.
	Body     string
	HTMLBody string

	Attachments []Attachment

	// Extra mail headers.
	// See https://cloud.google.com/appengine/docs/go/mail/
	// for permissible headers.
	Headers mail.Header
}

// An Attachment represents an email attachment.
type Attachment struct {
	// Name must be set to a valid file name.
	Name      string
	Data      []byte
	ContentID string
}

// Send sends an email message.
func Send(c context.Context, msg *Message) error {
	return send(c, "Send", msg)
}

// SendToAdmins sends an email message to the application's administrators.
func SendToAdmins(c context.Context, msg *Message) error {
	return send(c, "SendToAdmins", msg)
}

func send(c context.Context, method string, msg *Message) error {
	req := &pb.MailMessage{
		Sender:  &msg.Sender,
		To:      msg.To,
		Cc:      msg.Cc,
		Bcc:     msg.Bcc,
		Subject: &msg.Subject,
	}
	if msg.ReplyTo != "" {
		req.ReplyTo = &msg.ReplyTo
	}
	if msg.Body != "" {
		req.TextBody = &msg.Body
	}
	if msg.HTMLBody != "" {
		req.HtmlBody = &msg.HTMLBody
	}
	if len(msg.Attachments) > 0 {
		req.Attachment = make([]*pb.MailAttachment, len(msg.Attachments))
		for i, att := range msg.Attachments {
			req.Attachment[i] = &pb.MailAttachment{
				FileName: proto.String(att.Name),
				Data:     att.Data,
			}
			if att.ContentID != "" {
				req.Attachment[i].ContentID = proto.String(att.ContentID)
			}
		}
	}
	for key, vs := range msg.Headers {
		for _, v := range vs {
			req.Header = append(req.Header, &pb.MailHeader{
				Name:  proto.String(key),
				Value: proto.String(v),
			})
		}
	}
	res := &bpb.VoidProto{}
	if err := internal.Call(c, "mail", method, req, res); err != nil {
		return err
	}
	return nil
}

func init() {
	internal.RegisterErrorCodeMap("mail", pb.MailServiceError_ErrorCode_name)
}
