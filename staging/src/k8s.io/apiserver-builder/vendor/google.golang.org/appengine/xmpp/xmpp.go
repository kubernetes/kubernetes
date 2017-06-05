// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package xmpp provides the means to send and receive instant messages
to and from users of XMPP-compatible services.

To send a message,
	m := &xmpp.Message{
		To:   []string{"kaylee@example.com"},
		Body: `Hi! How's the carrot?`,
	}
	err := m.Send(c)

To receive messages,
	func init() {
		xmpp.Handle(handleChat)
	}

	func handleChat(c context.Context, m *xmpp.Message) {
		// ...
	}
*/
package xmpp // import "google.golang.org/appengine/xmpp"

import (
	"errors"
	"fmt"
	"net/http"

	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/xmpp"
)

// Message represents an incoming chat message.
type Message struct {
	// Sender is the JID of the sender.
	// Optional for outgoing messages.
	Sender string

	// To is the intended recipients of the message.
	// Incoming messages will have exactly one element.
	To []string

	// Body is the body of the message.
	Body string

	// Type is the message type, per RFC 3921.
	// It defaults to "chat".
	Type string

	// RawXML is whether the body contains raw XML.
	RawXML bool
}

// Presence represents an outgoing presence update.
type Presence struct {
	// Sender is the JID (optional).
	Sender string

	// The intended recipient of the presence update.
	To string

	// Type, per RFC 3921 (optional). Defaults to "available".
	Type string

	// State of presence (optional).
	// Valid values: "away", "chat", "xa", "dnd" (RFC 3921).
	State string

	// Free text status message (optional).
	Status string
}

var (
	ErrPresenceUnavailable = errors.New("xmpp: presence unavailable")
	ErrInvalidJID          = errors.New("xmpp: invalid JID")
)

// Handle arranges for f to be called for incoming XMPP messages.
// Only messages of type "chat" or "normal" will be handled.
func Handle(f func(c context.Context, m *Message)) {
	http.HandleFunc("/_ah/xmpp/message/chat/", func(_ http.ResponseWriter, r *http.Request) {
		f(appengine.NewContext(r), &Message{
			Sender: r.FormValue("from"),
			To:     []string{r.FormValue("to")},
			Body:   r.FormValue("body"),
		})
	})
}

// Send sends a message.
// If any failures occur with specific recipients, the error will be an appengine.MultiError.
func (m *Message) Send(c context.Context) error {
	req := &pb.XmppMessageRequest{
		Jid:    m.To,
		Body:   &m.Body,
		RawXml: &m.RawXML,
	}
	if m.Type != "" && m.Type != "chat" {
		req.Type = &m.Type
	}
	if m.Sender != "" {
		req.FromJid = &m.Sender
	}
	res := &pb.XmppMessageResponse{}
	if err := internal.Call(c, "xmpp", "SendMessage", req, res); err != nil {
		return err
	}

	if len(res.Status) != len(req.Jid) {
		return fmt.Errorf("xmpp: sent message to %d JIDs, but only got %d statuses back", len(req.Jid), len(res.Status))
	}
	me, any := make(appengine.MultiError, len(req.Jid)), false
	for i, st := range res.Status {
		if st != pb.XmppMessageResponse_NO_ERROR {
			me[i] = errors.New(st.String())
			any = true
		}
	}
	if any {
		return me
	}
	return nil
}

// Invite sends an invitation. If the from address is an empty string
// the default (yourapp@appspot.com/bot) will be used.
func Invite(c context.Context, to, from string) error {
	req := &pb.XmppInviteRequest{
		Jid: &to,
	}
	if from != "" {
		req.FromJid = &from
	}
	res := &pb.XmppInviteResponse{}
	return internal.Call(c, "xmpp", "SendInvite", req, res)
}

// Send sends a presence update.
func (p *Presence) Send(c context.Context) error {
	req := &pb.XmppSendPresenceRequest{
		Jid: &p.To,
	}
	if p.State != "" {
		req.Show = &p.State
	}
	if p.Type != "" {
		req.Type = &p.Type
	}
	if p.Sender != "" {
		req.FromJid = &p.Sender
	}
	if p.Status != "" {
		req.Status = &p.Status
	}
	res := &pb.XmppSendPresenceResponse{}
	return internal.Call(c, "xmpp", "SendPresence", req, res)
}

var presenceMap = map[pb.PresenceResponse_SHOW]string{
	pb.PresenceResponse_NORMAL:         "",
	pb.PresenceResponse_AWAY:           "away",
	pb.PresenceResponse_DO_NOT_DISTURB: "dnd",
	pb.PresenceResponse_CHAT:           "chat",
	pb.PresenceResponse_EXTENDED_AWAY:  "xa",
}

// GetPresence retrieves a user's presence.
// If the from address is an empty string the default
// (yourapp@appspot.com/bot) will be used.
// Possible return values are "", "away", "dnd", "chat", "xa".
// ErrPresenceUnavailable is returned if the presence is unavailable.
func GetPresence(c context.Context, to string, from string) (string, error) {
	req := &pb.PresenceRequest{
		Jid: &to,
	}
	if from != "" {
		req.FromJid = &from
	}
	res := &pb.PresenceResponse{}
	if err := internal.Call(c, "xmpp", "GetPresence", req, res); err != nil {
		return "", err
	}
	if !*res.IsAvailable || res.Presence == nil {
		return "", ErrPresenceUnavailable
	}
	presence, ok := presenceMap[*res.Presence]
	if ok {
		return presence, nil
	}
	return "", fmt.Errorf("xmpp: unknown presence %v", *res.Presence)
}

// GetPresenceMulti retrieves multiple users' presence.
// If the from address is an empty string the default
// (yourapp@appspot.com/bot) will be used.
// Possible return values are "", "away", "dnd", "chat", "xa".
// If any presence is unavailable, an appengine.MultiError is returned
func GetPresenceMulti(c context.Context, to []string, from string) ([]string, error) {
	req := &pb.BulkPresenceRequest{
		Jid: to,
	}
	if from != "" {
		req.FromJid = &from
	}
	res := &pb.BulkPresenceResponse{}

	if err := internal.Call(c, "xmpp", "BulkGetPresence", req, res); err != nil {
		return nil, err
	}

	presences := make([]string, 0, len(res.PresenceResponse))
	errs := appengine.MultiError{}

	addResult := func(presence string, err error) {
		presences = append(presences, presence)
		errs = append(errs, err)
	}

	anyErr := false
	for _, subres := range res.PresenceResponse {
		if !subres.GetValid() {
			anyErr = true
			addResult("", ErrInvalidJID)
			continue
		}
		if !*subres.IsAvailable || subres.Presence == nil {
			anyErr = true
			addResult("", ErrPresenceUnavailable)
			continue
		}
		presence, ok := presenceMap[*subres.Presence]
		if ok {
			addResult(presence, nil)
		} else {
			anyErr = true
			addResult("", fmt.Errorf("xmpp: unknown presence %q", *subres.Presence))
		}
	}
	if anyErr {
		return presences, errs
	}
	return presences, nil
}

func init() {
	internal.RegisterErrorCodeMap("xmpp", pb.XmppServiceError_ErrorCode_name)
}
