// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package xmpp

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/xmpp"
)

func newPresenceResponse(isAvailable bool, presence pb.PresenceResponse_SHOW, valid bool) *pb.PresenceResponse {
	return &pb.PresenceResponse{
		IsAvailable: proto.Bool(isAvailable),
		Presence:    presence.Enum(),
		Valid:       proto.Bool(valid),
	}
}

func setPresenceResponse(m *pb.PresenceResponse, isAvailable bool, presence pb.PresenceResponse_SHOW, valid bool) {
	m.IsAvailable = &isAvailable
	m.Presence = presence.Enum()
	m.Valid = &valid
}

func TestGetPresence(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "xmpp", "GetPresence", func(in *pb.PresenceRequest, out *pb.PresenceResponse) error {
		if jid := in.GetJid(); jid != "user@example.com" {
			return fmt.Errorf("bad jid %q", jid)
		}
		setPresenceResponse(out, true, pb.PresenceResponse_CHAT, true)
		return nil
	})

	presence, err := GetPresence(c, "user@example.com", "")
	if err != nil {
		t.Fatalf("GetPresence: %v", err)
	}

	if presence != "chat" {
		t.Errorf("GetPresence: got %#v, want %#v", presence, pb.PresenceResponse_CHAT)
	}
}

func TestGetPresenceMultiSingleJID(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "xmpp", "BulkGetPresence", func(in *pb.BulkPresenceRequest, out *pb.BulkPresenceResponse) error {
		if !reflect.DeepEqual(in.Jid, []string{"user@example.com"}) {
			return fmt.Errorf("bad request jids %#v", in.Jid)
		}
		out.PresenceResponse = []*pb.PresenceResponse{
			newPresenceResponse(true, pb.PresenceResponse_NORMAL, true),
		}
		return nil
	})

	presence, err := GetPresenceMulti(c, []string{"user@example.com"}, "")
	if err != nil {
		t.Fatalf("GetPresenceMulti: %v", err)
	}
	if !reflect.DeepEqual(presence, []string{""}) {
		t.Errorf("GetPresenceMulti: got %s, want %s", presence, []string{""})
	}
}

func TestGetPresenceMultiJID(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "xmpp", "BulkGetPresence", func(in *pb.BulkPresenceRequest, out *pb.BulkPresenceResponse) error {
		if !reflect.DeepEqual(in.Jid, []string{"user@example.com", "user2@example.com"}) {
			return fmt.Errorf("bad request jids %#v", in.Jid)
		}
		out.PresenceResponse = []*pb.PresenceResponse{
			newPresenceResponse(true, pb.PresenceResponse_NORMAL, true),
			newPresenceResponse(true, pb.PresenceResponse_AWAY, true),
		}
		return nil
	})

	jids := []string{"user@example.com", "user2@example.com"}
	presence, err := GetPresenceMulti(c, jids, "")
	if err != nil {
		t.Fatalf("GetPresenceMulti: %v", err)
	}
	want := []string{"", "away"}
	if !reflect.DeepEqual(presence, want) {
		t.Errorf("GetPresenceMulti: got %v, want %v", presence, want)
	}
}

func TestGetPresenceMultiFromJID(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "xmpp", "BulkGetPresence", func(in *pb.BulkPresenceRequest, out *pb.BulkPresenceResponse) error {
		if !reflect.DeepEqual(in.Jid, []string{"user@example.com", "user2@example.com"}) {
			return fmt.Errorf("bad request jids %#v", in.Jid)
		}
		if jid := in.GetFromJid(); jid != "bot@appspot.com" {
			return fmt.Errorf("bad from jid %q", jid)
		}
		out.PresenceResponse = []*pb.PresenceResponse{
			newPresenceResponse(true, pb.PresenceResponse_NORMAL, true),
			newPresenceResponse(true, pb.PresenceResponse_CHAT, true),
		}
		return nil
	})

	jids := []string{"user@example.com", "user2@example.com"}
	presence, err := GetPresenceMulti(c, jids, "bot@appspot.com")
	if err != nil {
		t.Fatalf("GetPresenceMulti: %v", err)
	}
	want := []string{"", "chat"}
	if !reflect.DeepEqual(presence, want) {
		t.Errorf("GetPresenceMulti: got %v, want %v", presence, want)
	}
}

func TestGetPresenceMultiInvalid(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "xmpp", "BulkGetPresence", func(in *pb.BulkPresenceRequest, out *pb.BulkPresenceResponse) error {
		if !reflect.DeepEqual(in.Jid, []string{"user@example.com", "user2@example.com"}) {
			return fmt.Errorf("bad request jids %#v", in.Jid)
		}
		out.PresenceResponse = []*pb.PresenceResponse{
			newPresenceResponse(true, pb.PresenceResponse_EXTENDED_AWAY, true),
			newPresenceResponse(true, pb.PresenceResponse_CHAT, false),
		}
		return nil
	})

	jids := []string{"user@example.com", "user2@example.com"}
	presence, err := GetPresenceMulti(c, jids, "")

	wantErr := appengine.MultiError{nil, ErrInvalidJID}
	if !reflect.DeepEqual(err, wantErr) {
		t.Fatalf("GetPresenceMulti: got %#v, want %#v", err, wantErr)
	}

	want := []string{"xa", ""}
	if !reflect.DeepEqual(presence, want) {
		t.Errorf("GetPresenceMulti: got %#v, want %#v", presence, want)
	}
}

func TestGetPresenceMultiUnavailable(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "xmpp", "BulkGetPresence", func(in *pb.BulkPresenceRequest, out *pb.BulkPresenceResponse) error {
		if !reflect.DeepEqual(in.Jid, []string{"user@example.com", "user2@example.com"}) {
			return fmt.Errorf("bad request jids %#v", in.Jid)
		}
		out.PresenceResponse = []*pb.PresenceResponse{
			newPresenceResponse(false, pb.PresenceResponse_AWAY, true),
			newPresenceResponse(false, pb.PresenceResponse_DO_NOT_DISTURB, true),
		}
		return nil
	})

	jids := []string{"user@example.com", "user2@example.com"}
	presence, err := GetPresenceMulti(c, jids, "")

	wantErr := appengine.MultiError{
		ErrPresenceUnavailable,
		ErrPresenceUnavailable,
	}
	if !reflect.DeepEqual(err, wantErr) {
		t.Fatalf("GetPresenceMulti: got %#v, want %#v", err, wantErr)
	}
	want := []string{"", ""}
	if !reflect.DeepEqual(presence, want) {
		t.Errorf("GetPresenceMulti: got %#v, want %#v", presence, want)
	}
}
