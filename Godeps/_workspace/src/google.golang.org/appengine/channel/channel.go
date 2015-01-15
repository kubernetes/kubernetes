// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package channel implements the server side of App Engine's Channel API.

Create creates a new channel associated with the given clientID,
which must be unique to the client that will use the returned token.

	token, err := channel.Create(c, "player1")
	if err != nil {
		// handle error
	}
	// return token to the client in an HTTP response

Send sends a message to the client over the channel identified by clientID.

	channel.Send(c, "player1", "Game over!")
*/
package channel

import (
	"encoding/json"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	basepb "google.golang.org/appengine/internal/base"
	pb "google.golang.org/appengine/internal/channel"
)

// Create creates a channel and returns a token for use by the client.
// The clientID is an application-provided string used to identify the client.
func Create(c appengine.Context, clientID string) (token string, err error) {
	req := &pb.CreateChannelRequest{
		ApplicationKey: &clientID,
	}
	resp := &pb.CreateChannelResponse{}
	err = c.Call(service, "CreateChannel", req, resp, nil)
	token = resp.GetToken()
	return token, remapError(err)
}

// Send sends a message on the channel associated with clientID.
func Send(c appengine.Context, clientID, message string) error {
	req := &pb.SendMessageRequest{
		ApplicationKey: &clientID,
		Message:        &message,
	}
	resp := &basepb.VoidProto{}
	return remapError(c.Call(service, "SendChannelMessage", req, resp, nil))
}

// SendJSON is a helper function that sends a JSON-encoded value
// on the channel associated with clientID.
func SendJSON(c appengine.Context, clientID string, value interface{}) error {
	m, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return Send(c, clientID, string(m))
}

// remapError fixes any APIError referencing "xmpp" into one referencing "channel".
func remapError(err error) error {
	if e, ok := err.(*internal.APIError); ok {
		if e.Service == "xmpp" {
			e.Service = "channel"
		}
	}
	return err
}

var service = "xmpp" // prod

func init() {
	if appengine.IsDevAppServer() {
		service = "channel" // dev
	}
	internal.RegisterErrorCodeMap("channel", pb.ChannelServiceError_ErrorCode_name)
}
