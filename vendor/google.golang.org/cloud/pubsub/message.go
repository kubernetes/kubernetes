// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pubsub

import (
	"encoding/base64"

	raw "google.golang.org/api/pubsub/v1"
)

// Message represents a Pub/Sub message.
type Message struct {
	// ID identifies this message.
	// This ID is assigned by the server and is populated for Messages obtained from a subscription.
	// It is otherwise ignored.
	ID string

	// Data is the actual data in the message.
	Data []byte

	// Attributes represents the key-value pairs the current message
	// is labelled with.
	Attributes map[string]string

	// AckID is the identifier to acknowledge this message.
	AckID string
	// TODO(mcgreevy): unexport AckID.

	// TODO(mcgreevy): add publish time.

	calledDone bool

	// The iterator that created this Message.
	it *Iterator
}

func toMessage(resp *raw.ReceivedMessage) (*Message, error) {
	if resp.Message == nil {
		return &Message{AckID: resp.AckId}, nil
	}
	data, err := base64.StdEncoding.DecodeString(resp.Message.Data)
	if err != nil {
		return nil, err
	}
	return &Message{
		AckID:      resp.AckId,
		Data:       data,
		Attributes: resp.Message.Attributes,
		ID:         resp.Message.MessageId,
	}, nil
}

// Done completes the processing of a Message that was returned from an Iterator.
// ack indicates whether the message should be acknowledged.
// Client code must call Done when finished for each Message returned by an iterator.
// Done may only be called on Messages returned by an iterator.
// If message acknowledgement fails, the Message will be redelivered.
// Calls to Done have no effect after the first call.
func (m *Message) Done(ack bool) {
	if m.calledDone {
		return
	}
	m.calledDone = true
	m.it.done(m.AckID, ack)
}
