// Copyright 2012 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redis

import "errors"

// Subscription represents a subscribe or unsubscribe notification.
type Subscription struct {

	// Kind is "subscribe", "unsubscribe", "psubscribe" or "punsubscribe"
	Kind string

	// The channel that was changed.
	Channel string

	// The current number of subscriptions for connection.
	Count int
}

// Message represents a message notification.
type Message struct {

	// The originating channel.
	Channel string

	// The message data.
	Data []byte
}

// PMessage represents a pmessage notification.
type PMessage struct {

	// The matched pattern.
	Pattern string

	// The originating channel.
	Channel string

	// The message data.
	Data []byte
}

// Pong represents a pubsub pong notification.
type Pong struct {
	Data string
}

// PubSubConn wraps a Conn with convenience methods for subscribers.
type PubSubConn struct {
	Conn Conn
}

// Close closes the connection.
func (c PubSubConn) Close() error {
	return c.Conn.Close()
}

// Subscribe subscribes the connection to the specified channels.
func (c PubSubConn) Subscribe(channel ...interface{}) error {
	c.Conn.Send("SUBSCRIBE", channel...)
	return c.Conn.Flush()
}

// PSubscribe subscribes the connection to the given patterns.
func (c PubSubConn) PSubscribe(channel ...interface{}) error {
	c.Conn.Send("PSUBSCRIBE", channel...)
	return c.Conn.Flush()
}

// Unsubscribe unsubscribes the connection from the given channels, or from all
// of them if none is given.
func (c PubSubConn) Unsubscribe(channel ...interface{}) error {
	c.Conn.Send("UNSUBSCRIBE", channel...)
	return c.Conn.Flush()
}

// PUnsubscribe unsubscribes the connection from the given patterns, or from all
// of them if none is given.
func (c PubSubConn) PUnsubscribe(channel ...interface{}) error {
	c.Conn.Send("PUNSUBSCRIBE", channel...)
	return c.Conn.Flush()
}

// Ping sends a PING to the server with the specified data.
func (c PubSubConn) Ping(data string) error {
	c.Conn.Send("PING", data)
	return c.Conn.Flush()
}

// Receive returns a pushed message as a Subscription, Message, PMessage, Pong
// or error. The return value is intended to be used directly in a type switch
// as illustrated in the PubSubConn example.
func (c PubSubConn) Receive() interface{} {
	reply, err := Values(c.Conn.Receive())
	if err != nil {
		return err
	}

	var kind string
	reply, err = Scan(reply, &kind)
	if err != nil {
		return err
	}

	switch kind {
	case "message":
		var m Message
		if _, err := Scan(reply, &m.Channel, &m.Data); err != nil {
			return err
		}
		return m
	case "pmessage":
		var pm PMessage
		if _, err := Scan(reply, &pm.Pattern, &pm.Channel, &pm.Data); err != nil {
			return err
		}
		return pm
	case "subscribe", "psubscribe", "unsubscribe", "punsubscribe":
		s := Subscription{Kind: kind}
		if _, err := Scan(reply, &s.Channel, &s.Count); err != nil {
			return err
		}
		return s
	case "pong":
		var p Pong
		if _, err := Scan(reply, &p.Data); err != nil {
			return err
		}
		return p
	}
	return errors.New("redigo: unknown pubsub notification")
}
