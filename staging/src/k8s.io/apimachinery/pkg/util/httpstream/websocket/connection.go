/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package websocket

import (
	"io"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"k8s.io/apimachinery/pkg/util/httpstream"
)

// Connection holds the underlying websocket connection
type Connection struct {
	httpstream.Connection
	Conn *websocket.Conn

	//store streams by channel id
	channels map[int]httpstream.Stream
}

// Stream is a websockets implementation of an httpstream
type Stream struct {
	httpstream.Stream

	streamIn  *io.PipeReader
	streamOut *io.PipeWriter
}

// CreateStream does nothing as httpstream.Stream is a SPDY function, not a websocket concept
func (connection *Connection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	return nil, nil
}

// CreateStream creates a stream by the channel id
func (connection *Connection) CreateStrean(channelID int) (httpstream.Stream, error) {
	newStream := &Stream{}
	return newStream, nil
}

// Close does nothing as httpstream.Stream is a SPDY function, not a websocket concept
func (connection *Connection) Close() error {
	return nil
}

// CloseChan does nothing as httpstream.Stream is a SPDY function, not a websocket concept
func (connection *Connection) CloseChan() <-chan bool {
	out := make(chan bool)
	return out
}

// SetIdleTimeout does nothing as httpstream.Stream is a SPDY function, not a websocket concept
func (connection *Connection) SetIdleTimeout(timeout time.Duration) {

}
