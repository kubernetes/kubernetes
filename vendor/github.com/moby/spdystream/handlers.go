/*
   Copyright 2014-2021 Docker Inc.

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

package spdystream

import (
	"io"
	"net/http"
)

// MirrorStreamHandler mirrors all streams.
func MirrorStreamHandler(stream *Stream) {
	replyErr := stream.SendReply(http.Header{}, false)
	if replyErr != nil {
		return
	}

	go func() {
		io.Copy(stream, stream)
		stream.Close()
	}()
	go func() {
		for {
			header, receiveErr := stream.ReceiveHeader()
			if receiveErr != nil {
				return
			}
			sendErr := stream.SendHeader(header, false)
			if sendErr != nil {
				return
			}
		}
	}()
}

// NoopStreamHandler does nothing when stream connects.
func NoOpStreamHandler(stream *Stream) {
	stream.SendReply(http.Header{}, false)
}
