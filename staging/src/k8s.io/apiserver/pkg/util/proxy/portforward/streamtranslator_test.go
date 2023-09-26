/*
Copyright 2023 The Kubernetes Authors.

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

package portforward

import (
	"fmt"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
)

func TestStreamTranslator_RequestStreams(t *testing.T) {
	rs := newRequestStreams()
	testRequestID := 1113
	assert.Equal(t, 0, len(rs.requestStreamsMap), "initially there are no requests in map")
	sp1 := rs.get(testRequestID)
	require.NotNil(t, sp1)
	assert.Equal(t, 1, len(rs.requestStreamsMap), "initial get creates a request")
	sp2 := rs.get(testRequestID)
	require.NotNil(t, sp2)
	assert.Equal(t, 1, len(rs.requestStreamsMap), "retrieving the same request does not create a request")
	assert.Equal(t, sp1, sp2)
	anotherRequestID := 32223
	sp3 := rs.get(anotherRequestID)
	require.NotNil(t, sp3)
	assert.Equal(t, 2, len(rs.requestStreamsMap), "retrieving separate request creates the request")
	rs.remove(testRequestID)
	assert.Equal(t, 1, len(rs.requestStreamsMap), "removing request should succeed")
	rs.remove(anotherRequestID)
	assert.Equal(t, 0, len(rs.requestStreamsMap), "removing another request should succeed")
}

func TestStreamTranslator_RequestStreamsCleanup(t *testing.T) {
	rs := newRequestStreams()
	testRequestID := 352532
	assert.Equal(t, 0, len(rs.requestStreamsMap))
	sp := rs.get(testRequestID)
	require.NotNil(t, sp)
	assert.Equal(t, 1, len(rs.requestStreamsMap))
	cleanupStreams := rs.cleanup(100 * time.Second)
	assert.Equal(t, 0, len(cleanupStreams), "streamPair without lastUpdate time should not timeout")
	sp.lastUpdate = time.Now().UnixMicro()
	cleanupStreams = rs.cleanup(1000 * time.Hour)
	assert.Equal(t, 0, len(cleanupStreams), "streamPair should not have timed out")
	sp.lastUpdate = int64(319324882000000) // 1980
	cleanupStreams = rs.cleanup(1 * time.Second)
	assert.Equal(t, 1, len(cleanupStreams), "streamPairs should have timed out")
	assert.Equal(t, sp, cleanupStreams[0], "streamPairs to cleanup should be the same")
}

func TestStreamTranslator_StreamPairsGetStreams(t *testing.T) {
	sp := &streamPairs{}
	// Initially there are no streams in the "streamPairs"
	assert.Equal(t, 0, len(sp.getWebsocketStreams()), "initially should have no websocket streams")
	assert.Equal(t, 0, len(sp.getSpdyStreams()), "initially should have no spdy streams")
	// Single pair of streams
	wsHeaders := http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeError)
	spdyHeaders := http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeError)
	wsStream1 := &fakeStream{id: 1, headers: wsHeaders}
	spdyStream1 := &fakeStream{id: 2, headers: spdyHeaders}
	// Add a first pair of streams to the "streamPairs"
	_, err := sp.addStreamPair(wsStream1, spdyStream1)
	require.NoError(t, err)
	assert.Equal(t, 1, len(sp.getWebsocketStreams()), "should return one websocket stream")
	assert.Equal(t, 1, len(sp.getSpdyStreams()), "should return one spdy stream")
	// Add a second pair of streams
	wsHeaders = http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeData)
	spdyHeaders = http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeData)
	wsStream2 := &fakeStream{id: 3, headers: wsHeaders}
	spdyStream2 := &fakeStream{id: 4, headers: spdyHeaders}
	_, err = sp.addStreamPair(wsStream2, spdyStream2)
	require.NoError(t, err)
	assert.Equal(t, 2, len(sp.getWebsocketStreams()), "should return two websocket streams")
	assert.Equal(t, 2, len(sp.getSpdyStreams()), "should return two spdy streams")
}

func TestStreamTranslator_StreamPairsAddStreamPair(t *testing.T) {
	// Single pair of streams is *not* ready.
	wsHeaders := http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeError)
	spdyHeaders := http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeError)
	wsStream1 := &fakeStream{id: 1, headers: wsHeaders}
	spdyStream1 := &fakeStream{id: 2, headers: spdyHeaders}
	sp := &streamPairs{}
	ready, err := sp.addStreamPair(wsStream1, spdyStream1)
	require.NoError(t, err)
	assert.Equal(t, false, ready, "single stream pair is not ready")
	// Second pair of streams *is* ready.
	wsHeaders = http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeData)
	spdyHeaders = http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeData)
	wsStream2 := &fakeStream{id: 3, headers: wsHeaders}
	spdyStream2 := &fakeStream{id: 4, headers: spdyHeaders}
	ready, err = sp.addStreamPair(wsStream2, spdyStream2)
	require.NoError(t, err)
	assert.Equal(t, true, ready, "both data and error streams means ready to portforward")
	// Streams with differing stream types is an error.
	_, err = sp.addStreamPair(wsStream2, spdyStream1)
	require.Error(t, err, "differing stream types is an error")
	// Duplicate streams are an error.
	wsStream3 := &fakeStream{id: 5, headers: wsHeaders}
	spdyStream3 := &fakeStream{id: 6, headers: spdyHeaders}
	_, err = sp.addStreamPair(wsStream3, spdyStream3)
	require.Error(t, err, "duplicate stream pairs should be an error")
	// Unknown stream types are error.
	wsHeaders = http.Header{}
	wsHeaders.Set(v1.StreamType, "unknown")
	spdyHeaders = http.Header{}
	spdyHeaders.Set(v1.StreamType, "unknown")
	unknownStream1 := &fakeStream{id: 7, headers: wsHeaders}
	unknownStream2 := &fakeStream{id: 8, headers: spdyHeaders}
	_, err = sp.addStreamPair(unknownStream1, unknownStream2)
	require.Error(t, err, "unknown stream type is an error")
	// Stream without type is an error.
	emptyHeaders := http.Header{}
	errStream := &fakeStream{id: 9, headers: emptyHeaders}
	sp = &streamPairs{}
	_, err = sp.addStreamPair(errStream, spdyStream1)
	require.Error(t, err, "missing stream type header is error")
}

func TestStreamTranslator_StreamPairsIsTimedOut(t *testing.T) {
	sp := &streamPairs{}
	assert.False(t, sp.isTimedOut(0*time.Second), "if lastUpdate time not set, no timeout")
	// Add a single set of streams to "streamPairs"
	wsHeaders := http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeError)
	spdyHeaders := http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeError)
	wsStream1 := &fakeStream{id: 1, headers: wsHeaders}
	spdyStream1 := &fakeStream{id: 2, headers: spdyHeaders}
	_, err := sp.addStreamPair(wsStream1, spdyStream1)
	require.NoError(t, err)
	// Set calculated timeout to be in distant future -- no timeout.
	sp.lastUpdate = int64(3285782482000000) // 2075
	assert.False(t, sp.isTimedOut(1*time.Second), "last update in distant future -- no timeout")
	// Set calculated timeout to be very long in the past--timeout.
	sp.lastUpdate = int64(319324882000000) // 1980
	assert.True(t, sp.isTimedOut(1*time.Second), "last update in distant past -- timeout")
}

func TestStreamTranslator_StreamPairsIsComplete(t *testing.T) {
	sp := &streamPairs{}
	assert.False(t, sp.isComplete(), "no streams means streamPairs is not complete")
	// Add a single set of streams to "streamPairs"
	wsHeaders := http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeError)
	spdyHeaders := http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeError)
	wsStream1 := &fakeStream{id: 1, headers: wsHeaders}
	spdyStream1 := &fakeStream{id: 2, headers: spdyHeaders}
	_, err := sp.addStreamPair(wsStream1, spdyStream1)
	require.NoError(t, err)
	assert.False(t, sp.isComplete(), "one stream pairs means streamPairs is not complete")
	wsHeaders = http.Header{}
	wsHeaders.Set(v1.StreamType, v1.StreamTypeData)
	spdyHeaders = http.Header{}
	spdyHeaders.Set(v1.StreamType, v1.StreamTypeData)
	wsStream2 := &fakeStream{id: 3, headers: wsHeaders}
	spdyStream2 := &fakeStream{id: 4, headers: spdyHeaders}
	_, err = sp.addStreamPair(wsStream2, spdyStream2)
	require.NoError(t, err)
	assert.True(t, sp.isComplete(), "two stream pairs means streamPairs is complete")
}

func TestStreamTranslator_GetRequestID(t *testing.T) {
	_, err := getRequestID(&fakeStream{})
	assert.Error(t, err, "missing requestID header is an error")
	headers := http.Header{}
	headers.Set(v1.PortForwardRequestIDHeader, "notaninteger")
	_, err = getRequestID(&fakeStream{id: 1, headers: headers})
	assert.Error(t, err, "requestID header not an integer is an error")
	expected := 3298892
	headers.Set(v1.PortForwardRequestIDHeader, strconv.Itoa(expected))
	actual, err := getRequestID(&fakeStream{id: 1, headers: headers})
	require.NoError(t, err)
	assert.Equal(t, expected, actual, "requestID's should be the same")
}

func TestStreamTranslator_IsConnectionClosedError(t *testing.T) {
	isError := isClosedConnectionError(nil)
	assert.False(t, isError, "nil error is not connection closed error")
	isError = isClosedConnectionError(fmt.Errorf("not a connection closed error"))
	assert.False(t, isError, "error is not connection closed error")
	isError = isClosedConnectionError(fmt.Errorf("use of closed network connection"))
	assert.True(t, isError, "error *is* connection closed error")
}

// fakeStream implements "httpstream.Stream".
type fakeStream struct {
	id      int
	headers http.Header
	toRead  []byte
	written []byte
}

func (fs *fakeStream) Read(p []byte) (n int, err error) {
	numRead := copy(p, fs.toRead)
	return numRead, nil
}

func (fs *fakeStream) Write(p []byte) (n int, err error) {
	numWritten := copy(fs.written, p)
	return numWritten, nil
}

func (fs *fakeStream) Close() error {
	return nil
}

func (fs *fakeStream) Reset() error {
	return nil
}

func (fs *fakeStream) Headers() http.Header {
	return fs.headers
}

func (fs *fakeStream) Identifier() uint32 {
	return uint32(fs.id)
}
