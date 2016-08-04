/*
Copyright 2015 The Kubernetes Authors.

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

package wsstream

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/websocket"

	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/validation"
)

// The Websocket subprotocol "channel.k8s.io" prepends each binary message with a byte indicating
// the channel number (zero indexed) the message was sent on. Messages in both directions should
// prefix their messages with this channel byte. When used for remote execution, the channel numbers
// are by convention defined to match the POSIX file-descriptors assigned to STDIN, STDOUT, and STDERR
// (0, 1, and 2). No other conversion is performed on the raw subprotocol - writes are sent as they
// are received by the server.
//
// Example client session:
//
//    CONNECT http://server.com with subprotocol "channel.k8s.io"
//    WRITE []byte{0, 102, 111, 111, 10} # send "foo\n" on channel 0 (STDIN)
//    READ  []byte{1, 10}                # receive "\n" on channel 1 (STDOUT)
//    CLOSE
//
const channelWebSocketProtocol = "channel.k8s.io"

// The Websocket subprotocol "base64.channel.k8s.io" base64 encodes each message with a character
// indicating the channel number (zero indexed) the message was sent on. Messages in both directions
// should prefix their messages with this channel char. When used for remote execution, the channel
// numbers are by convention defined to match the POSIX file-descriptors assigned to STDIN, STDOUT,
// and STDERR ('0', '1', and '2'). The data received on the server is base64 decoded (and must be
// be valid) and data written by the server to the client is base64 encoded.
//
// Example client session:
//
//    CONNECT http://server.com with subprotocol "base64.channel.k8s.io"
//    WRITE []byte{48, 90, 109, 57, 118, 67, 103, 111, 61} # send "foo\n" (base64: "Zm9vCgo=") on channel '0' (STDIN)
//    READ  []byte{49, 67, 103, 61, 61} # receive "\n" (base64: "Cg==") on channel '1' (STDOUT)
//    CLOSE
//
const base64ChannelWebSocketProtocol = "base64.channel.k8s.io"

type codecType int

const (
	rawCodec codecType = iota
	base64Codec
)

type ChannelType int

const (
	IgnoreChannel ChannelType = iota
	ReadChannel
	WriteChannel
	ReadWriteChannel
)

var (
	// connectionUpgradeRegex matches any Connection header value that includes upgrade
	connectionUpgradeRegex = regexp.MustCompile("(^|.*,\\s*)upgrade($|\\s*,)")
)

// IsWebSocketRequest returns true if the incoming request contains connection upgrade headers
// for WebSockets.
func IsWebSocketRequest(req *http.Request) bool {
	return connectionUpgradeRegex.MatchString(strings.ToLower(req.Header.Get("Connection"))) && strings.ToLower(req.Header.Get("Upgrade")) == "websocket"
}

// IgnoreReceives reads from a WebSocket until it is closed, then returns. If timeout is set, the
// read and write deadlines are pushed every time a new message is received.
func IgnoreReceives(ws *websocket.Conn, timeout time.Duration) {
	defer runtime.HandleCrash()
	var data []byte
	for {
		resetTimeout(ws, timeout)
		if err := websocket.Message.Receive(ws, &data); err != nil {
			return
		}
	}
}

// handshake ensures the provided user protocol matches one of the allowed protocols. It returns
// no error if no protocol is specified.
func handshake(config *websocket.Config, req *http.Request, allowed []string) error {
	protocols := config.Protocol
	if len(protocols) == 0 {
		return nil
	}
	for _, protocol := range protocols {
		for _, allow := range allowed {
			if allow == protocol {
				config.Protocol = []string{protocol}
				return nil
			}
		}
	}
	return fmt.Errorf("requested protocol(s) are not supported: %v; supports %v", config.Protocol, allowed)
}

// Conn supports sending multiple binary channels over a websocket connection.
// Supports only the "channel.k8s.io" subprotocol.
type Conn struct {
	channels           []*websocketChannel
	codec              codecType
	ready              chan struct{}
	ws                 *websocket.Conn
	timeout            time.Duration
	supportedProtocols []string
	protocolPrefix     string
}

// NewConn creates a WebSocket connection that supports a set of channels. Channels begin each
// web socket message with a single byte indicating the channel number (0-N). 255 is reserved for
// future use. The channel types for each channel are passed as an array, supporting the different
// duplex modes. Read and Write refer to whether the channel can be used as a Reader or Writer.
//
// One of the base64 ("base64.channel.k8s.io") and raw ("channel.k8s.io") protocol is negotiated
// during initial handshake. Given protocolPrefixes (being DNS 1123 subdomains or empty string)
// are provided all combinations of those and base64+raw are nogatiated, e.g. "", "v1", "v2" leads
// to the protocols "base64.channel.k8s.io", "channel.k8s.io", "v1.base64.channel.k8s.io",
// "v1.channel.k8s.io", "v2.base64.channel.k8s.io", "v2.channel.k8s.io".
func NewConn(protocolPrefixes []string, channels ...ChannelType) *Conn {
	conn := &Conn{
		ready:              make(chan struct{}),
		channels:           make([]*websocketChannel, len(channels)),
		supportedProtocols: generateProtocols(protocolPrefixes, []string{channelWebSocketProtocol, base64ChannelWebSocketProtocol}),
	}
	for i := range conn.channels {
		switch channels[i] {
		case ReadChannel:
			conn.channels[i] = newWebsocketChannel(conn, byte(i), true, false)
		case WriteChannel:
			conn.channels[i] = newWebsocketChannel(conn, byte(i), false, true)
		case ReadWriteChannel:
			conn.channels[i] = newWebsocketChannel(conn, byte(i), true, true)
		case IgnoreChannel:
			conn.channels[i] = newWebsocketChannel(conn, byte(i), false, false)
		}
	}
	return conn
}

func generateProtocols(prefixes []string, suffixes []string) []string {
	if len(prefixes) == 0 {
		prefixes = []string{""}
	}
	protos := make([]string, 0, len(suffixes)*len(prefixes))
	for _, prefix := range prefixes {
		if prefix != "" {
			if errs := validation.IsDNS1123Subdomain(prefix); len(errs) > 0 {
				glog.Fatalf("websocket protocol prefix %q is no DNS 1123 subdomain: %v", prefix, errs)
			}
		}

		for _, suffix := range suffixes {
			proto := suffix
			if prefix != "" {
				proto = prefix + "." + suffix
			}
			protos = append(protos, proto)
		}
	}
	return protos
}

// SetIdleTimeout sets the interval for both reads and writes before timeout. If not specified,
// there is no timeout on the connection.
func (conn *Conn) SetIdleTimeout(duration time.Duration) {
	conn.timeout = duration
}

// Open the connection and create channels for reading and writing.
func (conn *Conn) Open(w http.ResponseWriter, req *http.Request) ([]io.ReadWriteCloser, error) {
	go func() {
		defer runtime.HandleCrash()
		defer conn.Close()
		websocket.Server{Handshake: conn.handshake, Handler: conn.handle}.ServeHTTP(w, req)
	}()
	<-conn.ready
	rwc := make([]io.ReadWriteCloser, len(conn.channels))
	for i := range conn.channels {
		rwc[i] = conn.channels[i]
	}
	return rwc, nil
}

func splitProtocol(proto string, b64, raw string) (string, string) {
	if proto == b64 || proto == raw {
		return "", proto
	}
	if strings.HasSuffix(proto, "."+b64) {
		prefixLen := len(proto) - len(b64) - 1
		return proto[0:prefixLen], b64
	}

	prefixLen := len(proto) - len(raw) - 1
	return proto[0:prefixLen], raw
}

func (conn *Conn) initialize(ws *websocket.Conn) {
	protocols := ws.Config().Protocol
	if len(protocols) == 0 {
		conn.codec = rawCodec
	} else {
		prefix, suffix := splitProtocol(protocols[0], base64ChannelWebSocketProtocol, channelWebSocketProtocol)
		conn.protocolPrefix = prefix
		switch suffix {
		case channelWebSocketProtocol:
			conn.codec = rawCodec
		case base64ChannelWebSocketProtocol:
			conn.codec = base64Codec
		}
	}
	conn.ws = ws
	close(conn.ready)
}

func (conn *Conn) handshake(config *websocket.Config, req *http.Request) error {
	return handshake(config, req, conn.supportedProtocols)
}

// ProtocolPrefix returns the prefix of the negotiated protocol.
func (conn *Conn) ProtocolPrefix() string {
	return conn.protocolPrefix
}

func (conn *Conn) resetTimeout() {
	if conn.timeout > 0 {
		conn.ws.SetDeadline(time.Now().Add(conn.timeout))
	}
}

// Close is only valid after Open has been called
func (conn *Conn) Close() error {
	<-conn.ready
	for _, s := range conn.channels {
		s.Close()
	}
	conn.ws.Close()
	return nil
}

// handle implements a websocket handler.
func (conn *Conn) handle(ws *websocket.Conn) {
	defer conn.Close()
	conn.initialize(ws)

	for {
		conn.resetTimeout()
		var data []byte
		if err := websocket.Message.Receive(ws, &data); err != nil {
			if err != io.EOF {
				glog.Errorf("Error on socket receive: %v", err)
			}
			break
		}
		if len(data) == 0 {
			continue
		}
		channel := data[0]
		if conn.codec == base64Codec {
			channel = channel - '0'
		}
		data = data[1:]
		if int(channel) >= len(conn.channels) {
			glog.V(6).Infof("Frame is targeted for a reader %d that is not valid, possible protocol error", channel)
			continue
		}
		if _, err := conn.channels[channel].DataFromSocket(data); err != nil {
			glog.Errorf("Unable to write frame to %d: %v\n%s", channel, err, string(data))
			continue
		}
	}
}

// write multiplexes the specified channel onto the websocket
func (conn *Conn) write(num byte, data []byte) (int, error) {
	conn.resetTimeout()
	switch conn.codec {
	case rawCodec:
		frame := make([]byte, len(data)+1)
		frame[0] = num
		copy(frame[1:], data)
		if err := websocket.Message.Send(conn.ws, frame); err != nil {
			return 0, err
		}
	case base64Codec:
		frame := string('0'+num) + base64.StdEncoding.EncodeToString(data)
		if err := websocket.Message.Send(conn.ws, frame); err != nil {
			return 0, err
		}
	}
	return len(data), nil
}

// websocketChannel represents a channel in a connection
type websocketChannel struct {
	conn *Conn
	num  byte
	r    io.Reader
	w    io.WriteCloser

	read, write bool
}

// newWebsocketChannel creates a pipe for writing to a websocket. Do not write to this pipe
// prior to the connection being opened. It may be no, half, or full duplex depending on
// read and write.
func newWebsocketChannel(conn *Conn, num byte, read, write bool) *websocketChannel {
	r, w := io.Pipe()
	return &websocketChannel{conn, num, r, w, read, write}
}

func (p *websocketChannel) Write(data []byte) (int, error) {
	if !p.write {
		return len(data), nil
	}
	return p.conn.write(p.num, data)
}

// DataFromSocket is invoked by the connection receiver to move data from the connection
// into a specific channel.
func (p *websocketChannel) DataFromSocket(data []byte) (int, error) {
	if !p.read {
		return len(data), nil
	}

	switch p.conn.codec {
	case rawCodec:
		return p.w.Write(data)
	case base64Codec:
		dst := make([]byte, len(data))
		n, err := base64.StdEncoding.Decode(dst, data)
		if err != nil {
			return 0, err
		}
		return p.w.Write(dst[:n])
	}
	return 0, nil
}

func (p *websocketChannel) Read(data []byte) (int, error) {
	if !p.read {
		return 0, io.EOF
	}
	return p.r.Read(data)
}

func (p *websocketChannel) Close() error {
	return p.w.Close()
}
