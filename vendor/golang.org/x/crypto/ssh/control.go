// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"golang.org/x/crypto/cryptobyte"
)

const (
	muxProtocolVersion = 4

	muxMsgHello = 0x00000001
	muxCProxy   = 0x1000000f
	muxSProxy   = 0x8000000f
)

const controlProxyRequestID = 0

// handshakeControlProxy attempts to establish a transport connection with an
// OpenSSH ControlMaster socket in proxy mode. For details see:
// https://github.com/openssh/openssh-portable/blob/master/PROTOCOL.mux
func handshakeControlProxy(rw io.ReadWriteCloser) (connTransport, error) {
	if err := controlProxyWritePacket(rw, func(b *cryptobyte.Builder) {
		b.AddUint32(muxMsgHello)
		b.AddUint32(muxProtocolVersion)
	}); err != nil {
		return nil, fmt.Errorf("mux hello write failed: %w", err)
	}
	if err := controlProxyWritePacket(rw, func(b *cryptobyte.Builder) {
		b.AddUint32(muxCProxy)
		b.AddUint32(controlProxyRequestID)
	}); err != nil {
		return nil, fmt.Errorf("mux client proxy write failed: %w", err)
	}

	messageType, body, err := controlProxyReadMessage(rw)
	if err != nil {
		return nil, fmt.Errorf("mux hello read failed: %w", err)
	}
	if messageType != muxMsgHello {
		return nil, fmt.Errorf("expected hello response, got %v", messageType)
	}
	var v uint32
	if !body.ReadUint32(&v) {
		return nil, errors.New("EOF reading mux protocol version")
	}
	if v != muxProtocolVersion {
		return nil, fmt.Errorf("mux server has unsupported version %v", v)
	}
	messageType, body, err = controlProxyReadMessage(rw)
	if err != nil {
		return nil, fmt.Errorf("mux server proxy read failed: %w", err)
	}
	if messageType != muxSProxy {
		return nil, fmt.Errorf("expected server proxy response, got %v", messageType)
	}
	var reqID uint32
	if !body.ReadUint32(&reqID) {
		return nil, errors.New("EOF reading request id")
	}
	if reqID != controlProxyRequestID {
		return nil, fmt.Errorf("expected request id %v, got %v", controlProxyRequestID, reqID)
	}
	return &controlProxyTransport{rw}, nil
}

// controlProxyTransport implements the connTransport interface for
// ControlMaster connections. Each controlMessage has zero length padding and
// no MAC.
type controlProxyTransport struct {
	rw io.ReadWriteCloser
}

func (p *controlProxyTransport) Close() error {
	return p.rw.Close()
}

func (p *controlProxyTransport) writePacket(controlMessage []byte) error {
	return controlProxyWritePacket(p.rw, func(b *cryptobyte.Builder) {
		b.AddUint8(0) // Padding length.
		b.AddBytes(controlMessage)
	})
}

func (p *controlProxyTransport) readPacket() ([]byte, error) {
	buf, err := controlProxyReadPacket(p.rw)
	if err != nil {
		return nil, fmt.Errorf("ssh: error reading control message: %w", err)
	}
	// Discard the padding length.
	if len(buf) < 1 {
		return nil, errors.New("ssh: EOF reading padding length")
	}
	if buf[0] != 0 {
		return nil, errors.New("ssh: unexpected non-zero padding in control message")
	}
	return buf[1:], nil
}

func (p *controlProxyTransport) getAlgorithms() NegotiatedAlgorithms {
	return NegotiatedAlgorithms{}
}

func (p *controlProxyTransport) getSessionID() []byte {
	return nil
}

func (p *controlProxyTransport) waitSession() error {
	return nil
}

func controlProxyWritePacket(w io.Writer, f cryptobyte.BuilderContinuation) error {
	var buf []byte
	b := cryptobyte.NewBuilder(buf)
	b.AddUint32LengthPrefixed(f)
	out, err := b.Bytes()
	if err != nil {
		return err
	}
	_, err = w.Write(out)
	return err
}

func controlProxyReadPacket(r io.Reader) (cryptobyte.String, error) {
	var l uint32
	if err := binary.Read(r, binary.BigEndian, &l); err != nil {
		return nil, err
	}
	if l > maxPacket {
		return nil, fmt.Errorf("message length %v exceeds maximum %v", l, maxPacket)
	}
	buf := make([]byte, l)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func controlProxyReadMessage(r io.Reader) (messageType uint32, body cryptobyte.String, err error) {
	body, err = controlProxyReadPacket(r)
	if err != nil {
		return 0, nil, fmt.Errorf("error reading message body: %w", err)
	}
	if !body.ReadUint32(&messageType) {
		return 0, nil, errors.New("EOF reading message type")
	}
	return messageType, body, nil
}
