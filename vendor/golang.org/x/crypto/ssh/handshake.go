// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
)

// debugHandshake, if set, prints messages sent and received.  Key
// exchange messages are printed as if DH were used, so the debug
// messages are wrong when using ECDH.
const debugHandshake = false

// keyingTransport is a packet based transport that supports key
// changes. It need not be thread-safe. It should pass through
// msgNewKeys in both directions.
type keyingTransport interface {
	packetConn

	// prepareKeyChange sets up a key change. The key change for a
	// direction will be effected if a msgNewKeys message is sent
	// or received.
	prepareKeyChange(*algorithms, *kexResult) error

	// getSessionID returns the session ID. prepareKeyChange must
	// have been called once.
	getSessionID() []byte
}

// rekeyingTransport is the interface of handshakeTransport that we
// (internally) expose to ClientConn and ServerConn.
type rekeyingTransport interface {
	packetConn

	// requestKeyChange asks the remote side to change keys. All
	// writes are blocked until the key change succeeds, which is
	// signaled by reading a msgNewKeys.
	requestKeyChange() error

	// getSessionID returns the session ID. This is only valid
	// after the first key change has completed.
	getSessionID() []byte
}

// handshakeTransport implements rekeying on top of a keyingTransport
// and offers a thread-safe writePacket() interface.
type handshakeTransport struct {
	conn   keyingTransport
	config *Config

	serverVersion []byte
	clientVersion []byte

	// hostKeys is non-empty if we are the server. In that case,
	// it contains all host keys that can be used to sign the
	// connection.
	hostKeys []Signer

	// hostKeyAlgorithms is non-empty if we are the client. In that case,
	// we accept these key types from the server as host key.
	hostKeyAlgorithms []string

	// On read error, incoming is closed, and readError is set.
	incoming  chan []byte
	readError error

	// data for host key checking
	hostKeyCallback func(hostname string, remote net.Addr, key PublicKey) error
	dialAddress     string
	remoteAddr      net.Addr

	readSinceKex uint64

	// Protects the writing side of the connection
	mu              sync.Mutex
	cond            *sync.Cond
	sentInitPacket  []byte
	sentInitMsg     *kexInitMsg
	writtenSinceKex uint64
	writeError      error
}

func newHandshakeTransport(conn keyingTransport, config *Config, clientVersion, serverVersion []byte) *handshakeTransport {
	t := &handshakeTransport{
		conn:          conn,
		serverVersion: serverVersion,
		clientVersion: clientVersion,
		incoming:      make(chan []byte, 16),
		config:        config,
	}
	t.cond = sync.NewCond(&t.mu)
	return t
}

func newClientTransport(conn keyingTransport, clientVersion, serverVersion []byte, config *ClientConfig, dialAddr string, addr net.Addr) *handshakeTransport {
	t := newHandshakeTransport(conn, &config.Config, clientVersion, serverVersion)
	t.dialAddress = dialAddr
	t.remoteAddr = addr
	t.hostKeyCallback = config.HostKeyCallback
	if config.HostKeyAlgorithms != nil {
		t.hostKeyAlgorithms = config.HostKeyAlgorithms
	} else {
		t.hostKeyAlgorithms = supportedHostKeyAlgos
	}
	go t.readLoop()
	return t
}

func newServerTransport(conn keyingTransport, clientVersion, serverVersion []byte, config *ServerConfig) *handshakeTransport {
	t := newHandshakeTransport(conn, &config.Config, clientVersion, serverVersion)
	t.hostKeys = config.hostKeys
	go t.readLoop()
	return t
}

func (t *handshakeTransport) getSessionID() []byte {
	return t.conn.getSessionID()
}

func (t *handshakeTransport) id() string {
	if len(t.hostKeys) > 0 {
		return "server"
	}
	return "client"
}

func (t *handshakeTransport) readPacket() ([]byte, error) {
	p, ok := <-t.incoming
	if !ok {
		return nil, t.readError
	}
	return p, nil
}

func (t *handshakeTransport) readLoop() {
	for {
		p, err := t.readOnePacket()
		if err != nil {
			t.readError = err
			close(t.incoming)
			break
		}
		if p[0] == msgIgnore || p[0] == msgDebug {
			continue
		}
		t.incoming <- p
	}

	// If we can't read, declare the writing part dead too.
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.writeError == nil {
		t.writeError = t.readError
	}
	t.cond.Broadcast()
}

func (t *handshakeTransport) readOnePacket() ([]byte, error) {
	if t.readSinceKex > t.config.RekeyThreshold {
		if err := t.requestKeyChange(); err != nil {
			return nil, err
		}
	}

	p, err := t.conn.readPacket()
	if err != nil {
		return nil, err
	}

	t.readSinceKex += uint64(len(p))
	if debugHandshake {
		msg, err := decode(p)
		log.Printf("%s got %T %v (%v)", t.id(), msg, msg, err)
	}
	if p[0] != msgKexInit {
		return p, nil
	}
	err = t.enterKeyExchange(p)

	t.mu.Lock()
	if err != nil {
		// drop connection
		t.conn.Close()
		t.writeError = err
	}

	if debugHandshake {
		log.Printf("%s exited key exchange, err %v", t.id(), err)
	}

	// Unblock writers.
	t.sentInitMsg = nil
	t.sentInitPacket = nil
	t.cond.Broadcast()
	t.writtenSinceKex = 0
	t.mu.Unlock()

	if err != nil {
		return nil, err
	}

	t.readSinceKex = 0
	return []byte{msgNewKeys}, nil
}

// sendKexInit sends a key change message, and returns the message
// that was sent. After initiating the key change, all writes will be
// blocked until the change is done, and a failed key change will
// close the underlying transport. This function is safe for
// concurrent use by multiple goroutines.
func (t *handshakeTransport) sendKexInit() (*kexInitMsg, []byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.sendKexInitLocked()
}

func (t *handshakeTransport) requestKeyChange() error {
	_, _, err := t.sendKexInit()
	return err
}

// sendKexInitLocked sends a key change message. t.mu must be locked
// while this happens.
func (t *handshakeTransport) sendKexInitLocked() (*kexInitMsg, []byte, error) {
	// kexInits may be sent either in response to the other side,
	// or because our side wants to initiate a key change, so we
	// may have already sent a kexInit. In that case, don't send a
	// second kexInit.
	if t.sentInitMsg != nil {
		return t.sentInitMsg, t.sentInitPacket, nil
	}
	msg := &kexInitMsg{
		KexAlgos:                t.config.KeyExchanges,
		CiphersClientServer:     t.config.Ciphers,
		CiphersServerClient:     t.config.Ciphers,
		MACsClientServer:        t.config.MACs,
		MACsServerClient:        t.config.MACs,
		CompressionClientServer: supportedCompressions,
		CompressionServerClient: supportedCompressions,
	}
	io.ReadFull(rand.Reader, msg.Cookie[:])

	if len(t.hostKeys) > 0 {
		for _, k := range t.hostKeys {
			msg.ServerHostKeyAlgos = append(
				msg.ServerHostKeyAlgos, k.PublicKey().Type())
		}
	} else {
		msg.ServerHostKeyAlgos = t.hostKeyAlgorithms
	}
	packet := Marshal(msg)

	// writePacket destroys the contents, so save a copy.
	packetCopy := make([]byte, len(packet))
	copy(packetCopy, packet)

	if err := t.conn.writePacket(packetCopy); err != nil {
		return nil, nil, err
	}

	t.sentInitMsg = msg
	t.sentInitPacket = packet
	return msg, packet, nil
}

func (t *handshakeTransport) writePacket(p []byte) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.writtenSinceKex > t.config.RekeyThreshold {
		t.sendKexInitLocked()
	}
	for t.sentInitMsg != nil && t.writeError == nil {
		t.cond.Wait()
	}
	if t.writeError != nil {
		return t.writeError
	}
	t.writtenSinceKex += uint64(len(p))

	switch p[0] {
	case msgKexInit:
		return errors.New("ssh: only handshakeTransport can send kexInit")
	case msgNewKeys:
		return errors.New("ssh: only handshakeTransport can send newKeys")
	default:
		return t.conn.writePacket(p)
	}
}

func (t *handshakeTransport) Close() error {
	return t.conn.Close()
}

// enterKeyExchange runs the key exchange.
func (t *handshakeTransport) enterKeyExchange(otherInitPacket []byte) error {
	if debugHandshake {
		log.Printf("%s entered key exchange", t.id())
	}
	myInit, myInitPacket, err := t.sendKexInit()
	if err != nil {
		return err
	}

	otherInit := &kexInitMsg{}
	if err := Unmarshal(otherInitPacket, otherInit); err != nil {
		return err
	}

	magics := handshakeMagics{
		clientVersion: t.clientVersion,
		serverVersion: t.serverVersion,
		clientKexInit: otherInitPacket,
		serverKexInit: myInitPacket,
	}

	clientInit := otherInit
	serverInit := myInit
	if len(t.hostKeys) == 0 {
		clientInit = myInit
		serverInit = otherInit

		magics.clientKexInit = myInitPacket
		magics.serverKexInit = otherInitPacket
	}

	algs, err := findAgreedAlgorithms(clientInit, serverInit)
	if err != nil {
		return err
	}

	// We don't send FirstKexFollows, but we handle receiving it.
	if otherInit.FirstKexFollows && algs.kex != otherInit.KexAlgos[0] {
		// other side sent a kex message for the wrong algorithm,
		// which we have to ignore.
		if _, err := t.conn.readPacket(); err != nil {
			return err
		}
	}

	kex, ok := kexAlgoMap[algs.kex]
	if !ok {
		return fmt.Errorf("ssh: unexpected key exchange algorithm %v", algs.kex)
	}

	var result *kexResult
	if len(t.hostKeys) > 0 {
		result, err = t.server(kex, algs, &magics)
	} else {
		result, err = t.client(kex, algs, &magics)
	}

	if err != nil {
		return err
	}

	t.conn.prepareKeyChange(algs, result)
	if err = t.conn.writePacket([]byte{msgNewKeys}); err != nil {
		return err
	}
	if packet, err := t.conn.readPacket(); err != nil {
		return err
	} else if packet[0] != msgNewKeys {
		return unexpectedMessageError(msgNewKeys, packet[0])
	}
	return nil
}

func (t *handshakeTransport) server(kex kexAlgorithm, algs *algorithms, magics *handshakeMagics) (*kexResult, error) {
	var hostKey Signer
	for _, k := range t.hostKeys {
		if algs.hostKey == k.PublicKey().Type() {
			hostKey = k
		}
	}

	r, err := kex.Server(t.conn, t.config.Rand, magics, hostKey)
	return r, err
}

func (t *handshakeTransport) client(kex kexAlgorithm, algs *algorithms, magics *handshakeMagics) (*kexResult, error) {
	result, err := kex.Client(t.conn, t.config.Rand, magics)
	if err != nil {
		return nil, err
	}

	hostKey, err := ParsePublicKey(result.HostKey)
	if err != nil {
		return nil, err
	}

	if err := verifyHostKeySignature(hostKey, result); err != nil {
		return nil, err
	}

	if t.hostKeyCallback != nil {
		err = t.hostKeyCallback(t.dialAddress, t.remoteAddr, hostKey)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}
