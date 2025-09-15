// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"log"
)

// debugTransport if set, will print packet types as they go over the
// wire. No message decoding is done, to minimize the impact on timing.
const debugTransport = false

const (
	gcm128CipherID = "aes128-gcm@openssh.com"
	gcm256CipherID = "aes256-gcm@openssh.com"
	aes128cbcID    = "aes128-cbc"
	tripledescbcID = "3des-cbc"
)

// packetConn represents a transport that implements packet based
// operations.
type packetConn interface {
	// Encrypt and send a packet of data to the remote peer.
	writePacket(packet []byte) error

	// Read a packet from the connection. The read is blocking,
	// i.e. if error is nil, then the returned byte slice is
	// always non-empty.
	readPacket() ([]byte, error)

	// Close closes the write-side of the connection.
	Close() error
}

// transport is the keyingTransport that implements the SSH packet
// protocol.
type transport struct {
	reader connectionState
	writer connectionState

	bufReader *bufio.Reader
	bufWriter *bufio.Writer
	rand      io.Reader
	isClient  bool
	io.Closer

	strictMode     bool
	initialKEXDone bool
}

// packetCipher represents a combination of SSH encryption/MAC
// protocol.  A single instance should be used for one direction only.
type packetCipher interface {
	// writeCipherPacket encrypts the packet and writes it to w. The
	// contents of the packet are generally scrambled.
	writeCipherPacket(seqnum uint32, w io.Writer, rand io.Reader, packet []byte) error

	// readCipherPacket reads and decrypts a packet of data. The
	// returned packet may be overwritten by future calls of
	// readPacket.
	readCipherPacket(seqnum uint32, r io.Reader) ([]byte, error)
}

// connectionState represents one side (read or write) of the
// connection. This is necessary because each direction has its own
// keys, and can even have its own algorithms
type connectionState struct {
	packetCipher
	seqNum           uint32
	dir              direction
	pendingKeyChange chan packetCipher
}

func (t *transport) setStrictMode() error {
	if t.reader.seqNum != 1 {
		return errors.New("ssh: sequence number != 1 when strict KEX mode requested")
	}
	t.strictMode = true
	return nil
}

func (t *transport) setInitialKEXDone() {
	t.initialKEXDone = true
}

// prepareKeyChange sets up key material for a keychange. The key changes in
// both directions are triggered by reading and writing a msgNewKey packet
// respectively.
func (t *transport) prepareKeyChange(algs *algorithms, kexResult *kexResult) error {
	ciph, err := newPacketCipher(t.reader.dir, algs.r, kexResult)
	if err != nil {
		return err
	}
	t.reader.pendingKeyChange <- ciph

	ciph, err = newPacketCipher(t.writer.dir, algs.w, kexResult)
	if err != nil {
		return err
	}
	t.writer.pendingKeyChange <- ciph

	return nil
}

func (t *transport) printPacket(p []byte, write bool) {
	if len(p) == 0 {
		return
	}
	who := "server"
	if t.isClient {
		who = "client"
	}
	what := "read"
	if write {
		what = "write"
	}

	log.Println(what, who, p[0])
}

// Read and decrypt next packet.
func (t *transport) readPacket() (p []byte, err error) {
	for {
		p, err = t.reader.readPacket(t.bufReader, t.strictMode)
		if err != nil {
			break
		}
		// in strict mode we pass through DEBUG and IGNORE packets only during the initial KEX
		if len(p) == 0 || (t.strictMode && !t.initialKEXDone) || (p[0] != msgIgnore && p[0] != msgDebug) {
			break
		}
	}
	if debugTransport {
		t.printPacket(p, false)
	}

	return p, err
}

func (s *connectionState) readPacket(r *bufio.Reader, strictMode bool) ([]byte, error) {
	packet, err := s.packetCipher.readCipherPacket(s.seqNum, r)
	s.seqNum++
	if err == nil && len(packet) == 0 {
		err = errors.New("ssh: zero length packet")
	}

	if len(packet) > 0 {
		switch packet[0] {
		case msgNewKeys:
			select {
			case cipher := <-s.pendingKeyChange:
				s.packetCipher = cipher
				if strictMode {
					s.seqNum = 0
				}
			default:
				return nil, errors.New("ssh: got bogus newkeys message")
			}

		case msgDisconnect:
			// Transform a disconnect message into an
			// error. Since this is lowest level at which
			// we interpret message types, doing it here
			// ensures that we don't have to handle it
			// elsewhere.
			var msg disconnectMsg
			if err := Unmarshal(packet, &msg); err != nil {
				return nil, err
			}
			return nil, &msg
		}
	}

	// The packet may point to an internal buffer, so copy the
	// packet out here.
	fresh := make([]byte, len(packet))
	copy(fresh, packet)

	return fresh, err
}

func (t *transport) writePacket(packet []byte) error {
	if debugTransport {
		t.printPacket(packet, true)
	}
	return t.writer.writePacket(t.bufWriter, t.rand, packet, t.strictMode)
}

func (s *connectionState) writePacket(w *bufio.Writer, rand io.Reader, packet []byte, strictMode bool) error {
	changeKeys := len(packet) > 0 && packet[0] == msgNewKeys

	err := s.packetCipher.writeCipherPacket(s.seqNum, w, rand, packet)
	if err != nil {
		return err
	}
	if err = w.Flush(); err != nil {
		return err
	}
	s.seqNum++
	if changeKeys {
		select {
		case cipher := <-s.pendingKeyChange:
			s.packetCipher = cipher
			if strictMode {
				s.seqNum = 0
			}
		default:
			panic("ssh: no key material for msgNewKeys")
		}
	}
	return err
}

func newTransport(rwc io.ReadWriteCloser, rand io.Reader, isClient bool) *transport {
	t := &transport{
		bufReader: bufio.NewReader(rwc),
		bufWriter: bufio.NewWriter(rwc),
		rand:      rand,
		reader: connectionState{
			packetCipher:     &streamPacketCipher{cipher: noneCipher{}},
			pendingKeyChange: make(chan packetCipher, 1),
		},
		writer: connectionState{
			packetCipher:     &streamPacketCipher{cipher: noneCipher{}},
			pendingKeyChange: make(chan packetCipher, 1),
		},
		Closer: rwc,
	}
	t.isClient = isClient

	if isClient {
		t.reader.dir = serverKeys
		t.writer.dir = clientKeys
	} else {
		t.reader.dir = clientKeys
		t.writer.dir = serverKeys
	}

	return t
}

type direction struct {
	ivTag     []byte
	keyTag    []byte
	macKeyTag []byte
}

var (
	serverKeys = direction{[]byte{'B'}, []byte{'D'}, []byte{'F'}}
	clientKeys = direction{[]byte{'A'}, []byte{'C'}, []byte{'E'}}
)

// setupKeys sets the cipher and MAC keys from kex.K, kex.H and sessionId, as
// described in RFC 4253, section 6.4. direction should either be serverKeys
// (to setup server->client keys) or clientKeys (for client->server keys).
func newPacketCipher(d direction, algs directionAlgorithms, kex *kexResult) (packetCipher, error) {
	cipherMode := cipherModes[algs.Cipher]

	iv := make([]byte, cipherMode.ivSize)
	key := make([]byte, cipherMode.keySize)

	generateKeyMaterial(iv, d.ivTag, kex)
	generateKeyMaterial(key, d.keyTag, kex)

	var macKey []byte
	if !aeadCiphers[algs.Cipher] {
		macMode := macModes[algs.MAC]
		macKey = make([]byte, macMode.keySize)
		generateKeyMaterial(macKey, d.macKeyTag, kex)
	}

	return cipherModes[algs.Cipher].create(key, iv, macKey, algs)
}

// generateKeyMaterial fills out with key material generated from tag, K, H
// and sessionId, as specified in RFC 4253, section 7.2.
func generateKeyMaterial(out, tag []byte, r *kexResult) {
	var digestsSoFar []byte

	h := r.Hash.New()
	for len(out) > 0 {
		h.Reset()
		h.Write(r.K)
		h.Write(r.H)

		if len(digestsSoFar) == 0 {
			h.Write(tag)
			h.Write(r.SessionID)
		} else {
			h.Write(digestsSoFar)
		}

		digest := h.Sum(nil)
		n := copy(out, digest)
		out = out[n:]
		if len(out) > 0 {
			digestsSoFar = append(digestsSoFar, digest...)
		}
	}
}

const packageVersion = "SSH-2.0-Go"

// Sends and receives a version line.  The versionLine string should
// be US ASCII, start with "SSH-2.0-", and should not include a
// newline. exchangeVersions returns the other side's version line.
func exchangeVersions(rw io.ReadWriter, versionLine []byte) (them []byte, err error) {
	// Contrary to the RFC, we do not ignore lines that don't
	// start with "SSH-2.0-" to make the library usable with
	// nonconforming servers.
	for _, c := range versionLine {
		// The spec disallows non US-ASCII chars, and
		// specifically forbids null chars.
		if c < 32 {
			return nil, errors.New("ssh: junk character in version line")
		}
	}
	if _, err = rw.Write(append(versionLine, '\r', '\n')); err != nil {
		return
	}

	them, err = readVersion(rw)
	return them, err
}

// maxVersionStringBytes is the maximum number of bytes that we'll
// accept as a version string. RFC 4253 section 4.2 limits this at 255
// chars
const maxVersionStringBytes = 255

// Read version string as specified by RFC 4253, section 4.2.
func readVersion(r io.Reader) ([]byte, error) {
	versionString := make([]byte, 0, 64)
	var ok bool
	var buf [1]byte

	for length := 0; length < maxVersionStringBytes; length++ {
		_, err := io.ReadFull(r, buf[:])
		if err != nil {
			return nil, err
		}
		// The RFC says that the version should be terminated with \r\n
		// but several SSH servers actually only send a \n.
		if buf[0] == '\n' {
			if !bytes.HasPrefix(versionString, []byte("SSH-")) {
				// RFC 4253 says we need to ignore all version string lines
				// except the one containing the SSH version (provided that
				// all the lines do not exceed 255 bytes in total).
				versionString = versionString[:0]
				continue
			}
			ok = true
			break
		}

		// non ASCII chars are disallowed, but we are lenient,
		// since Go doesn't use null-terminated strings.

		// The RFC allows a comment after a space, however,
		// all of it (version and comments) goes into the
		// session hash.
		versionString = append(versionString, buf[0])
	}

	if !ok {
		return nil, errors.New("ssh: overflow reading version string")
	}

	// There might be a '\r' on the end which we should remove.
	if len(versionString) > 0 && versionString[len(versionString)-1] == '\r' {
		versionString = versionString[:len(versionString)-1]
	}
	return versionString, nil
}
