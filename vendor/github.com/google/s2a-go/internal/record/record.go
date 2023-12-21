/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package record implements the TLS 1.3 record protocol used by the S2A
// transport credentials.
package record

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"net"
	"sync"

	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
	"github.com/google/s2a-go/internal/record/internal/halfconn"
	"github.com/google/s2a-go/internal/tokenmanager"
	"google.golang.org/grpc/grpclog"
)

// recordType is the `ContentType` as described in
// https://tools.ietf.org/html/rfc8446#section-5.1.
type recordType byte

const (
	alert           recordType = 21
	handshake       recordType = 22
	applicationData recordType = 23
)

// keyUpdateRequest is the `KeyUpdateRequest` as described in
// https://tools.ietf.org/html/rfc8446#section-4.6.3.
type keyUpdateRequest byte

const (
	updateNotRequested keyUpdateRequest = 0
	updateRequested    keyUpdateRequest = 1
)

// alertDescription is the `AlertDescription` as described in
// https://tools.ietf.org/html/rfc8446#section-6.
type alertDescription byte

const (
	closeNotify alertDescription = 0
)

// sessionTicketState is used to determine whether session tickets have not yet
// been received, are in the process of being received, or have finished
// receiving.
type sessionTicketState byte

const (
	ticketsNotYetReceived sessionTicketState = 0
	receivingTickets      sessionTicketState = 1
	notReceivingTickets   sessionTicketState = 2
)

const (
	// The TLS 1.3-specific constants below (tlsRecordMaxPlaintextSize,
	// tlsRecordHeaderSize, tlsRecordTypeSize) were taken from
	// https://tools.ietf.org/html/rfc8446#section-5.1.

	// tlsRecordMaxPlaintextSize is the maximum size in bytes of the plaintext
	// in a single TLS 1.3 record.
	tlsRecordMaxPlaintextSize = 16384 // 2^14
	// tlsRecordTypeSize is the size in bytes of the TLS 1.3 record type.
	tlsRecordTypeSize = 1
	// tlsTagSize is the size in bytes of the tag of the following three
	// ciphersuites: AES-128-GCM-SHA256, AES-256-GCM-SHA384,
	// CHACHA20-POLY1305-SHA256.
	tlsTagSize = 16
	// tlsRecordMaxPayloadSize is the maximum size in bytes of the payload in a
	// single TLS 1.3 record. This is the maximum size of the plaintext plus the
	// record type byte and 16 bytes of the tag.
	tlsRecordMaxPayloadSize = tlsRecordMaxPlaintextSize + tlsRecordTypeSize + tlsTagSize
	// tlsRecordHeaderTypeSize is the size in bytes of the TLS 1.3 record
	// header type.
	tlsRecordHeaderTypeSize = 1
	// tlsRecordHeaderLegacyRecordVersionSize is the size in bytes of the TLS
	// 1.3 record header legacy record version.
	tlsRecordHeaderLegacyRecordVersionSize = 2
	// tlsRecordHeaderPayloadLengthSize is the size in bytes of the TLS 1.3
	// record header payload length.
	tlsRecordHeaderPayloadLengthSize = 2
	// tlsRecordHeaderSize is the size in bytes of the TLS 1.3 record header.
	tlsRecordHeaderSize = tlsRecordHeaderTypeSize + tlsRecordHeaderLegacyRecordVersionSize + tlsRecordHeaderPayloadLengthSize
	// tlsRecordMaxSize
	tlsRecordMaxSize = tlsRecordMaxPayloadSize + tlsRecordHeaderSize
	// tlsApplicationData is the application data type of the TLS 1.3 record
	// header.
	tlsApplicationData = 23
	// tlsLegacyRecordVersion is the legacy record version of the TLS record.
	tlsLegacyRecordVersion = 3
	// tlsAlertSize is the size in bytes of an alert of TLS 1.3.
	tlsAlertSize = 2
)

const (
	// These are TLS 1.3 handshake-specific constants.

	// tlsHandshakeNewSessionTicketType is the prefix of a handshake new session
	// ticket message of TLS 1.3.
	tlsHandshakeNewSessionTicketType = 4
	// tlsHandshakeKeyUpdateType is the prefix of a handshake key update message
	// of TLS 1.3.
	tlsHandshakeKeyUpdateType = 24
	// tlsHandshakeMsgTypeSize is the size in bytes of the TLS 1.3 handshake
	// message type field.
	tlsHandshakeMsgTypeSize = 1
	// tlsHandshakeLengthSize is the size in bytes of the TLS 1.3 handshake
	// message length field.
	tlsHandshakeLengthSize = 3
	// tlsHandshakeKeyUpdateMsgSize is the size in bytes of the TLS 1.3
	// handshake key update message.
	tlsHandshakeKeyUpdateMsgSize = 1
	// tlsHandshakePrefixSize is the size in bytes of the prefix of the TLS 1.3
	// handshake message.
	tlsHandshakePrefixSize = 4
	// tlsMaxSessionTicketSize is the maximum size of a NewSessionTicket message
	// in TLS 1.3. This is the sum of the max sizes of all the fields in the
	// NewSessionTicket struct specified in
	// https://tools.ietf.org/html/rfc8446#section-4.6.1.
	tlsMaxSessionTicketSize = 131338
)

const (
	// outBufMaxRecords is the maximum number of records that can fit in the
	// ourRecordsBuf buffer.
	outBufMaxRecords = 16
	// outBufMaxSize is the maximum size (in bytes) of the outRecordsBuf buffer.
	outBufMaxSize = outBufMaxRecords * tlsRecordMaxSize
	// maxAllowedTickets is the maximum number of session tickets that are
	// allowed. The number of tickets are limited to ensure that the size of the
	// ticket queue does not grow indefinitely. S2A also keeps a limit on the
	// number of tickets that it caches.
	maxAllowedTickets = 5
)

// preConstructedKeyUpdateMsg holds the key update message. This is needed as an
// optimization so that the same message does not need to be constructed every
// time a key update message is sent.
var preConstructedKeyUpdateMsg = buildKeyUpdateRequest()

// conn represents a secured TLS connection. It implements the net.Conn
// interface.
type conn struct {
	net.Conn
	// inConn is the half connection responsible for decrypting incoming bytes.
	inConn *halfconn.S2AHalfConnection
	// outConn is the half connection responsible for encrypting outgoing bytes.
	outConn *halfconn.S2AHalfConnection
	// pendingApplicationData holds data that has been read from the connection
	// and decrypted, but has not yet been returned by Read.
	pendingApplicationData []byte
	// unusedBuf holds data read from the network that has not yet been
	// decrypted. This data might not consist of a complete record. It may
	// consist of several records, the last of which could be incomplete.
	unusedBuf []byte
	// outRecordsBuf is a buffer used to store outgoing TLS records before
	// they are written to the network.
	outRecordsBuf []byte
	// nextRecord stores the next record info in the unusedBuf buffer.
	nextRecord []byte
	// overheadSize is the overhead size in bytes of each TLS 1.3 record, which
	// is computed as overheadSize = header size + record type byte + tag size.
	// Note that there is no padding by zeros in the overhead calculation.
	overheadSize int
	// readMutex guards against concurrent calls to Read. This is required since
	// Close may be called during a Read.
	readMutex sync.Mutex
	// writeMutex guards against concurrent calls to Write. This is required
	// since Close may be called during a Write, and also because a key update
	// message may be written during a Read.
	writeMutex sync.Mutex
	// handshakeBuf holds handshake messages while they are being processed.
	handshakeBuf []byte
	// ticketState is the current processing state of the session tickets.
	ticketState sessionTicketState
	// sessionTickets holds the completed session tickets until they are sent to
	// the handshaker service for processing.
	sessionTickets [][]byte
	// ticketSender sends session tickets to the S2A handshaker service.
	ticketSender s2aTicketSender
	// callComplete is a channel that blocks closing the record protocol until a
	// pending call to the S2A completes.
	callComplete chan bool
}

// ConnParameters holds the parameters used for creating a new conn object.
type ConnParameters struct {
	// NetConn is the TCP connection to the peer. This parameter is required.
	NetConn net.Conn
	// Ciphersuite is the TLS ciphersuite negotiated by the S2A handshaker
	// service. This parameter is required.
	Ciphersuite commonpb.Ciphersuite
	// TLSVersion is the TLS version number negotiated by the S2A handshaker
	// service. This parameter is required.
	TLSVersion commonpb.TLSVersion
	// InTrafficSecret is the traffic secret used to derive the session key for
	// the inbound direction. This parameter is required.
	InTrafficSecret []byte
	// OutTrafficSecret is the traffic secret used to derive the session key
	// for the outbound direction. This parameter is required.
	OutTrafficSecret []byte
	// UnusedBuf is the data read from the network that has not yet been
	// decrypted. This parameter is optional. If not provided, then no
	// application data was sent in the same flight of messages as the final
	// handshake message.
	UnusedBuf []byte
	// InSequence is the sequence number of the next, incoming, TLS record.
	// This parameter is required.
	InSequence uint64
	// OutSequence is the sequence number of the next, outgoing, TLS record.
	// This parameter is required.
	OutSequence uint64
	// HSAddr stores the address of the S2A handshaker service. This parameter
	// is optional. If not provided, then TLS resumption is disabled.
	HSAddr string
	// ConnectionId is the connection identifier that was created and sent by
	// S2A at the end of a handshake.
	ConnectionID uint64
	// LocalIdentity is the local identity that was used by S2A during session
	// setup and included in the session result.
	LocalIdentity *commonpb.Identity
	// EnsureProcessSessionTickets allows users to wait and ensure that all
	// available session tickets are sent to S2A before a process completes.
	EnsureProcessSessionTickets *sync.WaitGroup
}

// NewConn creates a TLS record protocol that wraps the TCP connection.
func NewConn(o *ConnParameters) (net.Conn, error) {
	if o == nil {
		return nil, errors.New("conn options must not be nil")
	}
	if o.TLSVersion != commonpb.TLSVersion_TLS1_3 {
		return nil, errors.New("TLS version must be TLS 1.3")
	}

	inConn, err := halfconn.New(o.Ciphersuite, o.InTrafficSecret, o.InSequence)
	if err != nil {
		return nil, fmt.Errorf("failed to create inbound half connection: %v", err)
	}
	outConn, err := halfconn.New(o.Ciphersuite, o.OutTrafficSecret, o.OutSequence)
	if err != nil {
		return nil, fmt.Errorf("failed to create outbound half connection: %v", err)
	}

	// The tag size for the in/out connections should be the same.
	overheadSize := tlsRecordHeaderSize + tlsRecordTypeSize + inConn.TagSize()
	var unusedBuf []byte
	if o.UnusedBuf == nil {
		// We pre-allocate unusedBuf to be of size
		// 2*tlsRecordMaxSize-1 during initialization. We only read from the
		// network into unusedBuf when unusedBuf does not contain a complete
		// record and the incomplete record is at most tlsRecordMaxSize-1
		// (bytes). And we read at most tlsRecordMaxSize bytes of data from the
		// network into unusedBuf at one time. Therefore, 2*tlsRecordMaxSize-1
		// is large enough to buffer data read from the network.
		unusedBuf = make([]byte, 0, 2*tlsRecordMaxSize-1)
	} else {
		unusedBuf = make([]byte, len(o.UnusedBuf))
		copy(unusedBuf, o.UnusedBuf)
	}

	tokenManager, err := tokenmanager.NewSingleTokenAccessTokenManager()
	if err != nil {
		grpclog.Infof("failed to create single token access token manager: %v", err)
	}

	s2aConn := &conn{
		Conn:          o.NetConn,
		inConn:        inConn,
		outConn:       outConn,
		unusedBuf:     unusedBuf,
		outRecordsBuf: make([]byte, tlsRecordMaxSize),
		nextRecord:    unusedBuf,
		overheadSize:  overheadSize,
		ticketState:   ticketsNotYetReceived,
		// Pre-allocate the buffer for one session ticket message and the max
		// plaintext size. This is the largest size that handshakeBuf will need
		// to hold. The largest incomplete handshake message is the
		// [handshake header size] + [max session ticket size] - 1.
		// Then, tlsRecordMaxPlaintextSize is the maximum size that will be
		// appended to the handshakeBuf before the handshake message is
		// completed. Therefore, the buffer size below should be large enough to
		// buffer any handshake messages.
		handshakeBuf: make([]byte, 0, tlsHandshakePrefixSize+tlsMaxSessionTicketSize+tlsRecordMaxPlaintextSize-1),
		ticketSender: &ticketSender{
			hsAddr:                      o.HSAddr,
			connectionID:                o.ConnectionID,
			localIdentity:               o.LocalIdentity,
			tokenManager:                tokenManager,
			ensureProcessSessionTickets: o.EnsureProcessSessionTickets,
		},
		callComplete: make(chan bool),
	}
	return s2aConn, nil
}

// Read reads and decrypts a TLS 1.3 record from the underlying connection, and
// copies any application data received from the peer into b. If the size of the
// payload is greater than len(b), Read retains the remaining bytes in an
// internal buffer, and subsequent calls to Read will read from this buffer
// until it is exhausted. At most 1 TLS record worth of application data is
// written to b for each call to Read.
//
// Note that for the user to efficiently call this method, the user should
// ensure that the buffer b is allocated such that the buffer does not have any
// unused segments. This can be done by calling Read via io.ReadFull, which
// continually calls Read until the specified buffer has been filled. Also note
// that the user should close the connection via Close() if an error is thrown
// by a call to Read.
func (p *conn) Read(b []byte) (n int, err error) {
	p.readMutex.Lock()
	defer p.readMutex.Unlock()
	// Check if p.pendingApplication data has leftover application data from
	// the previous call to Read.
	if len(p.pendingApplicationData) == 0 {
		// Read a full record from the wire.
		record, err := p.readFullRecord()
		if err != nil {
			return 0, err
		}
		// Now we have a complete record, so split the header and validate it
		// The TLS record is split into 2 pieces: the record header and the
		// payload. The payload has the following form:
		// [payload] = [ciphertext of application data]
		//           + [ciphertext of record type byte]
		//           + [(optionally) ciphertext of padding by zeros]
		//           + [tag]
		header, payload, err := splitAndValidateHeader(record)
		if err != nil {
			return 0, err
		}
		// Decrypt the ciphertext.
		p.pendingApplicationData, err = p.inConn.Decrypt(payload[:0], payload, header)
		if err != nil {
			return 0, err
		}
		// Remove the padding by zeros and the record type byte from the
		// p.pendingApplicationData buffer.
		msgType, err := p.stripPaddingAndType()
		if err != nil {
			return 0, err
		}
		// Check that the length of the plaintext after stripping the padding
		// and record type byte is under the maximum plaintext size.
		if len(p.pendingApplicationData) > tlsRecordMaxPlaintextSize {
			return 0, errors.New("plaintext size larger than maximum")
		}
		// The expected message types are application data, alert, and
		// handshake. For application data, the bytes are directly copied into
		// b. For an alert, the type of the alert is checked and the connection
		// is closed on a close notify alert. For a handshake message, the
		// handshake message type is checked. The handshake message type can be
		// a key update type, for which we advance the traffic secret, and a
		// new session ticket type, for which we send the received ticket to S2A
		// for processing.
		switch msgType {
		case applicationData:
			if len(p.handshakeBuf) > 0 {
				return 0, errors.New("application data received while processing fragmented handshake messages")
			}
			if p.ticketState == receivingTickets {
				p.ticketState = notReceivingTickets
				grpclog.Infof("Sending session tickets to S2A.")
				p.ticketSender.sendTicketsToS2A(p.sessionTickets, p.callComplete)
			}
		case alert:
			return 0, p.handleAlertMessage()
		case handshake:
			if err = p.handleHandshakeMessage(); err != nil {
				return 0, err
			}
			return 0, nil
		default:
			return 0, errors.New("unknown record type")
		}
	}
	// Write as much application data as possible to b, the output buffer.
	n = copy(b, p.pendingApplicationData)
	p.pendingApplicationData = p.pendingApplicationData[n:]
	return n, nil
}

// Write divides b into segments of size tlsRecordMaxPlaintextSize, builds a
// TLS 1.3 record (of type "application data") from each segment, and sends
// the record to the peer. It returns the number of plaintext bytes that were
// successfully sent to the peer.
func (p *conn) Write(b []byte) (n int, err error) {
	p.writeMutex.Lock()
	defer p.writeMutex.Unlock()
	return p.writeTLSRecord(b, tlsApplicationData)
}

// writeTLSRecord divides b into segments of size maxPlaintextBytesPerRecord,
// builds a TLS 1.3 record (of type recordType) from each segment, and sends
// the record to the peer. It returns the number of plaintext bytes that were
// successfully sent to the peer.
func (p *conn) writeTLSRecord(b []byte, recordType byte) (n int, err error) {
	// Create a record of only header, record type, and tag if given empty
	// byte array.
	if len(b) == 0 {
		recordEndIndex, _, err := p.buildRecord(b, recordType, 0)
		if err != nil {
			return 0, err
		}

		// Write the bytes stored in outRecordsBuf to p.Conn. Since we return
		// the number of plaintext bytes written without overhead, we will
		// always return 0 while p.Conn.Write returns the entire record length.
		_, err = p.Conn.Write(p.outRecordsBuf[:recordEndIndex])
		return 0, err
	}

	numRecords := int(math.Ceil(float64(len(b)) / float64(tlsRecordMaxPlaintextSize)))
	totalRecordsSize := len(b) + numRecords*p.overheadSize
	partialBSize := len(b)
	if totalRecordsSize > outBufMaxSize {
		totalRecordsSize = outBufMaxSize
		partialBSize = outBufMaxRecords * tlsRecordMaxPlaintextSize
	}
	if len(p.outRecordsBuf) < totalRecordsSize {
		p.outRecordsBuf = make([]byte, totalRecordsSize)
	}
	for bStart := 0; bStart < len(b); bStart += partialBSize {
		bEnd := bStart + partialBSize
		if bEnd > len(b) {
			bEnd = len(b)
		}
		partialB := b[bStart:bEnd]
		recordEndIndex := 0
		for len(partialB) > 0 {
			recordEndIndex, partialB, err = p.buildRecord(partialB, recordType, recordEndIndex)
			if err != nil {
				// Return the amount of bytes written prior to the error.
				return bStart, err
			}
		}
		// Write the bytes stored in outRecordsBuf to p.Conn. If there is an
		// error, calculate the total number of plaintext bytes of complete
		// records successfully written to the peer and return it.
		nn, err := p.Conn.Write(p.outRecordsBuf[:recordEndIndex])
		if err != nil {
			numberOfCompletedRecords := int(math.Floor(float64(nn) / float64(tlsRecordMaxSize)))
			return bStart + numberOfCompletedRecords*tlsRecordMaxPlaintextSize, err
		}
	}
	return len(b), nil
}

// buildRecord builds a TLS 1.3 record of type recordType from plaintext,
// and writes the record to outRecordsBuf at recordStartIndex. The record will
// have at most tlsRecordMaxPlaintextSize bytes of payload. It returns the
// index of outRecordsBuf where the current record ends, as well as any
// remaining plaintext bytes.
func (p *conn) buildRecord(plaintext []byte, recordType byte, recordStartIndex int) (n int, remainingPlaintext []byte, err error) {
	// Construct the payload, which consists of application data and record type.
	dataLen := len(plaintext)
	if dataLen > tlsRecordMaxPlaintextSize {
		dataLen = tlsRecordMaxPlaintextSize
	}
	remainingPlaintext = plaintext[dataLen:]
	newRecordBuf := p.outRecordsBuf[recordStartIndex:]

	copy(newRecordBuf[tlsRecordHeaderSize:], plaintext[:dataLen])
	newRecordBuf[tlsRecordHeaderSize+dataLen] = recordType
	payload := newRecordBuf[tlsRecordHeaderSize : tlsRecordHeaderSize+dataLen+1] // 1 is for the recordType.
	// Construct the header.
	newRecordBuf[0] = tlsApplicationData
	newRecordBuf[1] = tlsLegacyRecordVersion
	newRecordBuf[2] = tlsLegacyRecordVersion
	binary.BigEndian.PutUint16(newRecordBuf[3:], uint16(len(payload)+tlsTagSize))
	header := newRecordBuf[:tlsRecordHeaderSize]

	// Encrypt the payload using header as aad.
	encryptedPayload, err := p.outConn.Encrypt(newRecordBuf[tlsRecordHeaderSize:][:0], payload, header)
	if err != nil {
		return 0, plaintext, err
	}
	recordStartIndex += len(header) + len(encryptedPayload)
	return recordStartIndex, remainingPlaintext, nil
}

func (p *conn) Close() error {
	p.readMutex.Lock()
	defer p.readMutex.Unlock()
	p.writeMutex.Lock()
	defer p.writeMutex.Unlock()
	// If p.ticketState is equal to notReceivingTickets, then S2A has
	// been sent a flight of session tickets, and we must wait for the
	// call to S2A to complete before closing the record protocol.
	if p.ticketState == notReceivingTickets {
		<-p.callComplete
		grpclog.Infof("Safe to close the connection because sending tickets to S2A is (already) complete.")
	}
	return p.Conn.Close()
}

// stripPaddingAndType strips the padding by zeros and record type from
// p.pendingApplicationData and returns the record type. Note that
// p.pendingApplicationData should be of the form:
// [application data] + [record type byte] + [trailing zeros]
func (p *conn) stripPaddingAndType() (recordType, error) {
	if len(p.pendingApplicationData) == 0 {
		return 0, errors.New("application data had length 0")
	}
	i := len(p.pendingApplicationData) - 1
	// Search for the index of the record type byte.
	for i > 0 {
		if p.pendingApplicationData[i] != 0 {
			break
		}
		i--
	}
	rt := recordType(p.pendingApplicationData[i])
	p.pendingApplicationData = p.pendingApplicationData[:i]
	return rt, nil
}

// readFullRecord reads from the wire until a record is completed and returns
// the full record.
func (p *conn) readFullRecord() (fullRecord []byte, err error) {
	fullRecord, p.nextRecord, err = parseReadBuffer(p.nextRecord, tlsRecordMaxPayloadSize)
	if err != nil {
		return nil, err
	}
	// Check whether the next record to be decrypted has been completely
	// received.
	if len(fullRecord) == 0 {
		copy(p.unusedBuf, p.nextRecord)
		p.unusedBuf = p.unusedBuf[:len(p.nextRecord)]
		// Always copy next incomplete record to the beginning of the
		// unusedBuf buffer and reset nextRecord to it.
		p.nextRecord = p.unusedBuf
	}
	// Keep reading from the wire until we have a complete record.
	for len(fullRecord) == 0 {
		if len(p.unusedBuf) == cap(p.unusedBuf) {
			tmp := make([]byte, len(p.unusedBuf), cap(p.unusedBuf)+tlsRecordMaxSize)
			copy(tmp, p.unusedBuf)
			p.unusedBuf = tmp
		}
		n, err := p.Conn.Read(p.unusedBuf[len(p.unusedBuf):min(cap(p.unusedBuf), len(p.unusedBuf)+tlsRecordMaxSize)])
		if err != nil {
			return nil, err
		}
		p.unusedBuf = p.unusedBuf[:len(p.unusedBuf)+n]
		fullRecord, p.nextRecord, err = parseReadBuffer(p.unusedBuf, tlsRecordMaxPayloadSize)
		if err != nil {
			return nil, err
		}
	}
	return fullRecord, nil
}

// parseReadBuffer parses the provided buffer and returns a full record and any
// remaining bytes in that buffer. If the record is incomplete, nil is returned
// for the first return value and the given byte buffer is returned for the
// second return value. The length of the payload specified by the header should
// not be greater than maxLen, otherwise an error is returned. Note that this
// function does not allocate or copy any buffers.
func parseReadBuffer(b []byte, maxLen uint16) (fullRecord, remaining []byte, err error) {
	// If the header is not complete, return the provided buffer as remaining
	// buffer.
	if len(b) < tlsRecordHeaderSize {
		return nil, b, nil
	}
	msgLenField := b[tlsRecordHeaderTypeSize+tlsRecordHeaderLegacyRecordVersionSize : tlsRecordHeaderSize]
	length := binary.BigEndian.Uint16(msgLenField)
	if length > maxLen {
		return nil, nil, fmt.Errorf("record length larger than the limit %d", maxLen)
	}
	if len(b) < int(length)+tlsRecordHeaderSize {
		// Record is not complete yet.
		return nil, b, nil
	}
	return b[:tlsRecordHeaderSize+length], b[tlsRecordHeaderSize+length:], nil
}

// splitAndValidateHeader splits the header from the payload in the TLS 1.3
// record and returns them. Note that the header is checked for validity, and an
// error is returned when an invalid header is parsed. Also note that this
// function does not allocate or copy any buffers.
func splitAndValidateHeader(record []byte) (header, payload []byte, err error) {
	if len(record) < tlsRecordHeaderSize {
		return nil, nil, fmt.Errorf("record was smaller than the header size")
	}
	header = record[:tlsRecordHeaderSize]
	payload = record[tlsRecordHeaderSize:]
	if header[0] != tlsApplicationData {
		return nil, nil, fmt.Errorf("incorrect type in the header")
	}
	// Check the legacy record version, which should be 0x03, 0x03.
	if header[1] != 0x03 || header[2] != 0x03 {
		return nil, nil, fmt.Errorf("incorrect legacy record version in the header")
	}
	return header, payload, nil
}

// handleAlertMessage handles an alert message.
func (p *conn) handleAlertMessage() error {
	if len(p.pendingApplicationData) != tlsAlertSize {
		return errors.New("invalid alert message size")
	}
	alertType := p.pendingApplicationData[1]
	// Clear the body of the alert message.
	p.pendingApplicationData = p.pendingApplicationData[:0]
	if alertType == byte(closeNotify) {
		return errors.New("received a close notify alert")
	}
	// TODO(matthewstevenson88): Add support for more alert types.
	return fmt.Errorf("received an unrecognized alert type: %v", alertType)
}

// parseHandshakeHeader parses a handshake message from the handshake buffer.
// It returns the message type, the message length, the message, the raw message
// that includes the type and length bytes and a flag indicating whether the
// handshake message has been fully parsed. i.e. whether the entire handshake
// message was in the handshake buffer.
func (p *conn) parseHandshakeMsg() (msgType byte, msgLen uint32, msg []byte, rawMsg []byte, ok bool) {
	// Handle the case where the 4 byte handshake header is fragmented.
	if len(p.handshakeBuf) < tlsHandshakePrefixSize {
		return 0, 0, nil, nil, false
	}
	msgType = p.handshakeBuf[0]
	msgLen = bigEndianInt24(p.handshakeBuf[tlsHandshakeMsgTypeSize : tlsHandshakeMsgTypeSize+tlsHandshakeLengthSize])
	if msgLen > uint32(len(p.handshakeBuf)-tlsHandshakePrefixSize) {
		return 0, 0, nil, nil, false
	}
	msg = p.handshakeBuf[tlsHandshakePrefixSize : tlsHandshakePrefixSize+msgLen]
	rawMsg = p.handshakeBuf[:tlsHandshakeMsgTypeSize+tlsHandshakeLengthSize+msgLen]
	p.handshakeBuf = p.handshakeBuf[tlsHandshakePrefixSize+msgLen:]
	return msgType, msgLen, msg, rawMsg, true
}

// handleHandshakeMessage handles a handshake message. Note that the first
// complete handshake message from the handshake buffer is removed, if it
// exists.
func (p *conn) handleHandshakeMessage() error {
	// Copy the pending application data to the handshake buffer. At this point,
	// we are guaranteed that the pending application data contains only parts
	// of a handshake message.
	p.handshakeBuf = append(p.handshakeBuf, p.pendingApplicationData...)
	p.pendingApplicationData = p.pendingApplicationData[:0]
	// Several handshake messages may be coalesced into a single record.
	// Continue reading them until the handshake buffer is empty.
	for len(p.handshakeBuf) > 0 {
		handshakeMsgType, msgLen, msg, rawMsg, ok := p.parseHandshakeMsg()
		if !ok {
			// The handshake could not be fully parsed, so read in another
			// record and try again later.
			break
		}
		switch handshakeMsgType {
		case tlsHandshakeKeyUpdateType:
			if msgLen != tlsHandshakeKeyUpdateMsgSize {
				return errors.New("invalid handshake key update message length")
			}
			if len(p.handshakeBuf) != 0 {
				return errors.New("key update message must be the last message of a handshake record")
			}
			if err := p.handleKeyUpdateMsg(msg); err != nil {
				return err
			}
		case tlsHandshakeNewSessionTicketType:
			// Ignore tickets that are received after a batch of tickets has
			// been sent to S2A.
			if p.ticketState == notReceivingTickets {
				continue
			}
			if p.ticketState == ticketsNotYetReceived {
				p.ticketState = receivingTickets
			}
			p.sessionTickets = append(p.sessionTickets, rawMsg)
			if len(p.sessionTickets) == maxAllowedTickets {
				p.ticketState = notReceivingTickets
				grpclog.Infof("Sending session tickets to S2A.")
				p.ticketSender.sendTicketsToS2A(p.sessionTickets, p.callComplete)
			}
		default:
			return errors.New("unknown handshake message type")
		}
	}
	return nil
}

func buildKeyUpdateRequest() []byte {
	b := make([]byte, tlsHandshakePrefixSize+tlsHandshakeKeyUpdateMsgSize)
	b[0] = tlsHandshakeKeyUpdateType
	b[1] = 0
	b[2] = 0
	b[3] = tlsHandshakeKeyUpdateMsgSize
	b[4] = byte(updateNotRequested)
	return b
}

// handleKeyUpdateMsg handles a key update message.
func (p *conn) handleKeyUpdateMsg(msg []byte) error {
	keyUpdateRequest := msg[0]
	if keyUpdateRequest != byte(updateNotRequested) &&
		keyUpdateRequest != byte(updateRequested) {
		return errors.New("invalid handshake key update message")
	}
	if err := p.inConn.UpdateKey(); err != nil {
		return err
	}
	// Send a key update message back to the peer if requested.
	if keyUpdateRequest == byte(updateRequested) {
		p.writeMutex.Lock()
		defer p.writeMutex.Unlock()
		n, err := p.writeTLSRecord(preConstructedKeyUpdateMsg, byte(handshake))
		if err != nil {
			return err
		}
		if n != tlsHandshakePrefixSize+tlsHandshakeKeyUpdateMsgSize {
			return errors.New("key update request message wrote less bytes than expected")
		}
		if err = p.outConn.UpdateKey(); err != nil {
			return err
		}
	}
	return nil
}

// bidEndianInt24 converts the given byte buffer of at least size 3 and
// outputs the resulting 24 bit integer as a uint32. This is needed because
// TLS 1.3 requires 3 byte integers, and the binary.BigEndian package does
// not provide a way to transform a byte buffer into a 3 byte integer.
func bigEndianInt24(b []byte) uint32 {
	_ = b[2] // bounds check hint to compiler; see golang.org/issue/14808
	return uint32(b[2]) | uint32(b[1])<<8 | uint32(b[0])<<16
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
