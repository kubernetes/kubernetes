package handshake

import (
	"crypto"
	"crypto/cipher"
	"crypto/tls"
	"encoding/binary"
	"fmt"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/qtls"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/logging"
)

// KeyUpdateInterval is the maximum number of packets we send or receive before initiating a key update.
// It's a package-level variable to allow modifying it for testing purposes.
var KeyUpdateInterval uint64 = protocol.KeyUpdateInterval

type updatableAEAD struct {
	suite *qtls.CipherSuiteTLS13

	keyPhase           protocol.KeyPhase
	largestAcked       protocol.PacketNumber
	firstPacketNumber  protocol.PacketNumber
	handshakeConfirmed bool

	keyUpdateInterval  uint64
	invalidPacketLimit uint64
	invalidPacketCount uint64

	// Time when the keys should be dropped. Keys are dropped on the next call to Open().
	prevRcvAEADExpiry time.Time
	prevRcvAEAD       cipher.AEAD

	firstRcvdWithCurrentKey protocol.PacketNumber
	firstSentWithCurrentKey protocol.PacketNumber
	highestRcvdPN           protocol.PacketNumber // highest packet number received (which could be successfully unprotected)
	numRcvdWithCurrentKey   uint64
	numSentWithCurrentKey   uint64
	rcvAEAD                 cipher.AEAD
	sendAEAD                cipher.AEAD
	// caches cipher.AEAD.Overhead(). This speeds up calls to Overhead().
	aeadOverhead int

	nextRcvAEAD           cipher.AEAD
	nextSendAEAD          cipher.AEAD
	nextRcvTrafficSecret  []byte
	nextSendTrafficSecret []byte

	headerDecrypter headerProtector
	headerEncrypter headerProtector

	rttStats *utils.RTTStats

	tracer logging.ConnectionTracer
	logger utils.Logger

	// use a single slice to avoid allocations
	nonceBuf []byte
}

var (
	_ ShortHeaderOpener = &updatableAEAD{}
	_ ShortHeaderSealer = &updatableAEAD{}
)

func newUpdatableAEAD(rttStats *utils.RTTStats, tracer logging.ConnectionTracer, logger utils.Logger) *updatableAEAD {
	return &updatableAEAD{
		firstPacketNumber:       protocol.InvalidPacketNumber,
		largestAcked:            protocol.InvalidPacketNumber,
		firstRcvdWithCurrentKey: protocol.InvalidPacketNumber,
		firstSentWithCurrentKey: protocol.InvalidPacketNumber,
		keyUpdateInterval:       KeyUpdateInterval,
		rttStats:                rttStats,
		tracer:                  tracer,
		logger:                  logger,
	}
}

func (a *updatableAEAD) rollKeys() {
	if a.prevRcvAEAD != nil {
		a.logger.Debugf("Dropping key phase %d ahead of scheduled time. Drop time was: %s", a.keyPhase-1, a.prevRcvAEADExpiry)
		if a.tracer != nil {
			a.tracer.DroppedKey(a.keyPhase - 1)
		}
		a.prevRcvAEADExpiry = time.Time{}
	}

	a.keyPhase++
	a.firstRcvdWithCurrentKey = protocol.InvalidPacketNumber
	a.firstSentWithCurrentKey = protocol.InvalidPacketNumber
	a.numRcvdWithCurrentKey = 0
	a.numSentWithCurrentKey = 0
	a.prevRcvAEAD = a.rcvAEAD
	a.rcvAEAD = a.nextRcvAEAD
	a.sendAEAD = a.nextSendAEAD

	a.nextRcvTrafficSecret = a.getNextTrafficSecret(a.suite.Hash, a.nextRcvTrafficSecret)
	a.nextSendTrafficSecret = a.getNextTrafficSecret(a.suite.Hash, a.nextSendTrafficSecret)
	a.nextRcvAEAD = createAEAD(a.suite, a.nextRcvTrafficSecret)
	a.nextSendAEAD = createAEAD(a.suite, a.nextSendTrafficSecret)
}

func (a *updatableAEAD) startKeyDropTimer(now time.Time) {
	d := 3 * a.rttStats.PTO(true)
	a.logger.Debugf("Starting key drop timer to drop key phase %d (in %s)", a.keyPhase-1, d)
	a.prevRcvAEADExpiry = now.Add(d)
}

func (a *updatableAEAD) getNextTrafficSecret(hash crypto.Hash, ts []byte) []byte {
	return hkdfExpandLabel(hash, ts, []byte{}, "quic ku", hash.Size())
}

// For the client, this function is called before SetWriteKey.
// For the server, this function is called after SetWriteKey.
func (a *updatableAEAD) SetReadKey(suite *qtls.CipherSuiteTLS13, trafficSecret []byte) {
	a.rcvAEAD = createAEAD(suite, trafficSecret)
	a.headerDecrypter = newHeaderProtector(suite, trafficSecret, false)
	if a.suite == nil {
		a.setAEADParameters(a.rcvAEAD, suite)
	}

	a.nextRcvTrafficSecret = a.getNextTrafficSecret(suite.Hash, trafficSecret)
	a.nextRcvAEAD = createAEAD(suite, a.nextRcvTrafficSecret)
}

// For the client, this function is called after SetReadKey.
// For the server, this function is called before SetWriteKey.
func (a *updatableAEAD) SetWriteKey(suite *qtls.CipherSuiteTLS13, trafficSecret []byte) {
	a.sendAEAD = createAEAD(suite, trafficSecret)
	a.headerEncrypter = newHeaderProtector(suite, trafficSecret, false)
	if a.suite == nil {
		a.setAEADParameters(a.sendAEAD, suite)
	}

	a.nextSendTrafficSecret = a.getNextTrafficSecret(suite.Hash, trafficSecret)
	a.nextSendAEAD = createAEAD(suite, a.nextSendTrafficSecret)
}

func (a *updatableAEAD) setAEADParameters(aead cipher.AEAD, suite *qtls.CipherSuiteTLS13) {
	a.nonceBuf = make([]byte, aead.NonceSize())
	a.aeadOverhead = aead.Overhead()
	a.suite = suite
	switch suite.ID {
	case tls.TLS_AES_128_GCM_SHA256, tls.TLS_AES_256_GCM_SHA384:
		a.invalidPacketLimit = protocol.InvalidPacketLimitAES
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		a.invalidPacketLimit = protocol.InvalidPacketLimitChaCha
	default:
		panic(fmt.Sprintf("unknown cipher suite %d", suite.ID))
	}
}

func (a *updatableAEAD) DecodePacketNumber(wirePN protocol.PacketNumber, wirePNLen protocol.PacketNumberLen) protocol.PacketNumber {
	return protocol.DecodePacketNumber(wirePNLen, a.highestRcvdPN, wirePN)
}

func (a *updatableAEAD) Open(dst, src []byte, rcvTime time.Time, pn protocol.PacketNumber, kp protocol.KeyPhaseBit, ad []byte) ([]byte, error) {
	dec, err := a.open(dst, src, rcvTime, pn, kp, ad)
	if err == ErrDecryptionFailed {
		a.invalidPacketCount++
		if a.invalidPacketCount >= a.invalidPacketLimit {
			return nil, &qerr.TransportError{ErrorCode: qerr.AEADLimitReached}
		}
	}
	if err == nil {
		a.highestRcvdPN = utils.MaxPacketNumber(a.highestRcvdPN, pn)
	}
	return dec, err
}

func (a *updatableAEAD) open(dst, src []byte, rcvTime time.Time, pn protocol.PacketNumber, kp protocol.KeyPhaseBit, ad []byte) ([]byte, error) {
	if a.prevRcvAEAD != nil && !a.prevRcvAEADExpiry.IsZero() && rcvTime.After(a.prevRcvAEADExpiry) {
		a.prevRcvAEAD = nil
		a.logger.Debugf("Dropping key phase %d", a.keyPhase-1)
		a.prevRcvAEADExpiry = time.Time{}
		if a.tracer != nil {
			a.tracer.DroppedKey(a.keyPhase - 1)
		}
	}
	binary.BigEndian.PutUint64(a.nonceBuf[len(a.nonceBuf)-8:], uint64(pn))
	if kp != a.keyPhase.Bit() {
		if a.keyPhase > 0 && a.firstRcvdWithCurrentKey == protocol.InvalidPacketNumber || pn < a.firstRcvdWithCurrentKey {
			if a.prevRcvAEAD == nil {
				return nil, ErrKeysDropped
			}
			// we updated the key, but the peer hasn't updated yet
			dec, err := a.prevRcvAEAD.Open(dst, a.nonceBuf, src, ad)
			if err != nil {
				err = ErrDecryptionFailed
			}
			return dec, err
		}
		// try opening the packet with the next key phase
		dec, err := a.nextRcvAEAD.Open(dst, a.nonceBuf, src, ad)
		if err != nil {
			return nil, ErrDecryptionFailed
		}
		// Opening succeeded. Check if the peer was allowed to update.
		if a.keyPhase > 0 && a.firstSentWithCurrentKey == protocol.InvalidPacketNumber {
			return nil, &qerr.TransportError{
				ErrorCode:    qerr.KeyUpdateError,
				ErrorMessage: "keys updated too quickly",
			}
		}
		a.rollKeys()
		a.logger.Debugf("Peer updated keys to %d", a.keyPhase)
		// The peer initiated this key update. It's safe to drop the keys for the previous generation now.
		// Start a timer to drop the previous key generation.
		a.startKeyDropTimer(rcvTime)
		if a.tracer != nil {
			a.tracer.UpdatedKey(a.keyPhase, true)
		}
		a.firstRcvdWithCurrentKey = pn
		return dec, err
	}
	// The AEAD we're using here will be the qtls.aeadAESGCM13.
	// It uses the nonce provided here and XOR it with the IV.
	dec, err := a.rcvAEAD.Open(dst, a.nonceBuf, src, ad)
	if err != nil {
		return dec, ErrDecryptionFailed
	}
	a.numRcvdWithCurrentKey++
	if a.firstRcvdWithCurrentKey == protocol.InvalidPacketNumber {
		// We initiated the key updated, and now we received the first packet protected with the new key phase.
		// Therefore, we are certain that the peer rolled its keys as well. Start a timer to drop the old keys.
		if a.keyPhase > 0 {
			a.logger.Debugf("Peer confirmed key update to phase %d", a.keyPhase)
			a.startKeyDropTimer(rcvTime)
		}
		a.firstRcvdWithCurrentKey = pn
	}
	return dec, err
}

func (a *updatableAEAD) Seal(dst, src []byte, pn protocol.PacketNumber, ad []byte) []byte {
	if a.firstSentWithCurrentKey == protocol.InvalidPacketNumber {
		a.firstSentWithCurrentKey = pn
	}
	if a.firstPacketNumber == protocol.InvalidPacketNumber {
		a.firstPacketNumber = pn
	}
	a.numSentWithCurrentKey++
	binary.BigEndian.PutUint64(a.nonceBuf[len(a.nonceBuf)-8:], uint64(pn))
	// The AEAD we're using here will be the qtls.aeadAESGCM13.
	// It uses the nonce provided here and XOR it with the IV.
	return a.sendAEAD.Seal(dst, a.nonceBuf, src, ad)
}

func (a *updatableAEAD) SetLargestAcked(pn protocol.PacketNumber) error {
	if a.firstSentWithCurrentKey != protocol.InvalidPacketNumber &&
		pn >= a.firstSentWithCurrentKey && a.numRcvdWithCurrentKey == 0 {
		return &qerr.TransportError{
			ErrorCode:    qerr.KeyUpdateError,
			ErrorMessage: fmt.Sprintf("received ACK for key phase %d, but peer didn't update keys", a.keyPhase),
		}
	}
	a.largestAcked = pn
	return nil
}

func (a *updatableAEAD) SetHandshakeConfirmed() {
	a.handshakeConfirmed = true
}

func (a *updatableAEAD) updateAllowed() bool {
	if !a.handshakeConfirmed {
		return false
	}
	// the first key update is allowed as soon as the handshake is confirmed
	return a.keyPhase == 0 ||
		// subsequent key updates as soon as a packet sent with that key phase has been acknowledged
		(a.firstSentWithCurrentKey != protocol.InvalidPacketNumber &&
			a.largestAcked != protocol.InvalidPacketNumber &&
			a.largestAcked >= a.firstSentWithCurrentKey)
}

func (a *updatableAEAD) shouldInitiateKeyUpdate() bool {
	if !a.updateAllowed() {
		return false
	}
	if a.numRcvdWithCurrentKey >= a.keyUpdateInterval {
		a.logger.Debugf("Received %d packets with current key phase. Initiating key update to the next key phase: %d", a.numRcvdWithCurrentKey, a.keyPhase+1)
		return true
	}
	if a.numSentWithCurrentKey >= a.keyUpdateInterval {
		a.logger.Debugf("Sent %d packets with current key phase. Initiating key update to the next key phase: %d", a.numSentWithCurrentKey, a.keyPhase+1)
		return true
	}
	return false
}

func (a *updatableAEAD) KeyPhase() protocol.KeyPhaseBit {
	if a.shouldInitiateKeyUpdate() {
		a.rollKeys()
		a.logger.Debugf("Initiating key update to key phase %d", a.keyPhase)
		if a.tracer != nil {
			a.tracer.UpdatedKey(a.keyPhase, false)
		}
	}
	return a.keyPhase.Bit()
}

func (a *updatableAEAD) Overhead() int {
	return a.aeadOverhead
}

func (a *updatableAEAD) EncryptHeader(sample []byte, firstByte *byte, hdrBytes []byte) {
	a.headerEncrypter.EncryptHeader(sample, firstByte, hdrBytes)
}

func (a *updatableAEAD) DecryptHeader(sample []byte, firstByte *byte, hdrBytes []byte) {
	a.headerDecrypter.DecryptHeader(sample, firstByte, hdrBytes)
}

func (a *updatableAEAD) FirstPacketNumber() protocol.PacketNumber {
	return a.firstPacketNumber
}
