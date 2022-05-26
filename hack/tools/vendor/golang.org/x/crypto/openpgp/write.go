// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"crypto"
	"hash"
	"io"
	"strconv"
	"time"

	"golang.org/x/crypto/openpgp/armor"
	"golang.org/x/crypto/openpgp/errors"
	"golang.org/x/crypto/openpgp/packet"
	"golang.org/x/crypto/openpgp/s2k"
)

// DetachSign signs message with the private key from signer (which must
// already have been decrypted) and writes the signature to w.
// If config is nil, sensible defaults will be used.
func DetachSign(w io.Writer, signer *Entity, message io.Reader, config *packet.Config) error {
	return detachSign(w, signer, message, packet.SigTypeBinary, config)
}

// ArmoredDetachSign signs message with the private key from signer (which
// must already have been decrypted) and writes an armored signature to w.
// If config is nil, sensible defaults will be used.
func ArmoredDetachSign(w io.Writer, signer *Entity, message io.Reader, config *packet.Config) (err error) {
	return armoredDetachSign(w, signer, message, packet.SigTypeBinary, config)
}

// DetachSignText signs message (after canonicalising the line endings) with
// the private key from signer (which must already have been decrypted) and
// writes the signature to w.
// If config is nil, sensible defaults will be used.
func DetachSignText(w io.Writer, signer *Entity, message io.Reader, config *packet.Config) error {
	return detachSign(w, signer, message, packet.SigTypeText, config)
}

// ArmoredDetachSignText signs message (after canonicalising the line endings)
// with the private key from signer (which must already have been decrypted)
// and writes an armored signature to w.
// If config is nil, sensible defaults will be used.
func ArmoredDetachSignText(w io.Writer, signer *Entity, message io.Reader, config *packet.Config) error {
	return armoredDetachSign(w, signer, message, packet.SigTypeText, config)
}

func armoredDetachSign(w io.Writer, signer *Entity, message io.Reader, sigType packet.SignatureType, config *packet.Config) (err error) {
	out, err := armor.Encode(w, SignatureType, nil)
	if err != nil {
		return
	}
	err = detachSign(out, signer, message, sigType, config)
	if err != nil {
		return
	}
	return out.Close()
}

func detachSign(w io.Writer, signer *Entity, message io.Reader, sigType packet.SignatureType, config *packet.Config) (err error) {
	if signer.PrivateKey == nil {
		return errors.InvalidArgumentError("signing key doesn't have a private key")
	}
	if signer.PrivateKey.Encrypted {
		return errors.InvalidArgumentError("signing key is encrypted")
	}

	sig := new(packet.Signature)
	sig.SigType = sigType
	sig.PubKeyAlgo = signer.PrivateKey.PubKeyAlgo
	sig.Hash = config.Hash()
	sig.CreationTime = config.Now()
	sig.IssuerKeyId = &signer.PrivateKey.KeyId

	h, wrappedHash, err := hashForSignature(sig.Hash, sig.SigType)
	if err != nil {
		return
	}
	io.Copy(wrappedHash, message)

	err = sig.Sign(h, signer.PrivateKey, config)
	if err != nil {
		return
	}

	return sig.Serialize(w)
}

// FileHints contains metadata about encrypted files. This metadata is, itself,
// encrypted.
type FileHints struct {
	// IsBinary can be set to hint that the contents are binary data.
	IsBinary bool
	// FileName hints at the name of the file that should be written. It's
	// truncated to 255 bytes if longer. It may be empty to suggest that the
	// file should not be written to disk. It may be equal to "_CONSOLE" to
	// suggest the data should not be written to disk.
	FileName string
	// ModTime contains the modification time of the file, or the zero time if not applicable.
	ModTime time.Time
}

// SymmetricallyEncrypt acts like gpg -c: it encrypts a file with a passphrase.
// The resulting WriteCloser must be closed after the contents of the file have
// been written.
// If config is nil, sensible defaults will be used.
func SymmetricallyEncrypt(ciphertext io.Writer, passphrase []byte, hints *FileHints, config *packet.Config) (plaintext io.WriteCloser, err error) {
	if hints == nil {
		hints = &FileHints{}
	}

	key, err := packet.SerializeSymmetricKeyEncrypted(ciphertext, passphrase, config)
	if err != nil {
		return
	}
	w, err := packet.SerializeSymmetricallyEncrypted(ciphertext, config.Cipher(), key, config)
	if err != nil {
		return
	}

	literaldata := w
	if algo := config.Compression(); algo != packet.CompressionNone {
		var compConfig *packet.CompressionConfig
		if config != nil {
			compConfig = config.CompressionConfig
		}
		literaldata, err = packet.SerializeCompressed(w, algo, compConfig)
		if err != nil {
			return
		}
	}

	var epochSeconds uint32
	if !hints.ModTime.IsZero() {
		epochSeconds = uint32(hints.ModTime.Unix())
	}
	return packet.SerializeLiteral(literaldata, hints.IsBinary, hints.FileName, epochSeconds)
}

// intersectPreferences mutates and returns a prefix of a that contains only
// the values in the intersection of a and b. The order of a is preserved.
func intersectPreferences(a []uint8, b []uint8) (intersection []uint8) {
	var j int
	for _, v := range a {
		for _, v2 := range b {
			if v == v2 {
				a[j] = v
				j++
				break
			}
		}
	}

	return a[:j]
}

func hashToHashId(h crypto.Hash) uint8 {
	v, ok := s2k.HashToHashId(h)
	if !ok {
		panic("tried to convert unknown hash")
	}
	return v
}

// writeAndSign writes the data as a payload package and, optionally, signs
// it. hints contains optional information, that is also encrypted,
// that aids the recipients in processing the message. The resulting
// WriteCloser must be closed after the contents of the file have been
// written. If config is nil, sensible defaults will be used.
func writeAndSign(payload io.WriteCloser, candidateHashes []uint8, signed *Entity, hints *FileHints, config *packet.Config) (plaintext io.WriteCloser, err error) {
	var signer *packet.PrivateKey
	if signed != nil {
		signKey, ok := signed.signingKey(config.Now())
		if !ok {
			return nil, errors.InvalidArgumentError("no valid signing keys")
		}
		signer = signKey.PrivateKey
		if signer == nil {
			return nil, errors.InvalidArgumentError("no private key in signing key")
		}
		if signer.Encrypted {
			return nil, errors.InvalidArgumentError("signing key must be decrypted")
		}
	}

	var hash crypto.Hash
	for _, hashId := range candidateHashes {
		if h, ok := s2k.HashIdToHash(hashId); ok && h.Available() {
			hash = h
			break
		}
	}

	// If the hash specified by config is a candidate, we'll use that.
	if configuredHash := config.Hash(); configuredHash.Available() {
		for _, hashId := range candidateHashes {
			if h, ok := s2k.HashIdToHash(hashId); ok && h == configuredHash {
				hash = h
				break
			}
		}
	}

	if hash == 0 {
		hashId := candidateHashes[0]
		name, ok := s2k.HashIdToString(hashId)
		if !ok {
			name = "#" + strconv.Itoa(int(hashId))
		}
		return nil, errors.InvalidArgumentError("cannot encrypt because no candidate hash functions are compiled in. (Wanted " + name + " in this case.)")
	}

	if signer != nil {
		ops := &packet.OnePassSignature{
			SigType:    packet.SigTypeBinary,
			Hash:       hash,
			PubKeyAlgo: signer.PubKeyAlgo,
			KeyId:      signer.KeyId,
			IsLast:     true,
		}
		if err := ops.Serialize(payload); err != nil {
			return nil, err
		}
	}

	if hints == nil {
		hints = &FileHints{}
	}

	w := payload
	if signer != nil {
		// If we need to write a signature packet after the literal
		// data then we need to stop literalData from closing
		// encryptedData.
		w = noOpCloser{w}

	}
	var epochSeconds uint32
	if !hints.ModTime.IsZero() {
		epochSeconds = uint32(hints.ModTime.Unix())
	}
	literalData, err := packet.SerializeLiteral(w, hints.IsBinary, hints.FileName, epochSeconds)
	if err != nil {
		return nil, err
	}

	if signer != nil {
		return signatureWriter{payload, literalData, hash, hash.New(), signer, config}, nil
	}
	return literalData, nil
}

// Encrypt encrypts a message to a number of recipients and, optionally, signs
// it. hints contains optional information, that is also encrypted, that aids
// the recipients in processing the message. The resulting WriteCloser must
// be closed after the contents of the file have been written.
// If config is nil, sensible defaults will be used.
func Encrypt(ciphertext io.Writer, to []*Entity, signed *Entity, hints *FileHints, config *packet.Config) (plaintext io.WriteCloser, err error) {
	if len(to) == 0 {
		return nil, errors.InvalidArgumentError("no encryption recipient provided")
	}

	// These are the possible ciphers that we'll use for the message.
	candidateCiphers := []uint8{
		uint8(packet.CipherAES128),
		uint8(packet.CipherAES256),
		uint8(packet.CipherCAST5),
	}
	// These are the possible hash functions that we'll use for the signature.
	candidateHashes := []uint8{
		hashToHashId(crypto.SHA256),
		hashToHashId(crypto.SHA384),
		hashToHashId(crypto.SHA512),
		hashToHashId(crypto.SHA1),
		hashToHashId(crypto.RIPEMD160),
	}
	// In the event that a recipient doesn't specify any supported ciphers
	// or hash functions, these are the ones that we assume that every
	// implementation supports.
	defaultCiphers := candidateCiphers[len(candidateCiphers)-1:]
	defaultHashes := candidateHashes[len(candidateHashes)-1:]

	encryptKeys := make([]Key, len(to))
	for i := range to {
		var ok bool
		encryptKeys[i], ok = to[i].encryptionKey(config.Now())
		if !ok {
			return nil, errors.InvalidArgumentError("cannot encrypt a message to key id " + strconv.FormatUint(to[i].PrimaryKey.KeyId, 16) + " because it has no encryption keys")
		}

		sig := to[i].primaryIdentity().SelfSignature

		preferredSymmetric := sig.PreferredSymmetric
		if len(preferredSymmetric) == 0 {
			preferredSymmetric = defaultCiphers
		}
		preferredHashes := sig.PreferredHash
		if len(preferredHashes) == 0 {
			preferredHashes = defaultHashes
		}
		candidateCiphers = intersectPreferences(candidateCiphers, preferredSymmetric)
		candidateHashes = intersectPreferences(candidateHashes, preferredHashes)
	}

	if len(candidateCiphers) == 0 || len(candidateHashes) == 0 {
		return nil, errors.InvalidArgumentError("cannot encrypt because recipient set shares no common algorithms")
	}

	cipher := packet.CipherFunction(candidateCiphers[0])
	// If the cipher specified by config is a candidate, we'll use that.
	configuredCipher := config.Cipher()
	for _, c := range candidateCiphers {
		cipherFunc := packet.CipherFunction(c)
		if cipherFunc == configuredCipher {
			cipher = cipherFunc
			break
		}
	}

	symKey := make([]byte, cipher.KeySize())
	if _, err := io.ReadFull(config.Random(), symKey); err != nil {
		return nil, err
	}

	for _, key := range encryptKeys {
		if err := packet.SerializeEncryptedKey(ciphertext, key.PublicKey, cipher, symKey, config); err != nil {
			return nil, err
		}
	}

	payload, err := packet.SerializeSymmetricallyEncrypted(ciphertext, cipher, symKey, config)
	if err != nil {
		return
	}

	return writeAndSign(payload, candidateHashes, signed, hints, config)
}

// Sign signs a message. The resulting WriteCloser must be closed after the
// contents of the file have been written.  hints contains optional information
// that aids the recipients in processing the message.
// If config is nil, sensible defaults will be used.
func Sign(output io.Writer, signed *Entity, hints *FileHints, config *packet.Config) (input io.WriteCloser, err error) {
	if signed == nil {
		return nil, errors.InvalidArgumentError("no signer provided")
	}

	// These are the possible hash functions that we'll use for the signature.
	candidateHashes := []uint8{
		hashToHashId(crypto.SHA256),
		hashToHashId(crypto.SHA384),
		hashToHashId(crypto.SHA512),
		hashToHashId(crypto.SHA1),
		hashToHashId(crypto.RIPEMD160),
	}
	defaultHashes := candidateHashes[len(candidateHashes)-1:]
	preferredHashes := signed.primaryIdentity().SelfSignature.PreferredHash
	if len(preferredHashes) == 0 {
		preferredHashes = defaultHashes
	}
	candidateHashes = intersectPreferences(candidateHashes, preferredHashes)
	return writeAndSign(noOpCloser{output}, candidateHashes, signed, hints, config)
}

// signatureWriter hashes the contents of a message while passing it along to
// literalData. When closed, it closes literalData, writes a signature packet
// to encryptedData and then also closes encryptedData.
type signatureWriter struct {
	encryptedData io.WriteCloser
	literalData   io.WriteCloser
	hashType      crypto.Hash
	h             hash.Hash
	signer        *packet.PrivateKey
	config        *packet.Config
}

func (s signatureWriter) Write(data []byte) (int, error) {
	s.h.Write(data)
	return s.literalData.Write(data)
}

func (s signatureWriter) Close() error {
	sig := &packet.Signature{
		SigType:      packet.SigTypeBinary,
		PubKeyAlgo:   s.signer.PubKeyAlgo,
		Hash:         s.hashType,
		CreationTime: s.config.Now(),
		IssuerKeyId:  &s.signer.KeyId,
	}

	if err := sig.Sign(s.h, s.signer, s.config); err != nil {
		return err
	}
	if err := s.literalData.Close(); err != nil {
		return err
	}
	if err := sig.Serialize(s.encryptedData); err != nil {
		return err
	}
	return s.encryptedData.Close()
}

// noOpCloser is like an ioutil.NopCloser, but for an io.Writer.
// TODO: we have two of these in OpenPGP packages alone. This probably needs
// to be promoted somewhere more common.
type noOpCloser struct {
	w io.Writer
}

func (c noOpCloser) Write(data []byte) (n int, err error) {
	return c.w.Write(data)
}

func (c noOpCloser) Close() error {
	return nil
}
