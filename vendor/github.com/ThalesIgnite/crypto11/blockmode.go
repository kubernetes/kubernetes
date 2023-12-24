// Copyright 2018 Thales e-Security, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package crypto11

import (
	"crypto/cipher"
	"runtime"

	"github.com/miekg/pkcs11"
)

// cipher.BlockMode -----------------------------------------------------

// BlockModeCloser represents a block cipher running in a block-based mode (e.g. CBC).
//
// BlockModeCloser embeds cipher.BlockMode, and can be used as such.
// However, in this case
// (or if the Close() method is not explicitly called for any other reason),
// resources allocated to it may remain live indefinitely.
type BlockModeCloser interface {
	cipher.BlockMode

	// Close() releases resources associated with the block mode.
	Close()
}

const (
	modeEncrypt = iota // blockModeCloser is in encrypt mode
	modeDecrypt        // blockModeCloser is in decrypt mode
)

// NewCBCEncrypter returns a cipher.BlockMode which encrypts in cipher block chaining mode, using the given key.
// The length of iv must be the same as the key's block size.
//
// The new BlockMode acquires persistent resources which are released (eventually) by a finalizer.
// If this is a problem for your application then use NewCBCEncrypterCloser instead.
//
// If that is not possible then adding calls to runtime.GC() may help.
func (key *SecretKey) NewCBCEncrypter(iv []byte) (cipher.BlockMode, error) {
	return key.newBlockModeCloser(key.Cipher.CBCMech, modeEncrypt, iv, true)
}

// NewCBCDecrypter returns a cipher.BlockMode which decrypts in cipher block chaining mode, using the given key.
// The length of iv must be the same as the key's block size and must match the iv used to encrypt the data.
//
// The new BlockMode acquires persistent resources which are released (eventually) by a finalizer.
// If this is a problem for your application then use NewCBCDecrypterCloser instead.
//
// If that is not possible then adding calls to runtime.GC() may help.
func (key *SecretKey) NewCBCDecrypter(iv []byte) (cipher.BlockMode, error) {
	return key.newBlockModeCloser(key.Cipher.CBCMech, modeDecrypt, iv, true)
}

// NewCBCEncrypterCloser returns a  BlockModeCloser which encrypts in cipher block chaining mode, using the given key.
// The length of iv must be the same as the key's block size.
//
// Use of NewCBCEncrypterCloser rather than NewCBCEncrypter represents a commitment to call the Close() method
// of the returned BlockModeCloser.
func (key *SecretKey) NewCBCEncrypterCloser(iv []byte) (BlockModeCloser, error) {
	return key.newBlockModeCloser(key.Cipher.CBCMech, modeEncrypt, iv, false)
}

// NewCBCDecrypterCloser returns a  BlockModeCloser which decrypts in cipher block chaining mode, using the given key.
// The length of iv must be the same as the key's block size and must match the iv used to encrypt the data.
//
// Use of NewCBCDecrypterCloser rather than NewCBCEncrypter represents a commitment to call the Close() method
// of the returned BlockModeCloser.
func (key *SecretKey) NewCBCDecrypterCloser(iv []byte) (BlockModeCloser, error) {
	return key.newBlockModeCloser(key.Cipher.CBCMech, modeDecrypt, iv, false)
}

// blockModeCloser is a concrete implementation of BlockModeCloser supporting CBC.
type blockModeCloser struct {
	// PKCS#11 session to use
	session *pkcs11Session

	// Cipher block size
	blockSize int

	// modeDecrypt or modeEncrypt
	mode int

	// Cleanup function
	cleanup func()
}

// newBlockModeCloser creates a new blockModeCloser for the chosen mechanism and mode.
func (key *SecretKey) newBlockModeCloser(mech uint, mode int, iv []byte, setFinalizer bool) (*blockModeCloser, error) {

	session, err := key.context.getSession()
	if err != nil {
		return nil, err
	}

	bmc := &blockModeCloser{
		session:   session,
		blockSize: key.Cipher.BlockSize,
		mode:      mode,
		cleanup: func() {
			key.context.pool.Put(session)
		},
	}
	mechDescription := []*pkcs11.Mechanism{pkcs11.NewMechanism(mech, iv)}

	switch mode {
	case modeDecrypt:
		err = session.ctx.DecryptInit(session.handle, mechDescription, key.handle)
	case modeEncrypt:
		err = session.ctx.EncryptInit(bmc.session.handle, mechDescription, key.handle)
	default:
		panic("unexpected mode")
	}
	if err != nil {
		bmc.cleanup()
		return nil, err
	}
	if setFinalizer {
		runtime.SetFinalizer(bmc, finalizeBlockModeCloser)
	}

	return bmc, nil
}

func finalizeBlockModeCloser(obj interface{}) {
	obj.(*blockModeCloser).Close()
}

func (bmc *blockModeCloser) BlockSize() int {
	return bmc.blockSize
}

func (bmc *blockModeCloser) CryptBlocks(dst, src []byte) {
	if len(dst) < len(src) {
		panic("destination buffer too small")
	}
	if len(src)%bmc.blockSize != 0 {
		panic("input is not a whole number of blocks")
	}
	var result []byte
	var err error
	switch bmc.mode {
	case modeDecrypt:
		result, err = bmc.session.ctx.DecryptUpdate(bmc.session.handle, src)
	case modeEncrypt:
		result, err = bmc.session.ctx.EncryptUpdate(bmc.session.handle, src)
	}
	if err != nil {
		panic(err)
	}
	// PKCS#11 2.40 s5.2 says that the operation must produce as much output
	// as possible, so we should never have less than we submitted for CBC.
	// This could be different for other modes but we don't implement any yet.
	if len(result) != len(src) {
		panic("nontrivial result from *Final operation")
	}
	copy(dst[:len(result)], result)
	runtime.KeepAlive(bmc)
}

func (bmc *blockModeCloser) Close() {
	if bmc.session == nil {
		return
	}
	var result []byte
	var err error
	switch bmc.mode {
	case modeDecrypt:
		result, err = bmc.session.ctx.DecryptFinal(bmc.session.handle)
	case modeEncrypt:
		result, err = bmc.session.ctx.EncryptFinal(bmc.session.handle)
	}
	bmc.session = nil
	bmc.cleanup()
	if err != nil {
		panic(err)
	}
	// PKCS#11 2.40 s5.2 says that the operation must produce as much output
	// as possible, so we should never have any left over for CBC.
	// This could be different for other modes but we don't implement any yet.
	if len(result) > 0 {
		panic("nontrivial result from *Final operation")
	}
}
