package s3crypto

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"io"
)

// AESCBC is a symmetric crypto algorithm. This algorithm
// requires a padder due to CBC needing to be of the same block
// size. AES CBC is vulnerable to Padding Oracle attacks and
// so should be avoided when possible.
type aesCBC struct {
	encrypter cipher.BlockMode
	decrypter cipher.BlockMode
	padder    Padder
}

// newAESCBC creates a new AES CBC cipher. Expects keys to be of
// the correct size.
func newAESCBC(cd CipherData, padder Padder) (Cipher, error) {
	block, err := aes.NewCipher(cd.Key)
	if err != nil {
		return nil, err
	}

	encrypter := cipher.NewCBCEncrypter(block, cd.IV)
	decrypter := cipher.NewCBCDecrypter(block, cd.IV)

	return &aesCBC{encrypter, decrypter, padder}, nil
}

// Encrypt will encrypt the data using AES CBC by returning
// an io.Reader. The io.Reader will encrypt the data as Read
// is called.
func (c *aesCBC) Encrypt(src io.Reader) io.Reader {
	reader := &cbcEncryptReader{
		encrypter: c.encrypter,
		src:       src,
		padder:    c.padder,
	}
	return reader
}

type cbcEncryptReader struct {
	encrypter cipher.BlockMode
	src       io.Reader
	padder    Padder
	size      int
	buf       bytes.Buffer
}

// Read will read from our io.Reader and encrypt the data as necessary.
// Due to padding, we have to do some logic that when we encounter an
// end of file to pad properly.
func (reader *cbcEncryptReader) Read(data []byte) (int, error) {
	n, err := reader.src.Read(data)
	reader.size += n
	blockSize := reader.encrypter.BlockSize()
	reader.buf.Write(data[:n])

	if err == io.EOF {
		b := make([]byte, getSliceSize(blockSize, reader.buf.Len(), len(data)))
		n, err = reader.buf.Read(b)
		if err != nil && err != io.EOF {
			return n, err
		}
		// The buffer is now empty, we can now pad the data
		if reader.buf.Len() == 0 {
			b, err = reader.padder.Pad(b[:n], reader.size)
			if err != nil {
				return n, err
			}
			n = len(b)
			err = io.EOF
		}
		// We only want to encrypt if we have read anything
		if n > 0 {
			reader.encrypter.CryptBlocks(data, b)
		}
		return n, err
	}

	if err != nil {
		return n, err
	}

	if size := reader.buf.Len(); size >= blockSize {
		nBlocks := size / blockSize
		if size > len(data) {
			nBlocks = len(data) / blockSize
		}

		if nBlocks > 0 {
			b := make([]byte, nBlocks*blockSize)
			n, _ = reader.buf.Read(b)
			reader.encrypter.CryptBlocks(data, b[:n])
		}
	} else {
		n = 0
	}
	return n, nil
}

// Decrypt will decrypt the data using AES CBC
func (c *aesCBC) Decrypt(src io.Reader) io.Reader {
	return &cbcDecryptReader{
		decrypter: c.decrypter,
		src:       src,
		padder:    c.padder,
	}
}

type cbcDecryptReader struct {
	decrypter cipher.BlockMode
	src       io.Reader
	padder    Padder
	buf       bytes.Buffer
}

// Read will read from our io.Reader and decrypt the data as necessary.
// Due to padding, we have to do some logic that when we encounter an
// end of file to pad properly.
func (reader *cbcDecryptReader) Read(data []byte) (int, error) {
	n, err := reader.src.Read(data)
	blockSize := reader.decrypter.BlockSize()
	reader.buf.Write(data[:n])

	if err == io.EOF {
		b := make([]byte, getSliceSize(blockSize, reader.buf.Len(), len(data)))
		n, err = reader.buf.Read(b)
		if err != nil && err != io.EOF {
			return n, err
		}
		// We only want to decrypt if we have read anything
		if n > 0 {
			reader.decrypter.CryptBlocks(data, b)
		}

		if reader.buf.Len() == 0 {
			b, err = reader.padder.Unpad(data[:n])
			n = len(b)
			if err != nil {
				return n, err
			}
			err = io.EOF
		}
		return n, err
	}

	if err != nil {
		return n, err
	}

	if size := reader.buf.Len(); size >= blockSize {
		nBlocks := size / blockSize
		if size > len(data) {
			nBlocks = len(data) / blockSize
		}
		// The last block is always padded. This will allow us to unpad
		// when we receive an io.EOF error
		nBlocks -= blockSize

		if nBlocks > 0 {
			b := make([]byte, nBlocks*blockSize)
			n, _ = reader.buf.Read(b)
			reader.decrypter.CryptBlocks(data, b[:n])
		} else {
			n = 0
		}
	}

	return n, nil
}

// getSliceSize will return the correct amount of bytes we need to
// read with regards to padding.
func getSliceSize(blockSize, bufSize, dataSize int) int {
	size := bufSize
	if bufSize > dataSize {
		size = dataSize
	}
	size = size - (size % blockSize) - blockSize
	if size <= 0 {
		size = blockSize
	}

	return size
}
