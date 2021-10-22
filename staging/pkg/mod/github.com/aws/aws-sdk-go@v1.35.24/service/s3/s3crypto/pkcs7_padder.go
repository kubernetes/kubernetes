package s3crypto

// Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Portions Licensed under the MIT License. Copyright (c) 2016 Carl Jackson

import (
	"bytes"
	"crypto/subtle"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

const (
	pkcs7MaxPaddingSize = 255
)

type pkcs7Padder struct {
	blockSize int
}

// NewPKCS7Padder follows the RFC 2315: https://www.ietf.org/rfc/rfc2315.txt
// PKCS7 padding is subject to side-channel attacks and timing attacks. For
// the most secure data, use an authenticated crypto algorithm.
func NewPKCS7Padder(blockSize int) Padder {
	return pkcs7Padder{blockSize}
}

var errPKCS7Padding = awserr.New("InvalidPadding", "invalid padding", nil)

// Pad will pad the data relative to how many bytes have been read.
// Pad follows the PKCS7 standard.
func (padder pkcs7Padder) Pad(buf []byte, n int) ([]byte, error) {
	if padder.blockSize < 1 || padder.blockSize > pkcs7MaxPaddingSize {
		return nil, awserr.New("InvalidBlockSize", "block size must be between 1 and 255", nil)
	}
	size := padder.blockSize - (n % padder.blockSize)
	pad := bytes.Repeat([]byte{byte(size)}, size)
	buf = append(buf, pad...)
	return buf, nil
}

// Unpad will unpad the correct amount of bytes based off
// of the PKCS7 standard
func (padder pkcs7Padder) Unpad(buf []byte) ([]byte, error) {
	if len(buf) == 0 {
		return nil, errPKCS7Padding
	}

	// Here be dragons. We're attempting to check the padding in constant
	// time. The only piece of information here which is public is len(buf).
	// This code is modeled loosely after tls1_cbc_remove_padding from
	// OpenSSL.
	padLen := buf[len(buf)-1]
	toCheck := pkcs7MaxPaddingSize
	good := 1
	if toCheck > len(buf) {
		toCheck = len(buf)
	}
	for i := 0; i < toCheck; i++ {
		b := buf[len(buf)-1-i]

		outOfRange := subtle.ConstantTimeLessOrEq(int(padLen), i)
		equal := subtle.ConstantTimeByteEq(padLen, b)
		good &= subtle.ConstantTimeSelect(outOfRange, 1, equal)
	}

	good &= subtle.ConstantTimeLessOrEq(1, int(padLen))
	good &= subtle.ConstantTimeLessOrEq(int(padLen), len(buf))

	if good != 1 {
		return nil, errPKCS7Padding
	}

	return buf[:len(buf)-int(padLen)], nil
}

func (padder pkcs7Padder) Name() string {
	return "PKCS7Padding"
}
