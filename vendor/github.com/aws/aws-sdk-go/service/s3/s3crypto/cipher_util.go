package s3crypto

import (
	"encoding/base64"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

func (client *DecryptionClient) contentCipherFromEnvelope(env Envelope) (ContentCipher, error) {
	wrap, err := client.wrapFromEnvelope(env)
	if err != nil {
		return nil, err
	}

	return client.cekFromEnvelope(env, wrap)
}

func (client *DecryptionClient) wrapFromEnvelope(env Envelope) (CipherDataDecrypter, error) {
	f, ok := client.WrapRegistry[env.WrapAlg]
	if !ok || f == nil {
		return nil, awserr.New(
			"InvalidWrapAlgorithmError",
			"wrap algorithm isn't supported, "+env.WrapAlg,
			nil,
		)
	}
	return f(env)
}

// AESGCMNoPadding is the constant value that is used to specify
// the CEK algorithm consiting of AES GCM with no padding.
const AESGCMNoPadding = "AES/GCM/NoPadding"

// AESCBC is the string constant that signifies the AES CBC algorithm cipher.
const AESCBC = "AES/CBC"

func (client *DecryptionClient) cekFromEnvelope(env Envelope, decrypter CipherDataDecrypter) (ContentCipher, error) {
	f, ok := client.CEKRegistry[env.CEKAlg]
	if !ok || f == nil {
		return nil, awserr.New(
			"InvalidCEKAlgorithmError",
			"cek algorithm isn't supported, "+env.CEKAlg,
			nil,
		)
	}

	key, err := base64.StdEncoding.DecodeString(env.CipherKey)
	if err != nil {
		return nil, err
	}

	iv, err := base64.StdEncoding.DecodeString(env.IV)
	if err != nil {
		return nil, err
	}
	key, err = decrypter.DecryptKey(key)
	if err != nil {
		return nil, err
	}

	cd := CipherData{
		Key:          key,
		IV:           iv,
		CEKAlgorithm: env.CEKAlg,
		Padder:       client.getPadder(env.CEKAlg),
	}
	return f(cd)
}

// getPadder will return an unpadder with checking the cek algorithm specific padder.
// If there wasn't a cek algorithm specific padder, we check the padder itself.
// We return a no unpadder, if no unpadder was found. This means any customization
// either contained padding within the cipher implementation, and to maintain
// backwards compatility we will simply not unpad anything.
func (client *DecryptionClient) getPadder(cekAlg string) Padder {
	padder, ok := client.PadderRegistry[cekAlg]
	if !ok {
		padder, ok = client.PadderRegistry[cekAlg[strings.LastIndex(cekAlg, "/")+1:]]
		if !ok {
			return NoPadder
		}
	}
	return padder
}

func encodeMeta(reader hashReader, cd CipherData) (Envelope, error) {
	iv := base64.StdEncoding.EncodeToString(cd.IV)
	key := base64.StdEncoding.EncodeToString(cd.EncryptedKey)

	md5 := reader.GetValue()
	contentLength := reader.GetContentLength()

	md5Str := base64.StdEncoding.EncodeToString(md5)
	matdesc, err := cd.MaterialDescription.encodeDescription()
	if err != nil {
		return Envelope{}, err
	}

	return Envelope{
		CipherKey:             key,
		IV:                    iv,
		MatDesc:               string(matdesc),
		WrapAlg:               cd.WrapAlgorithm,
		CEKAlg:                cd.CEKAlgorithm,
		TagLen:                cd.TagLength,
		UnencryptedMD5:        md5Str,
		UnencryptedContentLen: strconv.FormatInt(contentLength, 10),
	}, nil
}
