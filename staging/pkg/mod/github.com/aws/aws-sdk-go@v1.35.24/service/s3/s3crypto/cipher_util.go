package s3crypto

import (
	"encoding/base64"
	"strconv"
)

// AESGCMNoPadding is the constant value that is used to specify
// the cek algorithm consiting of AES GCM with no padding.
const AESGCMNoPadding = "AES/GCM/NoPadding"

// AESCBC is the string constant that signifies the AES CBC algorithm cipher.
const AESCBC = "AES/CBC"

func encodeMeta(reader lengthReader, cd CipherData) (Envelope, error) {
	iv := base64.StdEncoding.EncodeToString(cd.IV)
	key := base64.StdEncoding.EncodeToString(cd.EncryptedKey)

	contentLength := reader.GetContentLength()

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
		UnencryptedContentLen: strconv.FormatInt(contentLength, 10),
	}, nil
}
