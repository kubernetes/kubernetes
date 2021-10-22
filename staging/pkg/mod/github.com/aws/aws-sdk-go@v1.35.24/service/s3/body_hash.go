package s3

import (
	"bytes"
	"crypto/md5"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"hash"
	"io"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

const (
	contentMD5Header    = "Content-Md5"
	contentSha256Header = "X-Amz-Content-Sha256"
	amzTeHeader         = "X-Amz-Te"
	amzTxEncodingHeader = "X-Amz-Transfer-Encoding"

	appendMD5TxEncoding = "append-md5"
)

// computeBodyHashes will add Content MD5 and Content Sha256 hashes to the
// request. If the body is not seekable or S3DisableContentMD5Validation set
// this handler will be ignored.
func computeBodyHashes(r *request.Request) {
	if aws.BoolValue(r.Config.S3DisableContentMD5Validation) {
		return
	}
	if r.IsPresigned() {
		return
	}
	if r.Error != nil || !aws.IsReaderSeekable(r.Body) {
		return
	}

	var md5Hash, sha256Hash hash.Hash
	hashers := make([]io.Writer, 0, 2)

	// Determine upfront which hashes can be set without overriding user
	// provide header data.
	if v := r.HTTPRequest.Header.Get(contentMD5Header); len(v) == 0 {
		md5Hash = md5.New()
		hashers = append(hashers, md5Hash)
	}

	if v := r.HTTPRequest.Header.Get(contentSha256Header); len(v) == 0 {
		sha256Hash = sha256.New()
		hashers = append(hashers, sha256Hash)
	}

	// Create the destination writer based on the hashes that are not already
	// provided by the user.
	var dst io.Writer
	switch len(hashers) {
	case 0:
		return
	case 1:
		dst = hashers[0]
	default:
		dst = io.MultiWriter(hashers...)
	}

	if _, err := aws.CopySeekableBody(dst, r.Body); err != nil {
		r.Error = awserr.New("BodyHashError", "failed to compute body hashes", err)
		return
	}

	// For the hashes created, set the associated headers that the user did not
	// already provide.
	if md5Hash != nil {
		sum := make([]byte, md5.Size)
		encoded := make([]byte, md5Base64EncLen)

		base64.StdEncoding.Encode(encoded, md5Hash.Sum(sum[0:0]))
		r.HTTPRequest.Header[contentMD5Header] = []string{string(encoded)}
	}

	if sha256Hash != nil {
		encoded := make([]byte, sha256HexEncLen)
		sum := make([]byte, sha256.Size)

		hex.Encode(encoded, sha256Hash.Sum(sum[0:0]))
		r.HTTPRequest.Header[contentSha256Header] = []string{string(encoded)}
	}
}

const (
	md5Base64EncLen = (md5.Size + 2) / 3 * 4 // base64.StdEncoding.EncodedLen
	sha256HexEncLen = sha256.Size * 2        // hex.EncodedLen
)

// Adds the x-amz-te: append_md5 header to the request. This requests the service
// responds with a trailing MD5 checksum.
//
// Will not ask for append MD5 if disabled, the request is presigned or,
// or the API operation does not support content MD5 validation.
func askForTxEncodingAppendMD5(r *request.Request) {
	if aws.BoolValue(r.Config.S3DisableContentMD5Validation) {
		return
	}
	if r.IsPresigned() {
		return
	}
	r.HTTPRequest.Header.Set(amzTeHeader, appendMD5TxEncoding)
}

func useMD5ValidationReader(r *request.Request) {
	if r.Error != nil {
		return
	}

	if v := r.HTTPResponse.Header.Get(amzTxEncodingHeader); v != appendMD5TxEncoding {
		return
	}

	var bodyReader *io.ReadCloser
	var contentLen int64
	switch tv := r.Data.(type) {
	case *GetObjectOutput:
		bodyReader = &tv.Body
		contentLen = aws.Int64Value(tv.ContentLength)
		// Update ContentLength hiden the trailing MD5 checksum.
		tv.ContentLength = aws.Int64(contentLen - md5.Size)
		tv.ContentRange = aws.String(r.HTTPResponse.Header.Get("X-Amz-Content-Range"))
	default:
		r.Error = awserr.New("ChecksumValidationError",
			fmt.Sprintf("%s: %s header received on unsupported API, %s",
				amzTxEncodingHeader, appendMD5TxEncoding, r.Operation.Name,
			), nil)
		return
	}

	if contentLen < md5.Size {
		r.Error = awserr.New("ChecksumValidationError",
			fmt.Sprintf("invalid Content-Length %d for %s %s",
				contentLen, appendMD5TxEncoding, amzTxEncodingHeader,
			), nil)
		return
	}

	// Wrap and swap the response body reader with the validation reader.
	*bodyReader = newMD5ValidationReader(*bodyReader, contentLen-md5.Size)
}

type md5ValidationReader struct {
	rawReader io.ReadCloser
	payload   io.Reader
	hash      hash.Hash

	payloadLen int64
	read       int64
}

func newMD5ValidationReader(reader io.ReadCloser, payloadLen int64) *md5ValidationReader {
	h := md5.New()
	return &md5ValidationReader{
		rawReader:  reader,
		payload:    io.TeeReader(&io.LimitedReader{R: reader, N: payloadLen}, h),
		hash:       h,
		payloadLen: payloadLen,
	}
}

func (v *md5ValidationReader) Read(p []byte) (n int, err error) {
	n, err = v.payload.Read(p)
	if err != nil && err != io.EOF {
		return n, err
	}

	v.read += int64(n)

	if err == io.EOF {
		if v.read != v.payloadLen {
			return n, io.ErrUnexpectedEOF
		}
		expectSum := make([]byte, md5.Size)
		actualSum := make([]byte, md5.Size)
		if _, sumReadErr := io.ReadFull(v.rawReader, expectSum); sumReadErr != nil {
			return n, sumReadErr
		}
		actualSum = v.hash.Sum(actualSum[0:0])
		if !bytes.Equal(expectSum, actualSum) {
			return n, awserr.New("InvalidChecksum",
				fmt.Sprintf("expected MD5 checksum %s, got %s",
					hex.EncodeToString(expectSum),
					hex.EncodeToString(actualSum),
				),
				nil)
		}
	}

	return n, err
}

func (v *md5ValidationReader) Close() error {
	return v.rawReader.Close()
}
