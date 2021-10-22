package s3crypto

import (
	"encoding/base64"
	"encoding/hex"
	"io"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdkio"
	"github.com/aws/aws-sdk-go/service/s3"
)

// clientConstructionErrorCode is used for operations that can't be completed due to invalid client construction
const clientConstructionErrorCode = "ClientConstructionError"

// mismatchWrapError is an error returned if a wrapping handler receives an unexpected envelope
var mismatchWrapError = awserr.New(clientConstructionErrorCode, "wrap algorithm provided did not match handler", nil)

func putObjectRequest(c EncryptionClientOptions, input *s3.PutObjectInput) (*request.Request, *s3.PutObjectOutput) {
	req, out := c.S3Client.PutObjectRequest(input)

	// Get Size of file
	n, err := aws.SeekerLen(input.Body)
	if err != nil {
		req.Error = err
		return req, out
	}

	dst, err := getWriterStore(req, c.TempFolderPath, n >= c.MinFileSize)
	if err != nil {
		req.Error = err
		return req, out
	}

	req.Handlers.Build.PushFront(func(r *request.Request) {
		if err != nil {
			r.Error = err
			return
		}
		var encryptor ContentCipher
		if v, ok := c.ContentCipherBuilder.(ContentCipherBuilderWithContext); ok {
			encryptor, err = v.ContentCipherWithContext(r.Context())
		} else {
			encryptor, err = c.ContentCipherBuilder.ContentCipher()
		}
		if err != nil {
			r.Error = err
			return
		}

		lengthReader := newContentLengthReader(input.Body)
		sha := newSHA256Writer(dst)
		reader, err := encryptor.EncryptContents(lengthReader)
		if err != nil {
			r.Error = err
			return
		}

		_, err = io.Copy(sha, reader)
		if err != nil {
			r.Error = err
			return
		}

		data := encryptor.GetCipherData()
		env, err := encodeMeta(lengthReader, data)
		if err != nil {
			r.Error = err
			return
		}

		shaHex := hex.EncodeToString(sha.GetValue())
		req.HTTPRequest.Header.Set("X-Amz-Content-Sha256", shaHex)

		dst.Seek(0, sdkio.SeekStart)
		input.Body = dst

		err = c.SaveStrategy.Save(env, r)
		r.Error = err
	})

	return req, out
}

func putObject(options EncryptionClientOptions, input *s3.PutObjectInput) (*s3.PutObjectOutput, error) {
	req, out := putObjectRequest(options, input)
	return out, req.Send()
}

func putObjectWithContext(options EncryptionClientOptions, ctx aws.Context, input *s3.PutObjectInput, opts ...request.Option) (*s3.PutObjectOutput, error) {
	req, out := putObjectRequest(options, input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

func getObjectRequest(options DecryptionClientOptions, input *s3.GetObjectInput) (*request.Request, *s3.GetObjectOutput) {
	req, out := options.S3Client.GetObjectRequest(input)
	req.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		env, err := options.LoadStrategy.Load(r)
		if err != nil {
			r.Error = err
			out.Body.Close()
			return
		}

		// If KMS should return the correct cek algorithm with the proper
		// KMS key provider
		cipher, err := contentCipherFromEnvelope(options, r.Context(), env)
		if err != nil {
			r.Error = err
			out.Body.Close()
			return
		}

		reader, err := cipher.DecryptContents(out.Body)
		if err != nil {
			r.Error = err
			out.Body.Close()
			return
		}
		out.Body = reader
	})
	return req, out
}

func getObject(options DecryptionClientOptions, input *s3.GetObjectInput) (*s3.GetObjectOutput, error) {
	req, out := getObjectRequest(options, input)
	return out, req.Send()
}

func getObjectWithContext(options DecryptionClientOptions, ctx aws.Context, input *s3.GetObjectInput, opts ...request.Option) (*s3.GetObjectOutput, error) {
	req, out := getObjectRequest(options, input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

func contentCipherFromEnvelope(options DecryptionClientOptions, ctx aws.Context, env Envelope) (ContentCipher, error) {
	wrap, err := wrapFromEnvelope(options, env)
	if err != nil {
		return nil, err
	}

	return cekFromEnvelope(options, ctx, env, wrap)
}

func wrapFromEnvelope(options DecryptionClientOptions, env Envelope) (CipherDataDecrypter, error) {
	f, ok := options.CryptoRegistry.GetWrap(env.WrapAlg)
	if !ok || f == nil {

		return nil, awserr.New(
			"InvalidWrapAlgorithmError",
			"wrap algorithm isn't supported, "+env.WrapAlg,
			nil,
		)
	}
	return f(env)
}

func cekFromEnvelope(options DecryptionClientOptions, ctx aws.Context, env Envelope, decrypter CipherDataDecrypter) (ContentCipher, error) {
	f, ok := options.CryptoRegistry.GetCEK(env.CEKAlg)
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

	if d, ok := decrypter.(CipherDataDecrypterWithContext); ok {
		key, err = d.DecryptKeyWithContext(ctx, key)
	} else {
		key, err = decrypter.DecryptKey(key)
	}

	if err != nil {
		return nil, err
	}

	cd := CipherData{
		Key:          key,
		IV:           iv,
		CEKAlgorithm: env.CEKAlg,
		Padder:       getPadder(options, env.CEKAlg),
	}
	return f(cd)
}

// getPadder will return an unpadder with checking the cek algorithm specific padder.
// If there wasn't a cek algorithm specific padder, we check the padder itself.
// We return a no unpadder, if no unpadder was found. This means any customization
// either contained padding within the cipher implementation, and to maintain
// backwards compatibility we will simply not unpad anything.
func getPadder(options DecryptionClientOptions, cekAlg string) Padder {
	padder, ok := options.CryptoRegistry.GetPadder(cekAlg)
	if !ok {
		padder, ok = options.CryptoRegistry.GetPadder(cekAlg[strings.LastIndex(cekAlg, "/")+1:])
		if !ok {
			return NoPadder
		}
	}
	return padder
}
