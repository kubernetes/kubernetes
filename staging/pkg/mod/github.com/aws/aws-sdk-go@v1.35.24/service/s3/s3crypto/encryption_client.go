package s3crypto

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

// DefaultMinFileSize is used to check whether we want to write to a temp file
// or store the data in memory.
const DefaultMinFileSize = 1024 * 512 * 5

// EncryptionClient is an S3 crypto client. By default the SDK will use Authentication mode which
// will use KMS for key wrapping and AES GCM for content encryption.
// AES GCM will load all data into memory. However, the rest of the content algorithms
// do not load the entire contents into memory.
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
type EncryptionClient struct {
	S3Client             s3iface.S3API
	ContentCipherBuilder ContentCipherBuilder
	// SaveStrategy will dictate where the envelope is saved.
	//
	// Defaults to the object's metadata
	SaveStrategy SaveStrategy
	// TempFolderPath is used to store temp files when calling PutObject.
	// Temporary files are needed to compute the X-Amz-Content-Sha256 header.
	TempFolderPath string
	// MinFileSize is the minimum size for the content to write to a
	// temporary file instead of using memory.
	MinFileSize int64
}

func validateV1EncryptionClientConstruction(c *EncryptionClient) error {
	builder, ok := c.ContentCipherBuilder.(compatibleEncryptionFixture)
	if !ok {
		return nil
	}

	err := builder.isEncryptionVersionCompatible(v1ClientVersion)
	if err != nil {
		return awserr.New(clientConstructionErrorCode, "invalid client configuration", err)
	}
	return nil
}

// NewEncryptionClient instantiates a new S3 crypto client
//
// Example:
//	cmkID := "arn:aws:kms:region:000000000000:key/00000000-0000-0000-0000-000000000000"
//  sess := session.Must(session.NewSession())
//	handler := s3crypto.NewKMSKeyGenerator(kms.New(sess), cmkID)
//	svc := s3crypto.NewEncryptionClient(sess, s3crypto.AESGCMContentCipherBuilder(handler))
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func NewEncryptionClient(prov client.ConfigProvider, builder ContentCipherBuilder, options ...func(*EncryptionClient)) *EncryptionClient {
	s3client := s3.New(prov)

	s3client.Handlers.Build.PushBack(func(r *request.Request) {
		request.AddToUserAgent(r, "S3CryptoV1n")
	})

	client := &EncryptionClient{
		S3Client:             s3client,
		ContentCipherBuilder: builder,
		SaveStrategy:         HeaderV2SaveStrategy{},
		MinFileSize:          DefaultMinFileSize,
	}

	for _, option := range options {
		option(client)
	}

	return client
}

// PutObjectRequest creates a temp file to encrypt the contents into. It then streams
// that data to S3.
//
// Example:
//	svc := s3crypto.NewEncryptionClient(session.Must(session.NewSession()), s3crypto.AESGCMContentCipherBuilder(handler))
//	req, out := svc.PutObjectRequest(&s3.PutObjectInput {
//	  Key: aws.String("testKey"),
//	  Bucket: aws.String("testBucket"),
//	  Body: strings.NewReader("test data"),
//	})
//	err := req.Send()
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func (c *EncryptionClient) PutObjectRequest(input *s3.PutObjectInput) (*request.Request, *s3.PutObjectOutput) {
	req, out := putObjectRequest(c.getClientOptions(), input)
	if err := validateV1EncryptionClientConstruction(c); err != nil {
		errHandler := setReqError(err)
		req.Error = err
		req.Handlers.Build.Clear()
		req.Handlers.Send.Clear()
		req.Handlers.Validate.PushFront(errHandler)
		req.Handlers.Build.PushFront(errHandler)
		req.Handlers.Send.PushFront(errHandler)
	}
	return req, out
}

func setReqError(err error) func(*request.Request) {
	return func(r *request.Request) {
		r.Error = err
	}
}

// PutObject is a wrapper for PutObjectRequest
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func (c *EncryptionClient) PutObject(input *s3.PutObjectInput) (*s3.PutObjectOutput, error) {
	if err := validateV1EncryptionClientConstruction(c); err != nil {
		return nil, err
	}
	return putObject(c.getClientOptions(), input)
}

// PutObjectWithContext is a wrapper for PutObjectRequest with the additional
// context, and request options support.
//
// PutObjectWithContext is the same as PutObject with the additional support for
// Context input parameters. The Context must not be nil. A nil Context will
// cause a panic. Use the Context to add deadlining, timeouts, etc. In the future
// this may create sub-contexts for individual underlying requests.
// PutObject is a wrapper for PutObjectRequest
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func (c *EncryptionClient) PutObjectWithContext(ctx aws.Context, input *s3.PutObjectInput, opts ...request.Option) (*s3.PutObjectOutput, error) {
	if err := validateV1EncryptionClientConstruction(c); err != nil {
		return nil, err
	}
	return putObjectWithContext(c.getClientOptions(), ctx, input, opts...)
}

func (c *EncryptionClient) getClientOptions() EncryptionClientOptions {
	return EncryptionClientOptions{
		S3Client:             c.S3Client,
		ContentCipherBuilder: c.ContentCipherBuilder,
		SaveStrategy:         c.SaveStrategy,
		TempFolderPath:       c.TempFolderPath,
		MinFileSize:          c.MinFileSize,
	}
}
