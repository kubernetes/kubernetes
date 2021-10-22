package s3crypto

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

const customTypeWarningMessage = "WARNING: The S3 Encryption Client is configured to write encrypted objects using types not provided by AWS. Security and compatibility with these types can not be guaranteed."

// EncryptionClientV2 is an S3 crypto client. By default the SDK will use Authentication mode which
// will use KMS for key wrapping and AES GCM for content encryption.
// AES GCM will load all data into memory. However, the rest of the content algorithms
// do not load the entire contents into memory.
type EncryptionClientV2 struct {
	options EncryptionClientOptions
}

// EncryptionClientOptions is the configuration options for EncryptionClientV2
type EncryptionClientOptions struct {
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

// NewEncryptionClientV2 instantiates a new S3 crypto client. An error will be returned to the caller if the provided
// contentCipherBuilder has been deprecated or was constructed with a deprecated component.
//
// Example:
//	cmkID := "arn:aws:kms:region:000000000000:key/00000000-0000-0000-0000-000000000000"
//  sess := session.Must(session.NewSession())
//	var matdesc s3crypto.MaterialDescription
//	handler := s3crypto.NewKMSContextKeyGenerator(kms.New(sess), cmkID, matdesc)
//	svc := s3crypto.NewEncryptionClientV2(sess, s3crypto.AESGCMContentCipherBuilderV2(handler))
func NewEncryptionClientV2(prov client.ConfigProvider, contentCipherBuilder ContentCipherBuilder, options ...func(clientOptions *EncryptionClientOptions),
) (
	client *EncryptionClientV2, err error,
) {
	s3client := s3.New(prov)
	s3client.Handlers.Build.PushBack(func(r *request.Request) {
		request.AddToUserAgent(r, "S3CryptoV2")
	})

	clientOptions := &EncryptionClientOptions{
		S3Client:             s3client,
		ContentCipherBuilder: contentCipherBuilder,
		SaveStrategy:         HeaderV2SaveStrategy{},
		MinFileSize:          DefaultMinFileSize,
	}
	for _, option := range options {
		option(clientOptions)
	}

	// Check that the configured client uses a compatible ContentCipherBuilder.
	// User provided types will not implement this method
	if fixture, ok := contentCipherBuilder.(compatibleEncryptionFixture); ok {
		if err := fixture.isEncryptionVersionCompatible(v2ClientVersion); err != nil {
			return nil, err
		}
	}

	// Check if the passed in type is an fixture, if not log a warning message to the user
	if fixture, ok := contentCipherBuilder.(awsFixture); !ok || !fixture.isAWSFixture() {
		if s3client.Config.Logger != nil {
			s3client.Config.Logger.Log(customTypeWarningMessage)
		}
	}

	client = &EncryptionClientV2{
		*clientOptions,
	}

	return client, err
}

// PutObjectRequest creates a temp file to encrypt the contents into. It then streams
// that data to S3.
//
// Example:
//	req, out := svc.PutObjectRequest(&s3.PutObjectInput {
//	  Key: aws.String("testKey"),
//	  Bucket: aws.String("testBucket"),
//	  Body: strings.NewReader("test data"),
//	})
//	err := req.Send()
func (c *EncryptionClientV2) PutObjectRequest(input *s3.PutObjectInput) (*request.Request, *s3.PutObjectOutput) {
	return putObjectRequest(c.options, input)
}

// PutObject is a wrapper for PutObjectRequest
func (c *EncryptionClientV2) PutObject(input *s3.PutObjectInput) (*s3.PutObjectOutput, error) {
	return putObject(c.options, input)
}

// PutObjectWithContext is a wrapper for PutObjectRequest with the additional
// context, and request options support.
//
// PutObjectWithContext is the same as PutObject with the additional support for
// Context input parameters. The Context must not be nil. A nil Context will
// cause a panic. Use the Context to add deadlining, timeouts, etc. In the future
// this may create sub-contexts for individual underlying requests.
func (c *EncryptionClientV2) PutObjectWithContext(ctx aws.Context, input *s3.PutObjectInput, opts ...request.Option) (*s3.PutObjectOutput, error) {
	return putObjectWithContext(c.options, ctx, input, opts...)
}
