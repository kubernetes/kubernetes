package s3crypto

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

// WrapEntry is builder that return a proper key decrypter and error
type WrapEntry func(Envelope) (CipherDataDecrypter, error)

// CEKEntry is a builder that returns a proper content decrypter and error
type CEKEntry func(CipherData) (ContentCipher, error)

// DecryptionClient is an S3 crypto client. The decryption client
// will handle all get object requests from Amazon S3.
// Supported key wrapping algorithms:
//	*AWS KMS
//
// Supported content ciphers:
//	* AES/GCM
//	* AES/CBC
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
type DecryptionClient struct {
	S3Client s3iface.S3API
	// LoadStrategy is used to load the metadata either from the metadata of the object
	// or from a separate file in s3.
	//
	// Defaults to our default load strategy.
	LoadStrategy LoadStrategy

	WrapRegistry   map[string]WrapEntry
	CEKRegistry    map[string]CEKEntry
	PadderRegistry map[string]Padder
}

// NewDecryptionClient instantiates a new S3 crypto client
//
// Example:
//	sess := session.Must(session.NewSession())
//	svc := s3crypto.NewDecryptionClient(sess, func(svc *s3crypto.DecryptionClient{
//		// Custom client options here
//	}))
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func NewDecryptionClient(prov client.ConfigProvider, options ...func(*DecryptionClient)) *DecryptionClient {
	s3client := s3.New(prov)

	s3client.Handlers.Build.PushBack(func(r *request.Request) {
		request.AddToUserAgent(r, "S3CryptoV1n")
	})

	kmsClient := kms.New(prov)
	client := &DecryptionClient{
		S3Client: s3client,
		LoadStrategy: defaultV2LoadStrategy{
			client: s3client,
		},
		WrapRegistry: map[string]WrapEntry{
			KMSWrap:        NewKMSWrapEntry(kmsClient),
			KMSContextWrap: newKMSContextWrapEntryWithAnyCMK(kmsClient),
		},
		CEKRegistry: map[string]CEKEntry{
			AESGCMNoPadding:                    newAESGCMContentCipher,
			AESCBC + "/" + AESCBCPadder.Name(): newAESCBCContentCipher,
		},
		PadderRegistry: map[string]Padder{
			AESCBC + "/" + AESCBCPadder.Name(): AESCBCPadder,
			NoPadder.Name():                    NoPadder,
		},
	}
	for _, option := range options {
		option(client)
	}

	return client
}

// GetObjectRequest will make a request to s3 and retrieve the object. In this process
// decryption will be done. The SDK only supports V2 reads of KMS and GCM.
//
// Example:
//  sess := session.Must(session.NewSession())
//	svc := s3crypto.NewDecryptionClient(sess)
//	req, out := svc.GetObjectRequest(&s3.GetObjectInput {
//	  Key: aws.String("testKey"),
//	  Bucket: aws.String("testBucket"),
//	})
//	err := req.Send()
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func (c *DecryptionClient) GetObjectRequest(input *s3.GetObjectInput) (*request.Request, *s3.GetObjectOutput) {
	return getObjectRequest(c.getClientOptions(), input)
}

// GetObject is a wrapper for GetObjectRequest
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func (c *DecryptionClient) GetObject(input *s3.GetObjectInput) (*s3.GetObjectOutput, error) {
	return getObject(c.getClientOptions(), input)
}

// GetObjectWithContext is a wrapper for GetObjectRequest with the additional
// context, and request options support.
//
// GetObjectWithContext is the same as GetObject with the additional support for
// Context input parameters. The Context must not be nil. A nil Context will
// cause a panic. Use the Context to add deadlining, timeouts, etc. In the future
// this may create sub-contexts for individual underlying requests.
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func (c *DecryptionClient) GetObjectWithContext(ctx aws.Context, input *s3.GetObjectInput, opts ...request.Option) (*s3.GetObjectOutput, error) {
	return getObjectWithContext(c.getClientOptions(), ctx, input, opts...)
}

func (c *DecryptionClient) getClientOptions() DecryptionClientOptions {
	return DecryptionClientOptions{
		S3Client:       c.S3Client,
		LoadStrategy:   c.LoadStrategy,
		CryptoRegistry: initCryptoRegistryFrom(c.WrapRegistry, c.CEKRegistry, c.PadderRegistry),
	}
}
