package s3crypto

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

// DecryptionClientV2 is an S3 crypto client. The decryption client
// will handle all get object requests from Amazon S3.
// Supported key wrapping algorithms:
//	* AWS KMS
//	* AWS KMS + Context
//
// Supported content ciphers:
//	* AES/GCM
//	* AES/CBC
type DecryptionClientV2 struct {
	options DecryptionClientOptions
}

// DecryptionClientOptions is the configuration options for DecryptionClientV2.
type DecryptionClientOptions struct {
	S3Client s3iface.S3API
	// LoadStrategy is used to load the metadata either from the metadata of the object
	// or from a separate file in s3.
	//
	// Defaults to our default load strategy.
	LoadStrategy LoadStrategy

	CryptoRegistry *CryptoRegistry
}

// NewDecryptionClientV2 instantiates a new DecryptionClientV2. The NewDecryptionClientV2 must be configured with the
// desired key wrapping and content encryption algorithms that are required to be read by the client. These algorithms
// are registered by providing the client a CryptoRegistry that has been constructed with the desired configuration.
// NewDecryptionClientV2 will return an error if no key wrapping or content encryption algorithms have been provided.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cr := s3crypto.NewCryptoRegistry()
// 	if err := s3crypto.RegisterKMSContextWrapWithAnyCMK(cr, kms.New(sess)); err != nil {
//		panic(err) // handle error
//	}
//	if err := s3crypto.RegisterAESGCMContentCipher(cr); err != nil {
//		panic(err) // handle error
//	}
//	svc, err := s3crypto.NewDecryptionClientV2(sess, cr, func(o *s3crypto.DecryptionClientOptions) {
//		// Custom client options here
//	})
//	if err != nil {
//		panic(err) // handle error
//	}
func NewDecryptionClientV2(
	prov client.ConfigProvider, cryptoRegistry *CryptoRegistry,
	options ...func(clientOptions *DecryptionClientOptions),
) (*DecryptionClientV2, error) {
	s3client := s3.New(prov)

	s3client.Handlers.Build.PushBack(func(r *request.Request) {
		request.AddToUserAgent(r, "S3CryptoV2")
	})

	clientOptions := &DecryptionClientOptions{
		S3Client: s3client,
		LoadStrategy: defaultV2LoadStrategy{
			client: s3client,
		},
		CryptoRegistry: cryptoRegistry,
	}
	for _, option := range options {
		option(clientOptions)
	}

	if err := cryptoRegistry.valid(); err != nil {
		return nil, err
	}

	return &DecryptionClientV2{options: *clientOptions}, nil
}

// GetObjectRequest will make a request to s3 and retrieve the object. In this process
// decryption will be done. The SDK only supports V2 reads of KMS and GCM.
//
// Example:
//	req, out := svc.GetObjectRequest(&s3.GetObjectInput {
//	  Key: aws.String("testKey"),
//	  Bucket: aws.String("testBucket"),
//	})
//	err := req.Send()
func (c *DecryptionClientV2) GetObjectRequest(input *s3.GetObjectInput) (*request.Request, *s3.GetObjectOutput) {
	return getObjectRequest(c.options, input)
}

// GetObject is a wrapper for GetObjectRequest
func (c *DecryptionClientV2) GetObject(input *s3.GetObjectInput) (*s3.GetObjectOutput, error) {
	req, out := getObjectRequest(c.options, input)
	return out, req.Send()
}

// GetObjectWithContext is a wrapper for GetObjectRequest with the additional
// context, and request options support.
//
// GetObjectWithContext is the same as GetObject with the additional support for
// Context input parameters. The Context must not be nil. A nil Context will
// cause a panic. Use the Context to add deadlining, timeouts, etc. In the future
// this may create sub-contexts for individual underlying requests.
func (c *DecryptionClientV2) GetObjectWithContext(ctx aws.Context, input *s3.GetObjectInput, opts ...request.Option) (*s3.GetObjectOutput, error) {
	req, out := getObjectRequest(c.options, input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}
