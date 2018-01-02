// +build integration

//Package s3crypto provides gucumber integration tests support.
package s3crypto

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3crypto"

	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@s3crypto", func() {
		sess := session.New((&aws.Config{
			Region: aws.String("us-west-2"),
		}))
		encryptionClient := s3crypto.NewEncryptionClient(sess, nil, func(c *s3crypto.EncryptionClient) {
		})
		gucumber.World["encryptionClient"] = encryptionClient

		decryptionClient := s3crypto.NewDecryptionClient(sess)
		gucumber.World["decryptionClient"] = decryptionClient

		gucumber.World["client"] = s3.New(sess)
	})
}
