// +build example

package main

import (
	"flag"
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

// Put an ACL on an S3 object
//
// Usage:
// putBucketAcl <params>
//	-region <region> // required
//	-bucket <bucket> // required
//	-key <key> // required
//	-owner-name <owner-name>
//	-owner-id <owner-id>
//	-grantee-type <some type> // required
//	-uri <uri to group>
//	-email <email address>
//	-user-id <user-id>
func main() {
	regionPtr := flag.String("region", "", "region of your request")
	bucketPtr := flag.String("bucket", "", "name of your bucket")
	keyPtr := flag.String("key", "", "of your object")
	ownerNamePtr := flag.String("owner-name", "", "of your request")
	ownerIDPtr := flag.String("owner-id", "", "of your request")
	granteeTypePtr := flag.String("grantee-type", "", "of your request")
	uriPtr := flag.String("uri", "", "of your grantee type")
	emailPtr := flag.String("email", "", "of your grantee type")
	userPtr := flag.String("user-id", "", "of your grantee type")
	displayNamePtr := flag.String("display-name", "", "of your grantee type")
	flag.Parse()

	// Based off the type, fields must be excluded.
	switch *granteeTypePtr {
	case s3.TypeCanonicalUser:
		emailPtr, uriPtr = nil, nil
		if *displayNamePtr == "" {
			displayNamePtr = nil
		}

		if *userPtr == "" {
			userPtr = nil
		}
	case s3.TypeAmazonCustomerByEmail:
		uriPtr, userPtr = nil, nil
	case s3.TypeGroup:
		emailPtr, userPtr = nil, nil
	}

	sess := session.Must(session.NewSession(&aws.Config{
		Region: regionPtr,
	}))

	svc := s3.New(sess)

	resp, err := svc.PutObjectAcl(&s3.PutObjectAclInput{
		Bucket: bucketPtr,
		Key:    keyPtr,
		AccessControlPolicy: &s3.AccessControlPolicy{
			Owner: &s3.Owner{
				DisplayName: ownerNamePtr,
				ID:          ownerIDPtr,
			},
			Grants: []*s3.Grant{
				{
					Grantee: &s3.Grantee{
						Type:         granteeTypePtr,
						DisplayName:  displayNamePtr,
						URI:          uriPtr,
						EmailAddress: emailPtr,
						ID:           userPtr,
					},
					Permission: aws.String(s3.BucketLogsPermissionFullControl),
				},
			},
		},
	})

	if err != nil {
		fmt.Println("failed", err)
	} else {
		fmt.Println("success", resp)
	}
}
