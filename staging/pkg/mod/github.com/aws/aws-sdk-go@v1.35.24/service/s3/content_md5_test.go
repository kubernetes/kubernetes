package s3_test

import (
	"crypto/md5"
	"encoding/base64"
	"io/ioutil"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func assertMD5(t *testing.T, req *request.Request) {
	err := req.Build()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	b, _ := ioutil.ReadAll(req.HTTPRequest.Body)
	out := md5.Sum(b)
	if len(b) == 0 {
		t.Error("expected non-empty value")
	}
	if a := req.HTTPRequest.Header.Get("Content-MD5"); len(a) == 0 {
		t.Fatal("Expected Content-MD5 header to be present in the operation request, was not")
	} else if e := base64.StdEncoding.EncodeToString(out[:]); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestMD5InPutBucketCors(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketCorsRequest(&s3.PutBucketCorsInput{
		Bucket: aws.String("bucketname"),
		CORSConfiguration: &s3.CORSConfiguration{
			CORSRules: []*s3.CORSRule{
				{
					AllowedMethods: []*string{aws.String("GET")},
					AllowedOrigins: []*string{aws.String("*")},
				},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketLifecycle(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketLifecycleRequest(&s3.PutBucketLifecycleInput{
		Bucket: aws.String("bucketname"),
		LifecycleConfiguration: &s3.LifecycleConfiguration{
			Rules: []*s3.Rule{
				{
					ID:     aws.String("ID"),
					Prefix: aws.String("Prefix"),
					Status: aws.String("Enabled"),
				},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketPolicy(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketPolicyRequest(&s3.PutBucketPolicyInput{
		Bucket: aws.String("bucketname"),
		Policy: aws.String("{}"),
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketTagging(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketTaggingRequest(&s3.PutBucketTaggingInput{
		Bucket: aws.String("bucketname"),
		Tagging: &s3.Tagging{
			TagSet: []*s3.Tag{
				{Key: aws.String("KEY"), Value: aws.String("VALUE")},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InDeleteObjects(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.DeleteObjectsRequest(&s3.DeleteObjectsInput{
		Bucket: aws.String("bucketname"),
		Delete: &s3.Delete{
			Objects: []*s3.ObjectIdentifier{
				{Key: aws.String("key")},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketLifecycleConfiguration(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketLifecycleConfigurationRequest(&s3.PutBucketLifecycleConfigurationInput{
		Bucket: aws.String("bucketname"),
		LifecycleConfiguration: &s3.BucketLifecycleConfiguration{
			Rules: []*s3.LifecycleRule{
				{Prefix: aws.String("prefix"), Status: aws.String(s3.ExpirationStatusEnabled)},
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketReplication(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketReplicationRequest(&s3.PutBucketReplicationInput{
		Bucket: aws.String("bucketname"),
		ReplicationConfiguration: &s3.ReplicationConfiguration{
			Role: aws.String("Role"),
			Rules: []*s3.ReplicationRule{
				{
					Destination: &s3.Destination{
						Bucket: aws.String("mock bucket"),
					},
					Status: aws.String(s3.ReplicationRuleStatusDisabled),
				},
			},
		},
		Token: aws.String("token"),
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketAcl(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketAclRequest(&s3.PutBucketAclInput{
		Bucket: aws.String("bucketname"),
		AccessControlPolicy: &s3.AccessControlPolicy{
			Grants: []*s3.Grant{{
				Grantee: &s3.Grantee{
					ID:   aws.String("mock id"),
					Type: aws.String("type"),
				},
				Permission: aws.String(s3.PermissionFullControl),
			}},
			Owner: &s3.Owner{
				DisplayName: aws.String("mock name"),
			},
		},
	})
	assertMD5(t, req)
}

func TestMD5InPutBucketEncryption(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketEncryptionRequest(&s3.PutBucketEncryptionInput{
		Bucket: aws.String("bucketname"),
		ServerSideEncryptionConfiguration: &s3.ServerSideEncryptionConfiguration{
			Rules: []*s3.ServerSideEncryptionRule{
				{
					ApplyServerSideEncryptionByDefault: &s3.ServerSideEncryptionByDefault{
						KMSMasterKeyID: aws.String("mock KMS master key id"),
						SSEAlgorithm:   aws.String("mock SSE algorithm"),
					},
				},
			},
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutBucketLogging(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketLoggingRequest(&s3.PutBucketLoggingInput{
		Bucket: aws.String("bucket name"),
		BucketLoggingStatus: &s3.BucketLoggingStatus{LoggingEnabled: &s3.LoggingEnabled{
			TargetBucket: aws.String("target bucket"),
			TargetPrefix: aws.String("target prefix"),
		}},
	})

	assertMD5(t, req)
}

func TestMD5InPutBucketNotification(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketNotificationRequest(&s3.PutBucketNotificationInput{
		Bucket: aws.String("bucket name"),
		NotificationConfiguration: &s3.NotificationConfigurationDeprecated{
			TopicConfiguration: &s3.TopicConfigurationDeprecated{
				Id: aws.String("id"),
			},
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutBucketRequestPayment(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketRequestPaymentRequest(&s3.PutBucketRequestPaymentInput{
		Bucket: aws.String("bucketname"),
		RequestPaymentConfiguration: &s3.RequestPaymentConfiguration{
			Payer: aws.String("payer"),
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutBucketVersioning(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketVersioningRequest(&s3.PutBucketVersioningInput{
		Bucket: aws.String("bucketname"),
		VersioningConfiguration: &s3.VersioningConfiguration{
			MFADelete: aws.String(s3.MFADeleteDisabled),
			Status:    aws.String(s3.BucketVersioningStatusSuspended),
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutBucketWebsite(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutBucketWebsiteRequest(&s3.PutBucketWebsiteInput{
		Bucket: aws.String("bucket name"),
		WebsiteConfiguration: &s3.WebsiteConfiguration{
			ErrorDocument: &s3.ErrorDocument{
				Key: aws.String("error"),
			},
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutObjectLegalHold(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectLegalHoldRequest(&s3.PutObjectLegalHoldInput{
		Bucket: aws.String("bucketname"),
		Key:    aws.String("key"),
		LegalHold: &s3.ObjectLockLegalHold{
			Status: aws.String(s3.ObjectLockLegalHoldStatusOff),
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutObjectRetention(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectRetentionRequest(&s3.PutObjectRetentionInput{
		Bucket:                    aws.String("bucket name"),
		BypassGovernanceRetention: nil,
		Key:                       aws.String("key"),
		RequestPayer:              nil,
		Retention: &s3.ObjectLockRetention{
			Mode: aws.String("mode"),
		},
		VersionId: nil,
	})

	assertMD5(t, req)
}

func TestMD5InPutObjectLockConfiguration(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectLockConfigurationRequest(&s3.PutObjectLockConfigurationInput{
		Bucket: aws.String("bucket name"),
		ObjectLockConfiguration: &s3.ObjectLockConfiguration{
			ObjectLockEnabled: aws.String(s3.ObjectLockEnabledEnabled),
		},
	})

	assertMD5(t, req)
}

func TestMD5InPutObjectAcl(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectAclRequest(&s3.PutObjectAclInput{
		AccessControlPolicy: &s3.AccessControlPolicy{
			Grants: []*s3.Grant{{
				Grantee: &s3.Grantee{
					ID:   aws.String("mock id"),
					Type: aws.String("type"),
				},
				Permission: aws.String(s3.PermissionFullControl),
			}},
			Owner: &s3.Owner{
				DisplayName: aws.String("mock name"),
			},
		},
		Bucket: aws.String("bucket name"),
		Key:    aws.String("key"),
	})

	assertMD5(t, req)
}

func TestMD5InPutObjectTagging(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutObjectTaggingRequest(&s3.PutObjectTaggingInput{
		Bucket: aws.String("bucket name"),
		Key:    aws.String("key"),
		Tagging: &s3.Tagging{TagSet: []*s3.Tag{
			{
				Key:   aws.String("key"),
				Value: aws.String("value"),
			},
		}},
	})

	assertMD5(t, req)
}

func TestMD5InPutPublicAccessBlock(t *testing.T) {
	svc := s3.New(unit.Session)
	req, _ := svc.PutPublicAccessBlockRequest(&s3.PutPublicAccessBlockInput{
		Bucket: aws.String("bucket name"),
		PublicAccessBlockConfiguration: &s3.PublicAccessBlockConfiguration{
			BlockPublicAcls:       aws.Bool(true),
			BlockPublicPolicy:     aws.Bool(true),
			IgnorePublicAcls:      aws.Bool(true),
			RestrictPublicBuckets: aws.Bool(true),
		},
	})

	assertMD5(t, req)
}
