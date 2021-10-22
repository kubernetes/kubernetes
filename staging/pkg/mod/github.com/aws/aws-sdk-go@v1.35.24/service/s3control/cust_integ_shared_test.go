// +build integration

package s3control_test

import (
	"crypto/tls"
	"flag"
	"fmt"
	"net/http"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/s3control"
	"github.com/aws/aws-sdk-go/service/sts"
)

var (
	svc                            *s3control.S3Control
	s3ControlEndpoint, stsEndpoint string
	accountID                      string
	insecureTLS, useDualstack      bool
)

func init() {
	flag.StringVar(&stsEndpoint, "sts-endpoint", "",
		"The optional `URL` endpoint for the STS service.",
	)
	flag.StringVar(&s3ControlEndpoint, "s3-control-endpoint", "",
		"The optional `URL` endpoint for the S3 Control service.",
	)
	flag.BoolVar(&insecureTLS, "insecure-tls", false,
		"Disables TLS validation on request endpoints.",
	)
	flag.BoolVar(&useDualstack, "dualstack", false,
		"Enables usage of dualstack endpoints.",
	)
	flag.StringVar(&accountID, "account", "",
		"The AWS account `ID`.",
	)
}

func TestMain(m *testing.M) {
	setup()
	flag.Parse()
	os.Exit(m.Run())
}

// Create a bucket for testing
func setup() {
	tlsCfg := &tls.Config{}
	if insecureTLS {
		tlsCfg.InsecureSkipVerify = true
	}

	sess := integration.SessionWithDefaultRegion("us-west-2")
	sess.Copy(&aws.Config{
		HTTPClient: &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: tlsCfg,
			},
		},
		UseDualStack: aws.Bool(useDualstack),
	})

	if len(accountID) == 0 {
		stsSvc := sts.New(sess, &aws.Config{
			Endpoint: &stsEndpoint,
		})
		identity, err := stsSvc.GetCallerIdentity(&sts.GetCallerIdentityInput{})
		if err != nil {
			panic(fmt.Sprintf("failed to get accountID, %v", err))
		}
		accountID = aws.StringValue(identity.Account)
	}

	svc = s3control.New(sess, &aws.Config{
		Endpoint: &s3ControlEndpoint,
	})
}
