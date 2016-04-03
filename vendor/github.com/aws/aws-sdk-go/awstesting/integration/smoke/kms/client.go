//Package kms provides gucumber integration tests support.
package kms

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/kms"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@kms", func() {
		World["client"] = kms.New(smoke.Session)
	})
}
