// +build integration

//Package cloudfront provides gucumber integration tests support.
package cloudfront

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudfront"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@cloudfront", func() {
		gucumber.World["client"] = cloudfront.New(smoke.Session)
	})
}
