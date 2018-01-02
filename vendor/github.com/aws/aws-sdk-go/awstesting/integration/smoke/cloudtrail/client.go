// +build integration

//Package cloudtrail provides gucumber integration tests support.
package cloudtrail

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudtrail"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@cloudtrail", func() {
		gucumber.World["client"] = cloudtrail.New(smoke.Session)
	})
}
