// +build integration

//Package cloudwatch provides gucumber integration tests support.
package cloudwatch

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudwatch"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@cloudwatch", func() {
		gucumber.World["client"] = cloudwatch.New(smoke.Session)
	})
}
