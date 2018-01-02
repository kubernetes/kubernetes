// +build integration

//Package cloudsearch provides gucumber integration tests support.
package cloudsearch

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudsearch"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@cloudsearch", func() {
		gucumber.World["client"] = cloudsearch.New(smoke.Session)
	})
}
