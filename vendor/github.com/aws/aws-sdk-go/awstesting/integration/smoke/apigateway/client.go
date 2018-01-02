// +build integration

//Package apigateway provides gucumber integration tests support.
package apigateway

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/apigateway"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@apigateway", func() {
		gucumber.World["client"] = apigateway.New(smoke.Session)
	})
}
