// +build integration

//Package cognitoidentity provides gucumber integration tests support.
package cognitoidentity

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cognitoidentity"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@cognitoidentity", func() {
		gucumber.World["client"] = cognitoidentity.New(smoke.Session)
	})
}
