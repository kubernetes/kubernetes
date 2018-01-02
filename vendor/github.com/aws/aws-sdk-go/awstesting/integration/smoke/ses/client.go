// +build integration

//Package ses provides gucumber integration tests support.
package ses

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/ses"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@ses", func() {
		gucumber.World["client"] = ses.New(smoke.Session)
	})
}
