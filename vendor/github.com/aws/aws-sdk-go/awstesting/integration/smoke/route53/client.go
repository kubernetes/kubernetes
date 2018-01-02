// +build integration

//Package route53 provides gucumber integration tests support.
package route53

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/route53"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@route53", func() {
		gucumber.World["client"] = route53.New(smoke.Session)
	})
}
