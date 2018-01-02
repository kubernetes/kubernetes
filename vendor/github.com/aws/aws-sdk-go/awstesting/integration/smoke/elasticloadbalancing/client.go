// +build integration

//Package elasticloadbalancing provides gucumber integration tests support.
package elasticloadbalancing

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@elasticloadbalancing", func() {
		gucumber.World["client"] = elb.New(smoke.Session)
	})
}
