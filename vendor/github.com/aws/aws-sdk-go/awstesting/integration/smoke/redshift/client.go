// +build integration

//Package redshift provides gucumber integration tests support.
package redshift

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/redshift"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@redshift", func() {
		gucumber.World["client"] = redshift.New(smoke.Session)
	})
}
