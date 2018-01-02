// +build integration

//Package rds provides gucumber integration tests support.
package rds

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/rds"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@rds", func() {
		gucumber.World["client"] = rds.New(smoke.Session)
	})
}
