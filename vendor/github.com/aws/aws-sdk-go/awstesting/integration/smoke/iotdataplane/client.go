// +build integration

//Package iotdataplane provides gucumber integration tests support.
package iotdataplane

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/iot"
	"github.com/aws/aws-sdk-go/service/iotdataplane"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@iotdataplane", func() {
		svc := iot.New(smoke.Session)
		result, err := svc.DescribeEndpoint(&iot.DescribeEndpointInput{})
		if err != nil {
			gucumber.World["error"] = err
			return
		}

		fmt.Println("IOT Data endpoint:", *result.EndpointAddress)
		gucumber.World["client"] = iotdataplane.New(smoke.Session, aws.NewConfig().
			WithEndpoint(*result.EndpointAddress))
	})
}
