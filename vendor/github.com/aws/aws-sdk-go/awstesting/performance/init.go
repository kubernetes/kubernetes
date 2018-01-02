// +build integration

package performance

import (
	"bytes"
	"errors"
	"fmt"
	"runtime"

	"github.com/gucumber/gucumber"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/awstesting/mock"
	"github.com/aws/aws-sdk-go/service/s3"
)

func init() {
	// Go loads all of its dependecies on compile
	gucumber.Given(`^I have loaded my SDK and its dependencies$`, func() {
	})

	// Performance
	gucumber.When(`^I create and discard (\d+) clients for each service$`, func(i1 int) {
		services := gucumber.World["services"].([]func())
		err := benchmarkTask(fmt.Sprintf("%d_create_and_discard_clients", i1), services, i1)
		gucumber.World["error"] = err
	})

	gucumber.Then(`^I should not have leaked any resources$`, func() {
		runtime.GC()
		err, ok := gucumber.World["error"].(awserr.Error)
		if ok {
			gucumber.T.Errorf("error returned")
		}
		if err != nil {
			gucumber.T.Errorf("expect no error, got %v", err)
		}
	})

	gucumber.And(`^I have a list of services$`, func() {
		mapCreateClients()
	})

	gucumber.And(`^I take a snapshot of my resources$`, func() {
		// Can't take a memory snapshot here, because gucumber does some
		// allocation between each instruction leading to unreliable numbers
	})

	gucumber.When(`^I create a client for each service$`, func() {
		buildAnArrayOfClients()
	})

	gucumber.And("^I execute (\\d+) command\\(s\\) on each client$", func(i1 int) {
		clientFns := gucumber.World["clientFns"].([]func())
		err := benchmarkTask(fmt.Sprintf("%d_commands_on_clients", i1), clientFns, i1)
		gucumber.World["error"] = err
	})

	gucumber.And(`^I destroy all the clients$`, func() {
		delete(gucumber.World, "clientFns")
		runtime.GC()
	})

	gucumber.Given(`^I have a (\d+) byte file$`, func(i1 int) {
		gucumber.World["file"] = make([]byte, i1)
	})

	gucumber.When(`^I upload the file$`, func() {
		svc := s3.New(mock.Session)
		memStatStart := &runtime.MemStats{}
		runtime.ReadMemStats(memStatStart)
		gucumber.World["start"] = memStatStart

		svc.PutObjectRequest(&s3.PutObjectInput{
			Bucket: aws.String("bucketmesilly"),
			Key:    aws.String("testKey"),
			Body:   bytes.NewReader(gucumber.World["file"].([]byte)),
		})
	})

	gucumber.And(`then download the file$`, func() {
		svc := s3.New(mock.Session)
		svc.GetObjectRequest(&s3.GetObjectInput{
			Bucket: aws.String("bucketmesilly"),
			Key:    aws.String("testKey"),
		})
		memStatEnd := &runtime.MemStats{}
		runtime.ReadMemStats(memStatEnd)
		memStatStart := gucumber.World["start"].(*runtime.MemStats)
		if memStatStart.Alloc < memStatEnd.Alloc {
			gucumber.World["error"] = errors.New("Leaked memory")
		}
	})
}
