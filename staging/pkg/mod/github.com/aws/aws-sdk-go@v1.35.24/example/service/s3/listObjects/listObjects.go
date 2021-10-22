// +build example

package main

import (
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

// Lists all objects in a bucket using pagination
//
// Usage:
// listObjects <bucket>
func main() {
	if len(os.Args) < 2 {
		fmt.Println("you must specify a bucket")
		return
	}

	sess := session.Must(session.NewSession())

	svc := s3.New(sess)

	i := 0
	err := svc.ListObjectsPages(&s3.ListObjectsInput{
		Bucket: &os.Args[1],
	}, func(p *s3.ListObjectsOutput, last bool) (shouldContinue bool) {
		fmt.Println("Page,", i)
		i++

		for _, obj := range p.Contents {
			fmt.Println("Object:", *obj.Key)
		}
		return true
	})
	if err != nil {
		fmt.Println("failed to list objects", err)
		return
	}
}
