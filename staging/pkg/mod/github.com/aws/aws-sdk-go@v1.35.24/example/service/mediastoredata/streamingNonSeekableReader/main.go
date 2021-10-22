// +build example

package main

import (
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/mediastore"
	"github.com/aws/aws-sdk-go/service/mediastoredata"
)

func main() {
	containerName := os.Args[1]
	objectPath := os.Args[2]

	// Create the SDK's session, and a AWS Elemental MediaStore Data client.
	sess := session.Must(session.NewSession())
	dataSvc, err := getMediaStoreDataClient(containerName, sess)
	if err != nil {
		log.Fatalf("failed to create client, %v", err)
	}

	// Create a random reader to simulate a unseekable reader, wrap the reader
	// in an io.LimitReader to prevent uploading forever.
	randReader := rand.New(rand.NewSource(0))
	reader := io.LimitReader(randReader, 1024*1024 /* 1MB */)

	// Wrap the unseekable reader with the SDK's RandSeekCloser. This type will
	// allow the SDK's to use the nonseekable reader.
	body := aws.ReadSeekCloser(reader)

	// make the PutObject API call with the nonseekable reader, causing the SDK
	// to send the request body payload as chunked transfer encoding.
	_, err = dataSvc.PutObject(&mediastoredata.PutObjectInput{
		Path: &objectPath,
		Body: body,
	})
	if err != nil {
		log.Fatalf("failed to upload object, %v", err)
	}

	fmt.Println("object uploaded")
}

// getMediaStoreDataClient uses the AWS Elemental MediaStore API to get the
// endpoint for a container. If the container endpoint can be retrieved a AWS
// Elemental MediaStore Data client will be created and returned. Otherwise
// error is returned.
func getMediaStoreDataClient(containerName string, sess *session.Session) (*mediastoredata.MediaStoreData, error) {
	endpoint, err := containerEndpoint(containerName, sess)
	if err != nil {
		return nil, err
	}

	dataSvc := mediastoredata.New(sess, &aws.Config{
		Endpoint: endpoint,
	})

	return dataSvc, nil
}

// ContainerEndpoint will attempt to get the endpoint for a container,
// returning error if the container doesn't exist, or is not active within a
// timeout.
func containerEndpoint(name string, sess *session.Session) (*string, error) {
	for i := 0; i < 3; i++ {
		ctrlSvc := mediastore.New(sess)
		descResp, err := ctrlSvc.DescribeContainer(&mediastore.DescribeContainerInput{
			ContainerName: &name,
		})
		if err != nil {
			return nil, err
		}

		if status := aws.StringValue(descResp.Container.Status); status != "ACTIVE" {
			log.Println("waiting for container to be active, ", status)
			time.Sleep(10 * time.Second)
			continue
		}

		return descResp.Container.Endpoint, nil
	}

	return nil, fmt.Errorf("container is not active")
}
