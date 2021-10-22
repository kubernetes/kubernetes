package s3

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
)

func ExampleS3_SelectObjectContent() {
	sess := session.Must(session.NewSession())
	svc := New(sess)

	/*
	   Example myObjectKey CSV content:

	   name,number
	   gopher,0
	   ᵷodɥǝɹ,1
	*/

	// Make the Select Object Content API request using the object uploaded.
	resp, err := svc.SelectObjectContent(&SelectObjectContentInput{
		Bucket:         aws.String("myBucket"),
		Key:            aws.String("myObjectKey"),
		Expression:     aws.String("SELECT name FROM S3Object WHERE cast(number as int) < 1"),
		ExpressionType: aws.String(ExpressionTypeSql),
		InputSerialization: &InputSerialization{
			CSV: &CSVInput{
				FileHeaderInfo: aws.String(FileHeaderInfoUse),
			},
		},
		OutputSerialization: &OutputSerialization{
			CSV: &CSVOutput{},
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed making API request, %v\n", err)
		return
	}
	defer resp.EventStream.Close()

	results, resultWriter := io.Pipe()
	go func() {
		defer resultWriter.Close()
		for event := range resp.EventStream.Events() {
			switch e := event.(type) {
			case *RecordsEvent:
				resultWriter.Write(e.Payload)
			case *StatsEvent:
				fmt.Printf("Processed %d bytes\n", *e.Details.BytesProcessed)
			}
		}
	}()

	// Printout the results
	resReader := csv.NewReader(results)
	for {
		record, err := resReader.Read()
		if err == io.EOF {
			break
		}
		fmt.Println(record)
	}

	if err := resp.EventStream.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "reading from event stream failed, %v\n", err)
	}
}
