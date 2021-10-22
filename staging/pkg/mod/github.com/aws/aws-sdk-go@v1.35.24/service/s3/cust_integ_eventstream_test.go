// +build integration

package s3_test

import (
	"bytes"
	"encoding/csv"
	"io"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/internal/sdkio"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestInteg_SelectObjectContent(t *testing.T) {
	keyName := "selectObject.csv"

	var header = []byte("A,B,C,D,E,F,G,H,I,J\n")
	var recordRow = []byte("0,0,0.5,217.371,217.658,218.002,269.445,487.447,2.106,489.554\n")

	buf := make([]byte, 0, 6*sdkio.MebiByte)
	buf = append(buf, []byte(header)...)
	for i := 0; i < (cap(buf)/len(recordRow))-1; i++ {
		buf = append(buf, recordRow...)
	}

	// Put a mock CSV file to the S3 bucket so that its contents can be
	// selected.
	putTestContent(t, bytes.NewReader(buf), keyName)

	resp, err := s3Svc.SelectObjectContent(&s3.SelectObjectContentInput{
		Bucket:         &integMetadata.Buckets.Source.Name,
		Key:            &keyName,
		Expression:     aws.String("Select * from S3Object"),
		ExpressionType: aws.String(s3.ExpressionTypeSql),
		InputSerialization: &s3.InputSerialization{
			CSV: &s3.CSVInput{
				FieldDelimiter: aws.String(","),
				FileHeaderInfo: aws.String(s3.FileHeaderInfoIgnore),
			},
		},
		OutputSerialization: &s3.OutputSerialization{
			CSV: &s3.CSVOutput{
				FieldDelimiter: aws.String(","),
			},
		},
	})
	if err != nil {
		t.Fatalf("expect no error, %v", err)
	}
	defer resp.EventStream.Close()

	recReader, recWriter := io.Pipe()

	var sum int64
	var processed int64

	var gotEndEvent bool
	go func(w *io.PipeWriter, resp *s3.SelectObjectContentOutput) {
		defer recWriter.Close()
		var numRecordEvents int64
		for event := range resp.EventStream.Events() {
			switch tv := event.(type) {
			case *s3.RecordsEvent:
				n, err := recWriter.Write(tv.Payload)
				if err != nil {
					t.Logf("failed to write to record writer, %v, %v", n, err)
				}
				sum += int64(n)
				numRecordEvents++
			case *s3.StatsEvent:
				processed = *tv.Details.BytesProcessed
			case *s3.EndEvent:
				gotEndEvent = true
				t.Logf("s3.EndEvent received")
			}
		}
		t.Logf("received %d record events", numRecordEvents)
	}(recWriter, resp)

	type Record []string

	records := make(chan []Record)
	go func(r io.Reader, records chan<- []Record, batchSize int) {
		defer close(records)

		csvReader := csv.NewReader(r)
		var count int64

		batch := make([]Record, 0, batchSize)
		for {
			count++
			record, err := csvReader.Read()
			if err != nil {
				if _, ok := err.(*csv.ParseError); ok {
					t.Logf("failed to decode record row, %v, %v", count, err)
					continue
				}
				if err != io.EOF {
					t.Logf("csv decode failed, %v", err)
				}
				err = nil
				break
			}
			batch = append(batch, record)
			if len(batch) >= batchSize {
				records <- batch
				batch = batch[0:0]
			}
		}
		if len(batch) != 0 {
			records <- batch
		}
	}(recReader, records, 10)

	var count int64
	for batch := range records {
		// To simulate processing of a batch, add sleep delay.
		count += int64(len(batch))

		if err := resp.EventStream.Err(); err != nil {
			t.Errorf("exect no error, got %v", err)
		}
	}

	if !gotEndEvent {
		t.Errorf("expected EndEvent, did not receive")
	}

	if e, a := int64(101474), count; e != a {
		t.Errorf("expect %d records, got %d", e, a)
	}

	if sum == 0 {
		t.Errorf("expect selected content, got none")
	}

	if processed == 0 {
		t.Errorf("expect selected status bytes processed, got none")
	}

	if err := resp.EventStream.Err(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
}

func TestInteg_SelectObjectContent_Error(t *testing.T) {
	keyName := "negativeSelect.csv"

	buf := make([]byte, 0, 6*sdkio.MebiByte)
	buf = append(buf, []byte("name,number\n")...)
	line := []byte("jj,0\n")
	for i := 0; i < (cap(buf)/len(line))-2; i++ {
		buf = append(buf, line...)
	}
	buf = append(buf, []byte("gg,NaN\n")...)

	putTestContent(t, bytes.NewReader(buf), keyName)

	resp, err := s3Svc.SelectObjectContent(&s3.SelectObjectContentInput{
		Bucket:         &integMetadata.Buckets.Source.Name,
		Key:            &keyName,
		Expression:     aws.String("SELECT name FROM S3Object WHERE cast(number as int) < 1"),
		ExpressionType: aws.String(s3.ExpressionTypeSql),
		InputSerialization: &s3.InputSerialization{
			CSV: &s3.CSVInput{
				FileHeaderInfo: aws.String(s3.FileHeaderInfoUse),
			},
		},
		OutputSerialization: &s3.OutputSerialization{
			CSV: &s3.CSVOutput{
				FieldDelimiter: aws.String(","),
			},
		},
	})
	if err != nil {
		t.Fatalf("expect no error, %v", err)
	}
	defer resp.EventStream.Close()

	var sum int64
	for event := range resp.EventStream.Events() {
		switch tv := event.(type) {
		case *s3.RecordsEvent:
			sum += int64(len(tv.Payload))
		}
	}

	if sum == 0 {
		t.Errorf("expect selected content")
	}

	err = resp.EventStream.Err()
	if err == nil {
		t.Fatalf("exepct error")
	}

	aerr := err.(awserr.Error)
	if a := aerr.Code(); len(a) == 0 {
		t.Errorf("expect, error code")
	}
	if a := aerr.Message(); len(a) == 0 {
		t.Errorf("expect, error message")
	}
}

func TestInteg_SelectObjectContent_Stream(t *testing.T) {
	keyName := "selectGopher.csv"

	buf := `name,number
gopher,0
ᵷodɥǝɹ,1
`
	// Put a mock CSV file to the S3 bucket so that its contents can be
	// selected.
	putTestContent(t, strings.NewReader(buf), keyName)

	// Make the Select Object Content API request using the object uploaded.
	resp, err := s3Svc.SelectObjectContent(&s3.SelectObjectContentInput{
		Bucket:         &integMetadata.Buckets.Source.Name,
		Key:            &keyName,
		Expression:     aws.String("SELECT name FROM S3Object WHERE cast(number as int) < 1"),
		ExpressionType: aws.String(s3.ExpressionTypeSql),
		InputSerialization: &s3.InputSerialization{
			CSV: &s3.CSVInput{
				FileHeaderInfo: aws.String(s3.FileHeaderInfoUse),
			},
		},
		OutputSerialization: &s3.OutputSerialization{
			CSV: &s3.CSVOutput{},
		},
	})
	if err != nil {
		t.Fatalf("failed making API request, %v\n", err)
	}
	defer resp.EventStream.Close()

	results, resultWriter := io.Pipe()
	go func() {
		defer resultWriter.Close()
		for event := range resp.EventStream.Events() {
			switch e := event.(type) {
			case *s3.RecordsEvent:
				resultWriter.Write(e.Payload)
			case *s3.StatsEvent:
				t.Logf("Processed %d bytes\n", *e.Details.BytesProcessed)
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
		t.Log(record)
	}

	if err := resp.EventStream.Err(); err != nil {
		t.Fatalf("reading from event stream failed, %v\n", err)
	}
}
