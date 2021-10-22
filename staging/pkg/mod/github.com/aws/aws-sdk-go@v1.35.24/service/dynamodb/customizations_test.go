package dynamodb_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

var db *dynamodb.DynamoDB

func TestMain(m *testing.M) {
	db = dynamodb.New(unit.Session, &aws.Config{
		MaxRetries: aws.Int(2),
	})
	db.Handlers.Send.Clear() // mock sending

	os.Exit(m.Run())
}

func mockCRCResponse(svc *dynamodb.DynamoDB, status int, body, crc string) (req *request.Request) {
	header := http.Header{}
	header.Set("x-amz-crc32", crc)

	req, _ = svc.ListTablesRequest(nil)
	req.Handlers.Build.RemoveByName("crr.endpointdiscovery")

	req.Handlers.Send.PushBack(func(*request.Request) {
		req.HTTPResponse = &http.Response{
			ContentLength: int64(len(body)),
			StatusCode:    status,
			Body:          ioutil.NopCloser(bytes.NewReader([]byte(body))),
			Header:        header,
		}
	})
	req.Send()
	return
}

func TestDefaultRetryRules(t *testing.T) {
	d := dynamodb.New(unit.Session, &aws.Config{MaxRetries: aws.Int(-1)})
	if e, a := 10, d.MaxRetries(); e != a {
		t.Errorf("expect %d max retries, got %d", e, a)
	}
}

func TestCustomRetryRules(t *testing.T) {
	d := dynamodb.New(unit.Session, &aws.Config{MaxRetries: aws.Int(2)})
	if e, a := 2, d.MaxRetries(); e != a {
		t.Errorf("expect %d max retries, got %d", e, a)
	}
}

type testCustomRetryer struct {
	client.DefaultRetryer
}

func TestCustomRetry_FromConfig(t *testing.T) {
	d := dynamodb.New(unit.Session, &aws.Config{
		Retryer: testCustomRetryer{client.DefaultRetryer{NumMaxRetries: 9}},
	})

	if _, ok := d.Retryer.(testCustomRetryer); !ok {
		t.Errorf("expect retryer to be testCustomRetryer, but got %T", d.Retryer)
	}

	if e, a := 9, d.MaxRetries(); e != a {
		t.Errorf("expect %d max retries from custom retryer, got %d", e, a)
	}
}

func TestValidateCRC32NoHeaderSkip(t *testing.T) {
	req := mockCRCResponse(db, 200, "{}", "")
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}
}

func TestValidateCRC32InvalidHeaderSkip(t *testing.T) {
	req := mockCRCResponse(db, 200, "{}", "ABC")
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}
}

func TestValidateCRC32AlreadyErrorSkip(t *testing.T) {
	req := mockCRCResponse(db, 400, "{}", "1234")
	if req.Error == nil {
		t.Fatalf("expect error, but got none")
	}

	aerr := req.Error.(awserr.Error)
	if aerr.Code() == "CRC32CheckFailed" {
		t.Errorf("expect error code not to be CRC32CheckFailed")
	}
}

func TestValidateCRC32IsValid(t *testing.T) {
	req := mockCRCResponse(db, 200, `{"TableNames":["A"]}`, "3090163698")
	if req.Error != nil {
		t.Fatalf("expect no error, got %v", req.Error)
	}

	// CRC check does not affect output parsing
	out := req.Data.(*dynamodb.ListTablesOutput)
	if e, a := "A", *out.TableNames[0]; e != a {
		t.Errorf("expect %q table name, got %q", e, a)
	}
}

func TestValidateCRC32DoesNotMatch(t *testing.T) {
	req := mockCRCResponse(db, 200, "{}", "1234")
	if req.Error == nil {
		t.Fatalf("expect error, but got none")
	}
	req.Handlers.Build.RemoveByName("crr.endpointdiscovery")

	aerr := req.Error.(awserr.Error)
	if e, a := "CRC32CheckFailed", aerr.Code(); e != a {
		t.Errorf("expect %s error code, got %s", e, a)
	}
	if e, a := 2, req.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
}

func TestValidateCRC32DoesNotMatchNoComputeChecksum(t *testing.T) {
	svc := dynamodb.New(unit.Session, &aws.Config{
		MaxRetries:              aws.Int(2),
		DisableComputeChecksums: aws.Bool(true),
	})
	svc.Handlers.Send.Clear() // mock sending

	req := mockCRCResponse(svc, 200, `{"TableNames":["A"]}`, "1234")
	if req.Error != nil {
		t.Fatalf("expect no error, got %v", req.Error)
	}

	if e, a := 0, req.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}

	// CRC check disabled. Does not affect output parsing
	out := req.Data.(*dynamodb.ListTablesOutput)
	if e, a := "A", *out.TableNames[0]; e != a {
		t.Errorf("expect %q table name, got %q", e, a)
	}
}
