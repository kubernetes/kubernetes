// +build !integration

package glacier_test

import (
	"bytes"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/glacier"
)

var (
	payloadBuf = func() *bytes.Reader {
		buf := make([]byte, 5767168) // 5.5MB buffer
		for i := range buf {
			buf[i] = '0' // Fill with zero characters
		}
		return bytes.NewReader(buf)
	}()

	svc = glacier.New(unit.Session)
)

func TestCustomizations(t *testing.T) {
	req, _ := svc.UploadArchiveRequest(&glacier.UploadArchiveInput{
		VaultName: aws.String("vault"),
		Body:      payloadBuf,
	})
	err := req.Build()
	if err != nil {
		t.Errorf("expect no err, got %v", err)
	}

	// Sets API version
	if e, a := req.ClientInfo.APIVersion, req.HTTPRequest.Header.Get("x-amz-glacier-version"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// Sets Account ID
	v, _ := awsutil.ValuesAtPath(req.Params, "AccountId")
	if e, a := "-", *(v[0].(*string)); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// Computes checksums
	linear := "68aff0c5a91aa0491752bfb96e3fef33eb74953804f6a2f7b708d5bcefa8ff6b"
	tree := "154e26c78fd74d0c2c9b3cc4644191619dc4f2cd539ae2a74d5fd07957a3ee6a"
	if e, a := linear, req.HTTPRequest.Header.Get("x-amz-content-sha256"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := tree, req.HTTPRequest.Header.Get("x-amz-sha256-tree-hash"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestShortcircuitTreehash(t *testing.T) {
	req, _ := svc.UploadArchiveRequest(&glacier.UploadArchiveInput{
		VaultName: aws.String("vault"),
		Body:      payloadBuf,
		Checksum:  aws.String("000"),
	})
	err := req.Build()
	if err != nil {
		t.Errorf("expect no err, got %v", err)
	}

	if e, a := "000", req.HTTPRequest.Header.Get("x-amz-sha256-tree-hash"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestFillAccountIDWithNilStruct(t *testing.T) {
	req, _ := svc.ListVaultsRequest(nil)
	err := req.Build()
	if err != nil {
		t.Errorf("expect no err, got %v", err)
	}

	empty := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

	// Sets Account ID
	v, _ := awsutil.ValuesAtPath(req.Params, "AccountId")
	if e, a := "-", *(v[0].(*string)); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// Does not set tree hash
	if e, a := empty, req.HTTPRequest.Header.Get("x-amz-content-sha256"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "", req.HTTPRequest.Header.Get("x-amz-sha256-tree-hash"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestHashOnce(t *testing.T) {
	req, _ := svc.UploadArchiveRequest(&glacier.UploadArchiveInput{
		VaultName: aws.String("vault"),
		Body:      payloadBuf,
	})
	req.HTTPRequest.Header.Set("X-Amz-Sha256-Tree-Hash", "0")

	err := req.Build()
	if err != nil {
		t.Errorf("expect no err, got %v", err)
	}

	if e, a := "0", req.HTTPRequest.Header.Get("x-amz-sha256-tree-hash"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
