// +build go1.7

package ec2metadata_test

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/internal/sdktesting"
)

func TestClientOverrideDefaultHTTPClientTimeout(t *testing.T) {
	svc := ec2metadata.New(unit.Session)

	if e, a := http.DefaultClient, svc.Config.HTTPClient; e == a {
		t.Errorf("expect %v, not to equal %v", e, a)
	}

	if e, a := 1*time.Second, svc.Config.HTTPClient.Timeout; e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
}

func TestClientNotOverrideDefaultHTTPClientTimeout(t *testing.T) {
	http.DefaultClient.Transport = &http.Transport{}
	defer func() {
		http.DefaultClient.Transport = nil
	}()

	svc := ec2metadata.New(unit.Session)

	if e, a := http.DefaultClient, svc.Config.HTTPClient; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	tr := svc.Config.HTTPClient.Transport.(*http.Transport)
	if tr == nil {
		t.Fatalf("expect transport not to be nil")
	}
	if tr.Dial != nil {
		t.Errorf("expect dial to be nil, was not")
	}
}

func TestClientDisableOverrideDefaultHTTPClientTimeout(t *testing.T) {
	svc := ec2metadata.New(unit.Session, aws.NewConfig().WithEC2MetadataDisableTimeoutOverride(true))

	if e, a := http.DefaultClient, svc.Config.HTTPClient; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestClientOverrideDefaultHTTPClientTimeoutRace(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("us-east-1a"))
	}))
	defer server.Close()

	cfg := aws.NewConfig().WithEndpoint(server.URL)
	runEC2MetadataClients(t, cfg, 50)
}

func TestClientOverrideDefaultHTTPClientTimeoutRaceWithTransport(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("us-east-1a"))
	}))
	defer server.Close()

	cfg := aws.NewConfig().WithEndpoint(server.URL).WithHTTPClient(&http.Client{
		Transport: &http.Transport{
			DisableKeepAlives: true,
		},
	})

	runEC2MetadataClients(t, cfg, 50)
}

func TestClientDisableIMDS(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_EC2_METADATA_DISABLED", "true")

	svc := ec2metadata.New(unit.Session)
	resp, err := svc.GetUserData()
	if err == nil {
		t.Fatalf("expect error, got none")
	}
	if len(resp) != 0 {
		t.Errorf("expect no response, got %v", resp)
	}

	aerr := err.(awserr.Error)
	if e, a := request.CanceledErrorCode, aerr.Code(); e != a {
		t.Errorf("expect %v error code, got %v", e, a)
	}
	if e, a := "AWS_EC2_METADATA_DISABLED", aerr.Message(); !strings.Contains(a, e) {
		t.Errorf("expect %v in error message, got %v", e, a)
	}
}

func TestClientStripPath(t *testing.T) {
	cases := map[string]struct {
		Endpoint string
		Expect   string
	}{
		"no change": {
			Endpoint: "http://example.aws",
			Expect:   "http://example.aws",
		},
		"strip path": {
			Endpoint: "http://example.aws/foo",
			Expect:   "http://example.aws",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			restoreEnvFn := sdktesting.StashEnv()
			defer restoreEnvFn()

			svc := ec2metadata.New(unit.Session, &aws.Config{
				Endpoint: aws.String(c.Endpoint),
			})

			if e, a := c.Expect, svc.ClientInfo.Endpoint; e != a {
				t.Errorf("expect %v endpoint, got %v", e, a)
			}
		})
	}
}

func runEC2MetadataClients(t *testing.T, cfg *aws.Config, atOnce int) {
	var wg sync.WaitGroup
	wg.Add(atOnce)
	svc := ec2metadata.New(unit.Session, cfg)
	for i := 0; i < atOnce; i++ {
		go func() {
			defer wg.Done()
			_, err := svc.GetUserData()
			if err != nil {
				t.Errorf("expect no error, got %v", err)
			}
		}()
	}
	wg.Wait()
}
