package polly

import (
	"regexp"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

func TestRestGETStrategy(t *testing.T) {
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})
	r, _ := svc.SynthesizeSpeechRequest(nil)

	if err := restGETPresignStrategy(r); err != nil {
		t.Error(err)
	}
	if "GET" != r.HTTPRequest.Method {
		t.Errorf("Expected 'GET', but received %s", r.HTTPRequest.Method)
	}
	if r.Operation.BeforePresignFn == nil {
		t.Error("Expected non-nil value for 'BeforePresignFn'")
	}
}

func TestPresign(t *testing.T) {
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})
	r, _ := svc.SynthesizeSpeechRequest(&SynthesizeSpeechInput{
		Text:         aws.String("Moo"),
		OutputFormat: aws.String("mp3"),
		VoiceId:      aws.String("Foo"),
	})
	url, err := r.Presign(time.Second)

	if err != nil {
		t.Error(err)
	}
	expectedURL := `^https://polly.us-west-2.amazonaws.com/v1/speech\?.*?OutputFormat=mp3.*?Text=Moo.*?VoiceId=Foo.*`
	if matched, err := regexp.MatchString(expectedURL, url); !matched || err != nil {
		t.Errorf("Expected:\n%q\nReceived:\n%q\nError:\n%v\n", expectedURL, url, err)
	}
}
