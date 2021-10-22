// +build go1.7

package v4

import (
	"encoding/hex"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/credentials"
)

type periodicBadCredentials struct {
	call        int
	credentials *credentials.Credentials
}

func (p *periodicBadCredentials) Get() (credentials.Value, error) {
	defer func() {
		p.call++
	}()

	if p.call%2 == 0 {
		return credentials.Value{}, fmt.Errorf("credentials error")
	}

	return p.credentials.Get()
}

type chunk struct {
	headers, payload []byte
}

func mustDecodeHex(b []byte, err error) []byte {
	if err != nil {
		panic(err)
	}

	return b
}

func TestStreamingChunkSigner(t *testing.T) {
	const (
		region        = "us-east-1"
		service       = "transcribe"
		seedSignature = "9d9ab996c81f32c9d4e6fc166c92584f3741d1cb5ce325cd11a77d1f962c8de2"
	)

	staticCredentials := credentials.NewStaticCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "")
	currentTime := time.Date(2019, 1, 27, 22, 37, 54, 0, time.UTC)

	cases := map[string]struct {
		credentials        credentialValueProvider
		chunks             []chunk
		expectedSignatures map[int]string
		expectedErrors     map[int]string
	}{
		"signature calculation": {
			credentials: staticCredentials,
			chunks: []chunk{
				{headers: []byte("headers"), payload: []byte("payload")},
				{headers: []byte("more headers"), payload: []byte("more payload")},
			},
			expectedSignatures: map[int]string{
				0: "681a7eaa82891536f24af7ec7e9219ee251ccd9bac2f1b981eab7c5ec8579115",
				1: "07633d9d4ab4d81634a2164934d1f648c7cbc6839a8cf0773d818127a267e4d6",
			},
		},
		"signature calculation errors": {
			credentials: &periodicBadCredentials{credentials: staticCredentials},
			chunks: []chunk{
				{headers: []byte("headers"), payload: []byte("payload")},
				{headers: []byte("headers"), payload: []byte("payload")},
				{headers: []byte("more headers"), payload: []byte("more payload")},
				{headers: []byte("more headers"), payload: []byte("more payload")},
			},
			expectedSignatures: map[int]string{
				1: "681a7eaa82891536f24af7ec7e9219ee251ccd9bac2f1b981eab7c5ec8579115",
				3: "07633d9d4ab4d81634a2164934d1f648c7cbc6839a8cf0773d818127a267e4d6",
			},
			expectedErrors: map[int]string{
				0: "credentials error",
				2: "credentials error",
			},
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			chunkSigner := &StreamSigner{
				region:      region,
				service:     service,
				credentials: tt.credentials,
				prevSig:     mustDecodeHex(hex.DecodeString(seedSignature)),
			}

			for i, chunk := range tt.chunks {
				var expectedError string
				if len(tt.expectedErrors) != 0 {
					_, ok := tt.expectedErrors[i]
					if ok {
						expectedError = tt.expectedErrors[i]
					}
				}

				signature, err := chunkSigner.GetSignature(chunk.headers, chunk.payload, currentTime)
				if err == nil && len(expectedError) > 0 {
					t.Errorf("expected error, but got nil")
					continue
				} else if err != nil && len(expectedError) == 0 {
					t.Errorf("expected no error, but got %v", err)
					continue
				} else if err != nil && len(expectedError) > 0 && !strings.Contains(err.Error(), expectedError) {
					t.Errorf("expected %v, but got %v", expectedError, err)
					continue
				} else if len(expectedError) > 0 {
					continue
				}

				expectedSignature, ok := tt.expectedSignatures[i]
				if !ok {
					t.Fatalf("expected signature not provided for test case")
				}

				if e, a := expectedSignature, hex.EncodeToString(signature); e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
			}
		})
	}
}
