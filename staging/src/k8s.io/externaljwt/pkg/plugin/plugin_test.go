/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package plugin

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"
	"gopkg.in/square/go-jose.v2/jwt"
	"k8s.io/kubernetes/pkg/serviceaccount"

	ejspb "k8s.io/externaljwt/apis/v1alpha1"
)

var (
	unixSocket      = "/tmp/mysocket.sock"
	commonSignature = "common-signature"

	rsaKey1 *rsa.PrivateKey
	rsaKey2 *rsa.PrivateKey
)

func init() {
	var err error

	rsaKey1, err = rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic("Error while generating first RSA key")
	}

	rsaKey2, err = rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic("Error while generating second RSA key")
	}
}

func TestExternalTokenGenerator(t *testing.T) {
	testCases := []struct {
		desc string

		publicClaims  jwt.Claims
		privateClaims privateClaimsT

		iss                 string
		backendSetKeyID     string
		backendSetAlgorithm string

		wantClaims unifiedClaimsT
		wantErr    error
	}{
		{
			desc: "correct token with correct claims returned",
			publicClaims: jwt.Claims{
				Subject: "some-subject",
				Audience: jwt.Audience{
					"some-audience-1",
					"some-audience-2",
				},
				ID: "id-1",
			},
			privateClaims: privateClaimsT{
				Kubernetes: kubernetesT{
					Namespace: "foo",
					Svcacct: refT{
						Name: "default",
						UID:  "abcdef",
					},
				},
			},
			iss:                 "some-issuer",
			backendSetKeyID:     "key-id-1",
			backendSetAlgorithm: "RS256",

			wantClaims: unifiedClaimsT{
				Issuer:  "some-issuer",
				Subject: "some-subject",
				Audience: jwt.Audience{
					"some-audience-1",
					"some-audience-2",
				},
				ID: "id-1",
				Kubernetes: kubernetesT{
					Namespace: "foo",
					Svcacct: refT{
						Name: "default",
						UID:  "abcdef",
					},
				},
			},
		},
		{
			desc:                "empty key ID returned from signer",
			iss:                 "some-issuer",
			backendSetKeyID:     "",
			backendSetAlgorithm: "RS256",
			wantErr:             fmt.Errorf("while validating header: key id missing"),
		},
		{
			desc:                "unsupported alg returned from signer",
			iss:                 "some-issuer",
			backendSetKeyID:     "key-id-1",
			backendSetAlgorithm: "something-unsupported",
			wantErr:             fmt.Errorf("while validating header: bad signing algorithm \"something-unsupported\""),
		},
		{
			desc:                "empty alg returned from signer",
			iss:                 "some-issuer",
			backendSetKeyID:     "key-id-1",
			backendSetAlgorithm: "",
			wantErr:             fmt.Errorf("while validating header: bad signing algorithm \"\""),
		},
	}

	for i, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := context.Background()

			sockname := fmt.Sprintf("@test-external-token-generator-%d.sock", i)

			addr := &net.UnixAddr{Name: sockname, Net: "unix"}
			listener, err := net.ListenUnix(addr.Network(), addr)
			if err != nil {
				t.Fatalf("Failed to start fake backend: %v", err)
			}

			grpcServer := grpc.NewServer()

			backend := &dummyExtrnalSigner{
				keyID:            tc.backendSetKeyID,
				signingAlgorithm: tc.backendSetAlgorithm,
				signature:        "abcdef",
			}
			ejspb.RegisterExternalJWTSignerServer(grpcServer, backend)

			go func() {
				if err := grpcServer.Serve(listener); err != nil {
					panic(fmt.Errorf("error returned from grpcServer: %w", err))
				}
			}()
			defer grpcServer.Stop()

			clientConn, err := grpc.DialContext(
				ctx,
				sockname,
				grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", path)
				}),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial buffconn client: %v", err)
			}
			defer clientConn.Close()

			plugin := NewPlugin(tc.iss, clientConn)

			gotToken, err := plugin.GenerateToken(&tc.publicClaims, tc.privateClaims)
			if err != nil && tc.wantErr != nil {
				if err.Error() != tc.wantErr.Error() {
					t.Fatalf("want error: %v, got error: %v", tc.wantErr, err)
				}
				return
			} else if err != nil && tc.wantErr == nil {
				t.Fatalf("Unexpected error generating token: %v", err)
			} else if err == nil && tc.wantErr != nil {
				t.Fatalf("Wanted error %q, but got nil", tc.wantErr)
			}

			tokenPieces := strings.Split(gotToken, ".")
			payloadBase64 := tokenPieces[1]

			gotClaimBytes, err := base64.RawURLEncoding.DecodeString(payloadBase64)
			if err != nil {
				t.Fatalf("error converting recieved tokens to bytes: %v", err)
			}

			t.Logf("gotClaimBytes=%q", gotClaimBytes)

			gotClaims := unifiedClaimsT{}
			if err := json.Unmarshal(gotClaimBytes, &gotClaims); err != nil {
				t.Fatalf("Error while unmarshaling claims from backend: %v", err)
			}

			if diff := cmp.Diff(gotClaims, tc.wantClaims); diff != "" {
				t.Fatalf("Bad claims; diff (-got +want):\n%s", diff)
			}

			// Don't check header or signature values since we're not testing
			// our (fake) backends.
		})
	}

}

func TestExternalPublicKeyGetter(t *testing.T) {
	testCases := []struct {
		desc                 string
		expectedErr          error
		signingPrivateKeys   map[string]any
		wantVerificationKeys *VerificationKeys
		refreshHintSec       int
	}{
		{
			desc: "single key in signer",
			signingPrivateKeys: map[string]any{
				"key-1": rsaKey1,
			},
			wantVerificationKeys: &VerificationKeys{
				Keys: []serviceaccount.PublicKey{
					{
						KeyID:     "key-1",
						PublicKey: &rsaKey1.PublicKey,
					},
				},
			},
			refreshHintSec: 20,
		},
		{
			desc: "multiple keys in signer",
			signingPrivateKeys: map[string]any{
				"key-1": rsaKey1,
				"key-2": rsaKey2,
			},
			wantVerificationKeys: &VerificationKeys{
				Keys: []serviceaccount.PublicKey{
					{
						KeyID:     "key-1",
						PublicKey: &rsaKey1.PublicKey,
					},
					{
						KeyID:     "key-2",
						PublicKey: &rsaKey2.PublicKey,
					},
				},
			},
			refreshHintSec: 10,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := context.Background()

			sockname := fmt.Sprintf("@test-external-public-key-getter-%d.sock", i)

			addr := &net.UnixAddr{Name: sockname, Net: "unix"}
			listener, err := net.ListenUnix(addr.Network(), addr)
			if err != nil {
				t.Fatalf("Failed to start fake backend: %v", err)
			}

			grpcServer := grpc.NewServer()

			backend := &dummyExtrnalSigner{
				signingPrivateKeys: tc.signingPrivateKeys,
				refreshHintSeconds: tc.refreshHintSec,
			}
			ejspb.RegisterExternalJWTSignerServer(grpcServer, backend)

			defer grpcServer.Stop()
			go func() {
				if err := grpcServer.Serve(listener); err != nil {
					panic(fmt.Errorf("error returned from grpcServer: %w", err))
				}
			}()

			clientConn, err := grpc.DialContext(
				ctx,
				sockname,
				grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", path)
				}),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial buffconn client: %v", err)
			}
			defer clientConn.Close()

			plugin := NewPlugin("iss", clientConn)

			signingKeys, refreshHint, err := plugin.GetTokenVerificationKeys(ctx)
			if err != nil {
				if tc.expectedErr == nil {
					t.Fatalf("error getting supported keys: %v", err)
				}
				if tc.expectedErr.Error() != err.Error() {
					t.Fatalf("want error: %v, got error: %v", tc.expectedErr, err)
					return
				}
			}

			if tc.expectedErr == nil {
				if diff := cmp.Diff(signingKeys, tc.wantVerificationKeys, cmpopts.SortSlices(sortPublicKeySlice)); diff != "" {
					t.Fatalf("Bad result from GetTokenSigningKeys; diff (-got +want)\n%s", diff)
				}
				if refreshHint != tc.refreshHintSec {
					t.Fatalf("refreshHint not as expected; got: %d want: %d", refreshHint, tc.refreshHintSec)
				}
			}
		})
	}

}

func sortPublicKeySlice(a, b serviceaccount.PublicKey) bool {
	if a.KeyID < b.KeyID {
		return true
	} else if a.KeyID == b.KeyID {
		// We don't have a meaningful way to order arbitary public keys.
		return false
	} else {
		return false
	}
}

type claim struct {
	Subject  string   `json:"sub,omitempty"`
	Audience []string `json:"aud,omitempty"`
	ID       string   `json:"jti,omitempty"`
	Pc       string   `json:"pc,omitempty"`
	Issuer   string   `json:"iss,omitempty"`
}

type headerT struct {
	Algorithm string `json:"alg"`
	KeyID     string `json:"kid,omitempty"`
	Type      string `json:"typ"`
}

type unifiedClaimsT struct {
	Issuer     string           `json:"iss,omitempty"`
	Subject    string           `json:"sub,omitempty"`
	Audience   jwt.Audience     `json:"aud,omitempty"`
	Expiry     *jwt.NumericDate `json:"exp,omitempty"`
	NotBefore  *jwt.NumericDate `json:"nbf,omitempty"`
	IssuedAt   *jwt.NumericDate `json:"iat,omitempty"`
	ID         string           `json:"jti,omitempty"`
	Kubernetes kubernetesT      `json:"kubernetes.io,omitempty"`
}

type privateClaimsT struct {
	Kubernetes kubernetesT `json:"kubernetes.io,omitempty"`
}

type kubernetesT struct {
	Namespace string           `json:"namespace,omitempty"`
	Svcacct   refT             `json:"serviceaccount,omitempty"`
	Pod       *refT            `json:"pod,omitempty"`
	Secret    *refT            `json:"secret,omitempty"`
	Node      *refT            `json:"node,omitempty"`
	WarnAfter *jwt.NumericDate `json:"warnafter,omitempty"`
}

type refT struct {
	Name string `json:"name,omitempty"`
	UID  string `json:"uid,omitempty"`
}

type dummyExtrnalSigner struct {
	ejspb.UnimplementedExternalJWTSignerServer

	// required for Sign()
	keyID            string
	signingAlgorithm string
	signature        string

	// required for FetchKeys()
	keyLock            sync.Mutex
	signingPrivateKeys map[string]any
	refreshHintSeconds int
}

func (des *dummyExtrnalSigner) Sign(ctx context.Context, r *ejspb.SignJWTRequest) (*ejspb.SignJWTResponse, error) {
	header := &headerT{
		Type:      "JWT",
		Algorithm: des.signingAlgorithm,
		KeyID:     des.keyID,
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("failed to create header for JWT response")
	}

	resp := &ejspb.SignJWTResponse{
		Header:    base64.RawURLEncoding.EncodeToString(headerJSON),
		Signature: des.signature,
	}
	return resp, nil
}

func (des *dummyExtrnalSigner) FetchKeys(ctx context.Context, r *ejspb.FetchKeysRequest) (*ejspb.FetchKeysResponse, error) {
	des.keyLock.Lock()
	defer des.keyLock.Unlock()

	pbKeys := []*ejspb.Key{}
	for kid, key := range des.signingPrivateKeys {
		keyBytes, err := x509.MarshalPKIXPublicKey(&key.(*rsa.PrivateKey).PublicKey)
		if err != nil {
			return nil, fmt.Errorf("while marshaling key: %w", err)
		}
		pbKey := &ejspb.Key{
			KeyId: kid,
			Key:   keyBytes,
		}
		pbKeys = append(pbKeys, pbKey)
	}

	return &ejspb.FetchKeysResponse{
		Keys:               pbKeys,
		DataTimestamp:      timestamppb.New(time.Time{}),
		RefreshHintSeconds: int64(des.refreshHintSeconds),
	}, nil
}
