/*
Copyright 2024 The Kubernetes Authors.

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
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"
	"gopkg.in/go-jose/go-jose.v2/jwt"

	"k8s.io/kubernetes/pkg/serviceaccount"

	utilnettesting "k8s.io/apimachinery/pkg/util/net/testing"
	externaljwtv1alpha1 "k8s.io/externaljwt/apis/v1alpha1"
)

var (
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

		iss                         string
		backendSetKeyID             string
		backendSetAlgorithm         string
		supportedKeys               map[string]supportedKeyT
		allowSigningWithNonOIDCKeys bool

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
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key: &rsaKey1.PublicKey,
				},
			},

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
			desc: "correct token with correct claims signed by key that's excluded from OIDC",
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
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key:             &rsaKey1.PublicKey,
					excludeFromOidc: true,
				},
			},

			wantErr: fmt.Errorf("while validating header: key used for signing JWT (kid: key-id-1) is excluded from OIDC discovery docs"),
		},
		{
			desc: "token signed with key that's excluded from OIDC but validation is disabled",
			publicClaims: jwt.Claims{
				Subject: "some-subject",
				Audience: jwt.Audience{
					"some-audience-1",
					"some-audience-2",
				},
				ID: "key-id-1",
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
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key:             &rsaKey1.PublicKey,
					excludeFromOidc: true,
				},
			},
			allowSigningWithNonOIDCKeys: true,

			wantClaims: unifiedClaimsT{
				Issuer:  "some-issuer",
				Subject: "some-subject",
				Audience: jwt.Audience{
					"some-audience-1",
					"some-audience-2",
				},
				ID: "key-id-1",
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
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key:             &rsaKey1.PublicKey,
					excludeFromOidc: true,
				},
			},
			wantErr: fmt.Errorf("while validating header: key id missing"),
		},
		{
			desc:                "key id longer than 1024 bytes returned from signer",
			iss:                 "some-issuer",
			backendSetKeyID:     string(make([]byte, 1025)),
			backendSetAlgorithm: "RS256",
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key:             &rsaKey1.PublicKey,
					excludeFromOidc: true,
				},
			},
			wantErr: fmt.Errorf("while validating header: key id longer than 1 kb"),
		},
		{
			desc:                "unsupported alg returned from signer",
			iss:                 "some-issuer",
			backendSetKeyID:     "key-id-1",
			backendSetAlgorithm: "something-unsupported",
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key:             &rsaKey1.PublicKey,
					excludeFromOidc: true,
				},
			},
			wantErr: fmt.Errorf("while validating header: bad signing algorithm \"something-unsupported\""),
		},
		{
			desc:                "empty alg returned from signer",
			iss:                 "some-issuer",
			backendSetKeyID:     "key-id-1",
			backendSetAlgorithm: "",
			supportedKeys: map[string]supportedKeyT{
				"key-id-1": {
					key:             &rsaKey1.PublicKey,
					excludeFromOidc: true,
				},
			},
			wantErr: fmt.Errorf("while validating header: bad signing algorithm \"\""),
		},
	}

	for i, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := context.Background()

			sockname := utilnettesting.MakeSocketNameForTest(t, fmt.Sprintf("test-external-token-generator-%d-%d.sock", time.Now().Nanosecond(), i))

			addr := &net.UnixAddr{Name: sockname, Net: "unix"}
			listener, err := net.ListenUnix(addr.Network(), addr)
			if err != nil {
				t.Fatalf("Failed to start fake backend: %v", err)
			}

			grpcServer := grpc.NewServer()

			backend := &dummyExtrnalSigner{
				keyID:              tc.backendSetKeyID,
				signingAlgorithm:   tc.backendSetAlgorithm,
				signature:          "abcdef",
				supportedKeys:      tc.supportedKeys,
				refreshHintSeconds: 10,
				DataTimeStamp:      timestamppb.New(time.Time{}),
			}
			externaljwtv1alpha1.RegisterExternalJWTSignerServer(grpcServer, backend)

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
			defer func() { _ = clientConn.Close() }()

			plugin := newPlugin(tc.iss, clientConn, tc.allowSigningWithNonOIDCKeys)
			err = plugin.keyCache.initialFill(ctx)
			if err != nil {
				t.Fatalf("initial fill failed: %v", err)
			}

			gotToken, err := plugin.GenerateToken(ctx, &tc.publicClaims, tc.privateClaims)
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
				t.Fatalf("error converting received tokens to bytes: %v", err)
			}

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

func sortPublicKeySlice(a, b serviceaccount.PublicKey) bool {
	return a.KeyID < b.KeyID
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

type supportedKeyT struct {
	key             *rsa.PublicKey
	excludeFromOidc bool
}

type dummyExtrnalSigner struct {
	externaljwtv1alpha1.UnimplementedExternalJWTSignerServer

	// required for Sign()
	keyID            string
	signingAlgorithm string
	signature        string

	// required for FetchKeys()
	keyLock               sync.Mutex
	supportedKeys         map[string]supportedKeyT
	refreshHintSeconds    int
	DataTimeStamp         *timestamppb.Timestamp
	SupportedKeysOverride []*externaljwtv1alpha1.Key
}

func (des *dummyExtrnalSigner) Sign(ctx context.Context, r *externaljwtv1alpha1.SignJWTRequest) (*externaljwtv1alpha1.SignJWTResponse, error) {
	header := &headerT{
		Type:      "JWT",
		Algorithm: des.signingAlgorithm,
		KeyID:     des.keyID,
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("failed to create header for JWT response")
	}

	resp := &externaljwtv1alpha1.SignJWTResponse{
		Header:    base64.RawURLEncoding.EncodeToString(headerJSON),
		Signature: des.signature,
	}
	return resp, nil
}

func (des *dummyExtrnalSigner) FetchKeys(ctx context.Context, r *externaljwtv1alpha1.FetchKeysRequest) (*externaljwtv1alpha1.FetchKeysResponse, error) {
	des.keyLock.Lock()
	defer des.keyLock.Unlock()

	pbKeys := []*externaljwtv1alpha1.Key{}
	if des.SupportedKeysOverride != nil {
		pbKeys = des.SupportedKeysOverride
	} else {
		for kid, k := range des.supportedKeys {
			keyBytes, err := x509.MarshalPKIXPublicKey(k.key)
			if err != nil {
				return nil, fmt.Errorf("while marshaling key: %w", err)
			}
			pbKey := &externaljwtv1alpha1.Key{
				KeyId:                    kid,
				Key:                      keyBytes,
				ExcludeFromOidcDiscovery: k.excludeFromOidc,
			}
			pbKeys = append(pbKeys, pbKey)
		}
	}

	return &externaljwtv1alpha1.FetchKeysResponse{
		Keys:               pbKeys,
		DataTimestamp:      des.DataTimeStamp,
		RefreshHintSeconds: int64(des.refreshHintSeconds),
	}, nil
}
