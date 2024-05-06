/*
 *
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package remotesigner offloads private key operations to S2Av2.
package remotesigner

import (
	"crypto"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"io"

	"github.com/google/s2a-go/stream"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"

	s2av2pb "github.com/google/s2a-go/internal/proto/v2/s2a_go_proto"
)

// remoteSigner implementes the crypto.Signer interface.
type remoteSigner struct {
	leafCert  *x509.Certificate
	s2AStream stream.S2AStream
}

// New returns an instance of RemoteSigner, an implementation of the
// crypto.Signer interface.
func New(leafCert *x509.Certificate, s2AStream stream.S2AStream) crypto.Signer {
	return &remoteSigner{leafCert, s2AStream}
}

func (s *remoteSigner) Public() crypto.PublicKey {
	return s.leafCert.PublicKey
}

func (s *remoteSigner) Sign(rand io.Reader, digest []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	signatureAlgorithm, err := getSignatureAlgorithm(opts, s.leafCert)
	if err != nil {
		return nil, err
	}

	req, err := getSignReq(signatureAlgorithm, digest)
	if err != nil {
		return nil, err
	}
	if grpclog.V(1) {
		grpclog.Infof("Sending request to S2Av2 for signing operation.")
	}
	if err := s.s2AStream.Send(&s2av2pb.SessionReq{
		ReqOneof: &s2av2pb.SessionReq_OffloadPrivateKeyOperationReq{
			OffloadPrivateKeyOperationReq: req,
		},
	}); err != nil {
		grpclog.Infof("Failed to send request to S2Av2 for signing operation.")
		return nil, err
	}

	resp, err := s.s2AStream.Recv()
	if err != nil {
		grpclog.Infof("Failed to receive signing operation response from S2Av2.")
		return nil, err
	}

	if (resp.GetStatus() != nil) && (resp.GetStatus().Code != uint32(codes.OK)) {
		return nil, fmt.Errorf("failed to offload signing with private key to S2A: %d, %v", resp.GetStatus().Code, resp.GetStatus().Details)
	}

	return resp.GetOffloadPrivateKeyOperationResp().GetOutBytes(), nil
}

// getCert returns the leafCert field in s.
func (s *remoteSigner) getCert() *x509.Certificate {
	return s.leafCert
}

// getStream returns the s2AStream field in s.
func (s *remoteSigner) getStream() stream.S2AStream {
	return s.s2AStream
}

func getSignReq(signatureAlgorithm s2av2pb.SignatureAlgorithm, digest []byte) (*s2av2pb.OffloadPrivateKeyOperationReq, error) {
	if (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PKCS1_SHA256) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ECDSA_SECP256R1_SHA256) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PSS_RSAE_SHA256) {
		return &s2av2pb.OffloadPrivateKeyOperationReq{
			Operation:          s2av2pb.OffloadPrivateKeyOperationReq_SIGN,
			SignatureAlgorithm: signatureAlgorithm,
			InBytes: &s2av2pb.OffloadPrivateKeyOperationReq_Sha256Digest{
				Sha256Digest: digest,
			},
		}, nil
	} else if (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PKCS1_SHA384) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ECDSA_SECP384R1_SHA384) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PSS_RSAE_SHA384) {
		return &s2av2pb.OffloadPrivateKeyOperationReq{
			Operation:          s2av2pb.OffloadPrivateKeyOperationReq_SIGN,
			SignatureAlgorithm: signatureAlgorithm,
			InBytes: &s2av2pb.OffloadPrivateKeyOperationReq_Sha384Digest{
				Sha384Digest: digest,
			},
		}, nil
	} else if (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PKCS1_SHA512) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ECDSA_SECP521R1_SHA512) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PSS_RSAE_SHA512) || (signatureAlgorithm == s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ED25519) {
		return &s2av2pb.OffloadPrivateKeyOperationReq{
			Operation:          s2av2pb.OffloadPrivateKeyOperationReq_SIGN,
			SignatureAlgorithm: signatureAlgorithm,
			InBytes: &s2av2pb.OffloadPrivateKeyOperationReq_Sha512Digest{
				Sha512Digest: digest,
			},
		}, nil
	} else {
		return nil, fmt.Errorf("unknown signature algorithm: %v", signatureAlgorithm)
	}
}

// getSignatureAlgorithm returns the signature algorithm that S2A must use when
// performing a signing operation that has been offloaded by an application
// using the crypto/tls libraries.
func getSignatureAlgorithm(opts crypto.SignerOpts, leafCert *x509.Certificate) (s2av2pb.SignatureAlgorithm, error) {
	if opts == nil || leafCert == nil {
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_UNSPECIFIED, fmt.Errorf("unknown signature algorithm")
	}
	switch leafCert.PublicKeyAlgorithm {
	case x509.RSA:
		if rsaPSSOpts, ok := opts.(*rsa.PSSOptions); ok {
			return rsaPSSAlgorithm(rsaPSSOpts)
		}
		return rsaPPKCS1Algorithm(opts)
	case x509.ECDSA:
		return ecdsaAlgorithm(opts)
	case x509.Ed25519:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ED25519, nil
	default:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_UNSPECIFIED, fmt.Errorf("unknown signature algorithm: %q", leafCert.PublicKeyAlgorithm)
	}
}

func rsaPSSAlgorithm(opts *rsa.PSSOptions) (s2av2pb.SignatureAlgorithm, error) {
	switch opts.HashFunc() {
	case crypto.SHA256:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PSS_RSAE_SHA256, nil
	case crypto.SHA384:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PSS_RSAE_SHA384, nil
	case crypto.SHA512:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PSS_RSAE_SHA512, nil
	default:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_UNSPECIFIED, fmt.Errorf("unknown signature algorithm")
	}
}

func rsaPPKCS1Algorithm(opts crypto.SignerOpts) (s2av2pb.SignatureAlgorithm, error) {
	switch opts.HashFunc() {
	case crypto.SHA256:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PKCS1_SHA256, nil
	case crypto.SHA384:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PKCS1_SHA384, nil
	case crypto.SHA512:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_RSA_PKCS1_SHA512, nil
	default:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_UNSPECIFIED, fmt.Errorf("unknown signature algorithm")
	}
}

func ecdsaAlgorithm(opts crypto.SignerOpts) (s2av2pb.SignatureAlgorithm, error) {
	switch opts.HashFunc() {
	case crypto.SHA256:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ECDSA_SECP256R1_SHA256, nil
	case crypto.SHA384:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ECDSA_SECP384R1_SHA384, nil
	case crypto.SHA512:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_ECDSA_SECP521R1_SHA512, nil
	default:
		return s2av2pb.SignatureAlgorithm_S2A_SSL_SIGN_UNSPECIFIED, fmt.Errorf("unknown signature algorithm")
	}
}
