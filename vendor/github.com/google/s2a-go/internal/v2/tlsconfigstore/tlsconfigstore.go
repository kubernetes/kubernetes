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

// Package tlsconfigstore offloads operations to S2Av2.
package tlsconfigstore

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"

	"github.com/google/s2a-go/internal/tokenmanager"
	"github.com/google/s2a-go/internal/v2/certverifier"
	"github.com/google/s2a-go/internal/v2/remotesigner"
	"github.com/google/s2a-go/stream"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"

	commonpbv1 "github.com/google/s2a-go/internal/proto/common_go_proto"
	commonpb "github.com/google/s2a-go/internal/proto/v2/common_go_proto"
	s2av2pb "github.com/google/s2a-go/internal/proto/v2/s2a_go_proto"
)

const (
	// HTTP/2
	h2 = "h2"
)

// GetTLSConfigurationForClient returns a tls.Config instance for use by a client application.
func GetTLSConfigurationForClient(serverHostname string, s2AStream stream.S2AStream, tokenManager tokenmanager.AccessTokenManager, localIdentity *commonpbv1.Identity, verificationMode s2av2pb.ValidatePeerCertificateChainReq_VerificationMode, serverAuthorizationPolicy []byte) (*tls.Config, error) {
	authMechanisms := getAuthMechanisms(tokenManager, []*commonpbv1.Identity{localIdentity})

	if grpclog.V(1) {
		grpclog.Infof("Sending request to S2Av2 for client TLS config.")
	}
	// Send request to S2Av2 for config.
	if err := s2AStream.Send(&s2av2pb.SessionReq{
		LocalIdentity:            localIdentity,
		AuthenticationMechanisms: authMechanisms,
		ReqOneof: &s2av2pb.SessionReq_GetTlsConfigurationReq{
			GetTlsConfigurationReq: &s2av2pb.GetTlsConfigurationReq{
				ConnectionSide: commonpb.ConnectionSide_CONNECTION_SIDE_CLIENT,
			},
		},
	}); err != nil {
		grpclog.Infof("Failed to send request to S2Av2 for client TLS config")
		return nil, err
	}

	// Get the response containing config from S2Av2.
	resp, err := s2AStream.Recv()
	if err != nil {
		grpclog.Infof("Failed to receive client TLS config response from S2Av2.")
		return nil, err
	}

	// TODO(rmehta19): Add unit test for this if statement.
	if (resp.GetStatus() != nil) && (resp.GetStatus().Code != uint32(codes.OK)) {
		return nil, fmt.Errorf("failed to get TLS configuration from S2A: %d, %v", resp.GetStatus().Code, resp.GetStatus().Details)
	}

	// Extract TLS configiguration from SessionResp.
	tlsConfig := resp.GetGetTlsConfigurationResp().GetClientTlsConfiguration()

	var cert tls.Certificate
	for i, v := range tlsConfig.CertificateChain {
		// Populate Certificates field.
		block, _ := pem.Decode([]byte(v))
		if block == nil {
			return nil, errors.New("certificate in CertificateChain obtained from S2Av2 is empty")
		}
		x509Cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return nil, err
		}
		cert.Certificate = append(cert.Certificate, x509Cert.Raw)
		if i == 0 {
			cert.Leaf = x509Cert
		}
	}

	if len(tlsConfig.CertificateChain) > 0 {
		cert.PrivateKey = remotesigner.New(cert.Leaf, s2AStream)
		if cert.PrivateKey == nil {
			return nil, errors.New("failed to retrieve Private Key from Remote Signer Library")
		}
	}

	minVersion, maxVersion, err := getTLSMinMaxVersionsClient(tlsConfig)
	if err != nil {
		return nil, err
	}

	// Create mTLS credentials for client.
	config := &tls.Config{
		VerifyPeerCertificate:  certverifier.VerifyServerCertificateChain(serverHostname, verificationMode, s2AStream, serverAuthorizationPolicy),
		ServerName:             serverHostname,
		InsecureSkipVerify:     true, // NOLINT
		ClientSessionCache:     nil,
		SessionTicketsDisabled: true,
		MinVersion:             minVersion,
		MaxVersion:             maxVersion,
		NextProtos:             []string{h2},
	}
	if len(tlsConfig.CertificateChain) > 0 {
		config.Certificates = []tls.Certificate{cert}
	}
	return config, nil
}

// GetTLSConfigurationForServer returns a tls.Config instance for use by a server application.
func GetTLSConfigurationForServer(s2AStream stream.S2AStream, tokenManager tokenmanager.AccessTokenManager, localIdentities []*commonpbv1.Identity, verificationMode s2av2pb.ValidatePeerCertificateChainReq_VerificationMode) (*tls.Config, error) {
	return &tls.Config{
		GetConfigForClient: ClientConfig(tokenManager, localIdentities, verificationMode, s2AStream),
	}, nil
}

// ClientConfig builds a TLS config for a server to establish a secure
// connection with a client, based on SNI communicated during ClientHello.
// Ensures that server presents the correct certificate to establish a TLS
// connection.
func ClientConfig(tokenManager tokenmanager.AccessTokenManager, localIdentities []*commonpbv1.Identity, verificationMode s2av2pb.ValidatePeerCertificateChainReq_VerificationMode, s2AStream stream.S2AStream) func(chi *tls.ClientHelloInfo) (*tls.Config, error) {
	return func(chi *tls.ClientHelloInfo) (*tls.Config, error) {
		tlsConfig, err := getServerConfigFromS2Av2(tokenManager, localIdentities, chi.ServerName, s2AStream)
		if err != nil {
			return nil, err
		}

		var cert tls.Certificate
		for i, v := range tlsConfig.CertificateChain {
			// Populate Certificates field.
			block, _ := pem.Decode([]byte(v))
			if block == nil {
				return nil, errors.New("certificate in CertificateChain obtained from S2Av2 is empty")
			}
			x509Cert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return nil, err
			}
			cert.Certificate = append(cert.Certificate, x509Cert.Raw)
			if i == 0 {
				cert.Leaf = x509Cert
			}
		}

		cert.PrivateKey = remotesigner.New(cert.Leaf, s2AStream)
		if cert.PrivateKey == nil {
			return nil, errors.New("failed to retrieve Private Key from Remote Signer Library")
		}

		minVersion, maxVersion, err := getTLSMinMaxVersionsServer(tlsConfig)
		if err != nil {
			return nil, err
		}

		clientAuth := getTLSClientAuthType(tlsConfig)

		var cipherSuites []uint16
		cipherSuites = getCipherSuites(tlsConfig.Ciphersuites)

		// Create mTLS credentials for server.
		return &tls.Config{
			Certificates:           []tls.Certificate{cert},
			VerifyPeerCertificate:  certverifier.VerifyClientCertificateChain(verificationMode, s2AStream),
			ClientAuth:             clientAuth,
			CipherSuites:           cipherSuites,
			SessionTicketsDisabled: true,
			MinVersion:             minVersion,
			MaxVersion:             maxVersion,
			NextProtos:             []string{h2},
		}, nil
	}
}

func getCipherSuites(tlsConfigCipherSuites []commonpb.Ciphersuite) []uint16 {
	var tlsGoCipherSuites []uint16
	for _, v := range tlsConfigCipherSuites {
		s := getTLSCipherSuite(v)
		if s != 0xffff {
			tlsGoCipherSuites = append(tlsGoCipherSuites, s)
		}
	}
	return tlsGoCipherSuites
}

func getTLSCipherSuite(tlsCipherSuite commonpb.Ciphersuite) uint16 {
	switch tlsCipherSuite {
	case commonpb.Ciphersuite_CIPHERSUITE_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:
		return tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
	case commonpb.Ciphersuite_CIPHERSUITE_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:
		return tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
	case commonpb.Ciphersuite_CIPHERSUITE_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:
		return tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	case commonpb.Ciphersuite_CIPHERSUITE_ECDHE_RSA_WITH_AES_128_GCM_SHA256:
		return tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	case commonpb.Ciphersuite_CIPHERSUITE_ECDHE_RSA_WITH_AES_256_GCM_SHA384:
		return tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	case commonpb.Ciphersuite_CIPHERSUITE_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:
		return tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	default:
		return 0xffff
	}
}

func getServerConfigFromS2Av2(tokenManager tokenmanager.AccessTokenManager, localIdentities []*commonpbv1.Identity, sni string, s2AStream stream.S2AStream) (*s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration, error) {
	authMechanisms := getAuthMechanisms(tokenManager, localIdentities)
	var locID *commonpbv1.Identity
	if localIdentities != nil {
		locID = localIdentities[0]
	}

	if err := s2AStream.Send(&s2av2pb.SessionReq{
		LocalIdentity:            locID,
		AuthenticationMechanisms: authMechanisms,
		ReqOneof: &s2av2pb.SessionReq_GetTlsConfigurationReq{
			GetTlsConfigurationReq: &s2av2pb.GetTlsConfigurationReq{
				ConnectionSide: commonpb.ConnectionSide_CONNECTION_SIDE_SERVER,
				Sni:            sni,
			},
		},
	}); err != nil {
		return nil, err
	}

	resp, err := s2AStream.Recv()
	if err != nil {
		return nil, err
	}

	// TODO(rmehta19): Add unit test for this if statement.
	if (resp.GetStatus() != nil) && (resp.GetStatus().Code != uint32(codes.OK)) {
		return nil, fmt.Errorf("failed to get TLS configuration from S2A: %d, %v", resp.GetStatus().Code, resp.GetStatus().Details)
	}

	return resp.GetGetTlsConfigurationResp().GetServerTlsConfiguration(), nil
}

func getTLSClientAuthType(tlsConfig *s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration) tls.ClientAuthType {
	var clientAuth tls.ClientAuthType
	switch x := tlsConfig.RequestClientCertificate; x {
	case s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration_DONT_REQUEST_CLIENT_CERTIFICATE:
		clientAuth = tls.NoClientCert
	case s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration_REQUEST_CLIENT_CERTIFICATE_BUT_DONT_VERIFY:
		clientAuth = tls.RequestClientCert
	case s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration_REQUEST_CLIENT_CERTIFICATE_AND_VERIFY:
		// This case actually maps to tls.VerifyClientCertIfGiven. However this
		// mapping triggers normal verification, followed by custom verification,
		// specified in VerifyPeerCertificate. To bypass normal verification, and
		// only do custom verification we set clientAuth to RequireAnyClientCert or
		// RequestClientCert. See https://github.com/google/s2a-go/pull/43 for full
		// discussion.
		clientAuth = tls.RequireAnyClientCert
	case s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_BUT_DONT_VERIFY:
		clientAuth = tls.RequireAnyClientCert
	case s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY:
		// This case actually maps to tls.RequireAndVerifyClientCert. However this
		// mapping triggers normal verification, followed by custom verification,
		// specified in VerifyPeerCertificate. To bypass normal verification, and
		// only do custom verification we set clientAuth to RequireAnyClientCert or
		// RequestClientCert. See https://github.com/google/s2a-go/pull/43 for full
		// discussion.
		clientAuth = tls.RequireAnyClientCert
	default:
		clientAuth = tls.RequireAnyClientCert
	}
	return clientAuth
}

func getAuthMechanisms(tokenManager tokenmanager.AccessTokenManager, localIdentities []*commonpbv1.Identity) []*s2av2pb.AuthenticationMechanism {
	if tokenManager == nil {
		return nil
	}
	if len(localIdentities) == 0 {
		token, err := tokenManager.DefaultToken()
		if err != nil {
			grpclog.Infof("Unable to get token for empty local identity: %v", err)
			return nil
		}
		return []*s2av2pb.AuthenticationMechanism{
			{
				MechanismOneof: &s2av2pb.AuthenticationMechanism_Token{
					Token: token,
				},
			},
		}
	}
	var authMechanisms []*s2av2pb.AuthenticationMechanism
	for _, localIdentity := range localIdentities {
		if localIdentity == nil {
			token, err := tokenManager.DefaultToken()
			if err != nil {
				grpclog.Infof("Unable to get default token for local identity %v: %v", localIdentity, err)
				continue
			}
			authMechanisms = append(authMechanisms, &s2av2pb.AuthenticationMechanism{
				Identity: localIdentity,
				MechanismOneof: &s2av2pb.AuthenticationMechanism_Token{
					Token: token,
				},
			})
		} else {
			token, err := tokenManager.Token(localIdentity)
			if err != nil {
				grpclog.Infof("Unable to get token for local identity %v: %v", localIdentity, err)
				continue
			}
			authMechanisms = append(authMechanisms, &s2av2pb.AuthenticationMechanism{
				Identity: localIdentity,
				MechanismOneof: &s2av2pb.AuthenticationMechanism_Token{
					Token: token,
				},
			})
		}
	}
	return authMechanisms
}

// TODO(rmehta19): refactor switch statements into a helper function.
func getTLSMinMaxVersionsClient(tlsConfig *s2av2pb.GetTlsConfigurationResp_ClientTlsConfiguration) (uint16, uint16, error) {
	// Map S2Av2 TLSVersion to consts defined in tls package.
	var minVersion uint16
	var maxVersion uint16
	switch x := tlsConfig.MinTlsVersion; x {
	case commonpb.TLSVersion_TLS_VERSION_1_0:
		minVersion = tls.VersionTLS10
	case commonpb.TLSVersion_TLS_VERSION_1_1:
		minVersion = tls.VersionTLS11
	case commonpb.TLSVersion_TLS_VERSION_1_2:
		minVersion = tls.VersionTLS12
	case commonpb.TLSVersion_TLS_VERSION_1_3:
		minVersion = tls.VersionTLS13
	default:
		return minVersion, maxVersion, fmt.Errorf("S2Av2 provided invalid MinTlsVersion: %v", x)
	}

	switch x := tlsConfig.MaxTlsVersion; x {
	case commonpb.TLSVersion_TLS_VERSION_1_0:
		maxVersion = tls.VersionTLS10
	case commonpb.TLSVersion_TLS_VERSION_1_1:
		maxVersion = tls.VersionTLS11
	case commonpb.TLSVersion_TLS_VERSION_1_2:
		maxVersion = tls.VersionTLS12
	case commonpb.TLSVersion_TLS_VERSION_1_3:
		maxVersion = tls.VersionTLS13
	default:
		return minVersion, maxVersion, fmt.Errorf("S2Av2 provided invalid MaxTlsVersion: %v", x)
	}
	if minVersion > maxVersion {
		return minVersion, maxVersion, errors.New("S2Av2 provided minVersion > maxVersion")
	}
	return minVersion, maxVersion, nil
}

func getTLSMinMaxVersionsServer(tlsConfig *s2av2pb.GetTlsConfigurationResp_ServerTlsConfiguration) (uint16, uint16, error) {
	// Map S2Av2 TLSVersion to consts defined in tls package.
	var minVersion uint16
	var maxVersion uint16
	switch x := tlsConfig.MinTlsVersion; x {
	case commonpb.TLSVersion_TLS_VERSION_1_0:
		minVersion = tls.VersionTLS10
	case commonpb.TLSVersion_TLS_VERSION_1_1:
		minVersion = tls.VersionTLS11
	case commonpb.TLSVersion_TLS_VERSION_1_2:
		minVersion = tls.VersionTLS12
	case commonpb.TLSVersion_TLS_VERSION_1_3:
		minVersion = tls.VersionTLS13
	default:
		return minVersion, maxVersion, fmt.Errorf("S2Av2 provided invalid MinTlsVersion: %v", x)
	}

	switch x := tlsConfig.MaxTlsVersion; x {
	case commonpb.TLSVersion_TLS_VERSION_1_0:
		maxVersion = tls.VersionTLS10
	case commonpb.TLSVersion_TLS_VERSION_1_1:
		maxVersion = tls.VersionTLS11
	case commonpb.TLSVersion_TLS_VERSION_1_2:
		maxVersion = tls.VersionTLS12
	case commonpb.TLSVersion_TLS_VERSION_1_3:
		maxVersion = tls.VersionTLS13
	default:
		return minVersion, maxVersion, fmt.Errorf("S2Av2 provided invalid MaxTlsVersion: %v", x)
	}
	if minVersion > maxVersion {
		return minVersion, maxVersion, errors.New("S2Av2 provided minVersion > maxVersion")
	}
	return minVersion, maxVersion, nil
}
