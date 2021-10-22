/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package main

import (
	"flag"
	"net"
	"strconv"

	"google.golang.org/grpc"
	_ "google.golang.org/grpc/balancer/grpclb"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/alts"
	"google.golang.org/grpc/credentials/google"
	"google.golang.org/grpc/credentials/oauth"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/interop"
	testpb "google.golang.org/grpc/interop/grpc_testing"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/testdata"
)

const (
	googleDefaultCredsName = "google_default_credentials"
	computeEngineCredsName = "compute_engine_channel_creds"
)

var (
	caFile                = flag.String("ca_file", "", "The file containning the CA root cert file")
	useTLS                = flag.Bool("use_tls", false, "Connection uses TLS if true")
	useALTS               = flag.Bool("use_alts", false, "Connection uses ALTS if true (this option can only be used on GCP)")
	customCredentialsType = flag.String("custom_credentials_type", "", "Custom creds to use, excluding TLS or ALTS")
	altsHSAddr            = flag.String("alts_handshaker_service_address", "", "ALTS handshaker gRPC service address")
	testCA                = flag.Bool("use_test_ca", false, "Whether to replace platform root CAs with test CA as the CA root")
	serviceAccountKeyFile = flag.String("service_account_key_file", "", "Path to service account json key file")
	oauthScope            = flag.String("oauth_scope", "", "The scope for OAuth2 tokens")
	defaultServiceAccount = flag.String("default_service_account", "", "Email of GCE default service account")
	serverHost            = flag.String("server_host", "localhost", "The server host name")
	serverPort            = flag.Int("server_port", 10000, "The server port number")
	tlsServerName         = flag.String("server_host_override", "", "The server name use to verify the hostname returned by TLS handshake if it is not empty. Otherwise, --server_host is used.")
	testCase              = flag.String("test_case", "large_unary",
		`Configure different test cases. Valid options are:
        empty_unary : empty (zero bytes) request and response;
        large_unary : single request and (large) response;
        client_streaming : request streaming with single response;
        server_streaming : single request with response streaming;
        ping_pong : full-duplex streaming;
        empty_stream : full-duplex streaming with zero message;
        timeout_on_sleeping_server: fullduplex streaming on a sleeping server;
        compute_engine_creds: large_unary with compute engine auth;
        service_account_creds: large_unary with service account auth;
        jwt_token_creds: large_unary with jwt token auth;
        per_rpc_creds: large_unary with per rpc token;
        oauth2_auth_token: large_unary with oauth2 token auth;
        google_default_credentials: large_unary with google default credentials
        compute_engine_channel_credentials: large_unary with compute engine creds
        cancel_after_begin: cancellation after metadata has been sent but before payloads are sent;
        cancel_after_first_response: cancellation after receiving 1st message from the server;
        status_code_and_message: status code propagated back to client;
        special_status_message: Unicode and whitespace is correctly processed in status message;
        custom_metadata: server will echo custom metadata;
        unimplemented_method: client attempts to call unimplemented method;
        unimplemented_service: client attempts to call unimplemented service;
        pick_first_unary: all requests are sent to one server despite multiple servers are resolved.`)
)

type credsMode uint8

const (
	credsNone credsMode = iota
	credsTLS
	credsALTS
	credsGoogleDefaultCreds
	credsComputeEngineCreds
)

func main() {
	flag.Parse()
	var useGDC bool // use google default creds
	var useCEC bool // use compute engine creds
	if *customCredentialsType != "" {
		switch *customCredentialsType {
		case googleDefaultCredsName:
			useGDC = true
		case computeEngineCredsName:
			useCEC = true
		default:
			grpclog.Fatalf("If set, custom_credentials_type can only be set to one of %v or %v",
				googleDefaultCredsName, computeEngineCredsName)
		}
	}
	if (*useTLS && *useALTS) || (*useTLS && useGDC) || (*useALTS && useGDC) || (*useTLS && useCEC) || (*useALTS && useCEC) {
		grpclog.Fatalf("only one of TLS, ALTS, google default creds, or compute engine creds can be used")
	}

	var credsChosen credsMode
	switch {
	case *useTLS:
		credsChosen = credsTLS
	case *useALTS:
		credsChosen = credsALTS
	case useGDC:
		credsChosen = credsGoogleDefaultCreds
	case useCEC:
		credsChosen = credsComputeEngineCreds
	}

	resolver.SetDefaultScheme("dns")
	serverAddr := net.JoinHostPort(*serverHost, strconv.Itoa(*serverPort))
	var opts []grpc.DialOption
	switch credsChosen {
	case credsTLS:
		var sn string
		if *tlsServerName != "" {
			sn = *tlsServerName
		}
		var creds credentials.TransportCredentials
		if *testCA {
			var err error
			if *caFile == "" {
				*caFile = testdata.Path("ca.pem")
			}
			creds, err = credentials.NewClientTLSFromFile(*caFile, sn)
			if err != nil {
				grpclog.Fatalf("Failed to create TLS credentials %v", err)
			}
		} else {
			creds = credentials.NewClientTLSFromCert(nil, sn)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	case credsALTS:
		altsOpts := alts.DefaultClientOptions()
		if *altsHSAddr != "" {
			altsOpts.HandshakerServiceAddress = *altsHSAddr
		}
		altsTC := alts.NewClientCreds(altsOpts)
		opts = append(opts, grpc.WithTransportCredentials(altsTC))
	case credsGoogleDefaultCreds:
		opts = append(opts, grpc.WithCredentialsBundle(google.NewDefaultCredentials()))
	case credsComputeEngineCreds:
		opts = append(opts, grpc.WithCredentialsBundle(google.NewComputeEngineCredentials()))
	case credsNone:
		opts = append(opts, grpc.WithInsecure())
	default:
		grpclog.Fatal("Invalid creds")
	}
	if credsChosen == credsTLS {
		if *testCase == "compute_engine_creds" {
			opts = append(opts, grpc.WithPerRPCCredentials(oauth.NewComputeEngine()))
		} else if *testCase == "service_account_creds" {
			jwtCreds, err := oauth.NewServiceAccountFromFile(*serviceAccountKeyFile, *oauthScope)
			if err != nil {
				grpclog.Fatalf("Failed to create JWT credentials: %v", err)
			}
			opts = append(opts, grpc.WithPerRPCCredentials(jwtCreds))
		} else if *testCase == "jwt_token_creds" {
			jwtCreds, err := oauth.NewJWTAccessFromFile(*serviceAccountKeyFile)
			if err != nil {
				grpclog.Fatalf("Failed to create JWT credentials: %v", err)
			}
			opts = append(opts, grpc.WithPerRPCCredentials(jwtCreds))
		} else if *testCase == "oauth2_auth_token" {
			opts = append(opts, grpc.WithPerRPCCredentials(oauth.NewOauthAccess(interop.GetToken(*serviceAccountKeyFile, *oauthScope))))
		}
	}
	opts = append(opts, grpc.WithBlock())
	conn, err := grpc.Dial(serverAddr, opts...)
	if err != nil {
		grpclog.Fatalf("Fail to dial: %v", err)
	}
	defer conn.Close()
	tc := testpb.NewTestServiceClient(conn)
	switch *testCase {
	case "empty_unary":
		interop.DoEmptyUnaryCall(tc)
		grpclog.Infoln("EmptyUnaryCall done")
	case "large_unary":
		interop.DoLargeUnaryCall(tc)
		grpclog.Infoln("LargeUnaryCall done")
	case "client_streaming":
		interop.DoClientStreaming(tc)
		grpclog.Infoln("ClientStreaming done")
	case "server_streaming":
		interop.DoServerStreaming(tc)
		grpclog.Infoln("ServerStreaming done")
	case "ping_pong":
		interop.DoPingPong(tc)
		grpclog.Infoln("Pingpong done")
	case "empty_stream":
		interop.DoEmptyStream(tc)
		grpclog.Infoln("Emptystream done")
	case "timeout_on_sleeping_server":
		interop.DoTimeoutOnSleepingServer(tc)
		grpclog.Infoln("TimeoutOnSleepingServer done")
	case "compute_engine_creds":
		if credsChosen != credsTLS {
			grpclog.Fatalf("TLS credentials need to be set for compute_engine_creds test case.")
		}
		interop.DoComputeEngineCreds(tc, *defaultServiceAccount, *oauthScope)
		grpclog.Infoln("ComputeEngineCreds done")
	case "service_account_creds":
		if credsChosen != credsTLS {
			grpclog.Fatalf("TLS credentials need to be set for service_account_creds test case.")
		}
		interop.DoServiceAccountCreds(tc, *serviceAccountKeyFile, *oauthScope)
		grpclog.Infoln("ServiceAccountCreds done")
	case "jwt_token_creds":
		if credsChosen != credsTLS {
			grpclog.Fatalf("TLS credentials need to be set for jwt_token_creds test case.")
		}
		interop.DoJWTTokenCreds(tc, *serviceAccountKeyFile)
		grpclog.Infoln("JWTtokenCreds done")
	case "per_rpc_creds":
		if credsChosen != credsTLS {
			grpclog.Fatalf("TLS credentials need to be set for per_rpc_creds test case.")
		}
		interop.DoPerRPCCreds(tc, *serviceAccountKeyFile, *oauthScope)
		grpclog.Infoln("PerRPCCreds done")
	case "oauth2_auth_token":
		if credsChosen != credsTLS {
			grpclog.Fatalf("TLS credentials need to be set for oauth2_auth_token test case.")
		}
		interop.DoOauth2TokenCreds(tc, *serviceAccountKeyFile, *oauthScope)
		grpclog.Infoln("Oauth2TokenCreds done")
	case "google_default_credentials":
		if credsChosen != credsGoogleDefaultCreds {
			grpclog.Fatalf("GoogleDefaultCredentials need to be set for google_default_credentials test case.")
		}
		interop.DoGoogleDefaultCredentials(tc, *defaultServiceAccount)
		grpclog.Infoln("GoogleDefaultCredentials done")
	case "compute_engine_channel_credentials":
		if credsChosen != credsComputeEngineCreds {
			grpclog.Fatalf("ComputeEngineCreds need to be set for compute_engine_channel_credentials test case.")
		}
		interop.DoComputeEngineChannelCredentials(tc, *defaultServiceAccount)
		grpclog.Infoln("ComputeEngineChannelCredentials done")
	case "cancel_after_begin":
		interop.DoCancelAfterBegin(tc)
		grpclog.Infoln("CancelAfterBegin done")
	case "cancel_after_first_response":
		interop.DoCancelAfterFirstResponse(tc)
		grpclog.Infoln("CancelAfterFirstResponse done")
	case "status_code_and_message":
		interop.DoStatusCodeAndMessage(tc)
		grpclog.Infoln("StatusCodeAndMessage done")
	case "special_status_message":
		interop.DoSpecialStatusMessage(tc)
		grpclog.Infoln("SpecialStatusMessage done")
	case "custom_metadata":
		interop.DoCustomMetadata(tc)
		grpclog.Infoln("CustomMetadata done")
	case "unimplemented_method":
		interop.DoUnimplementedMethod(conn)
		grpclog.Infoln("UnimplementedMethod done")
	case "unimplemented_service":
		interop.DoUnimplementedService(testpb.NewUnimplementedServiceClient(conn))
		grpclog.Infoln("UnimplementedService done")
	case "pick_first_unary":
		interop.DoPickFirstUnary(tc)
		grpclog.Infoln("PickFirstUnary done")
	default:
		grpclog.Fatal("Unsupported test case: ", *testCase)
	}
}
