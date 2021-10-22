/*
 *
 * Copyright 2019 gRPC authors.
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

package bootstrap

import (
	"os"
	"testing"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/google"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	structpb "github.com/golang/protobuf/ptypes/struct"
)

var (
	nodeProto = &corepb.Node{
		Id: "ENVOY_NODE_ID",
		Metadata: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"TRAFFICDIRECTOR_GRPC_HOSTNAME": {
					Kind: &structpb.Value_StringValue{StringValue: "trafficdirector"},
				},
			},
		},
		BuildVersion: gRPCVersion,
	}
	nilCredsConfig = &Config{
		BalancerName: "trafficdirector.googleapis.com:443",
		Creds:        nil,
		NodeProto:    nodeProto,
	}
	nonNilCredsConfig = &Config{
		BalancerName: "trafficdirector.googleapis.com:443",
		Creds:        grpc.WithCredentialsBundle(google.NewComputeEngineCredentials()),
		NodeProto:    nodeProto,
	}
)

// TestNewConfig exercises the functionality in NewConfig with different
// bootstrap file contents. It overrides the fileReadFunc by returning
// bootstrap file contents defined in this test, instead of reading from a
// file.
func TestNewConfig(t *testing.T) {
	bootstrapFileMap := map[string]string{
		"empty":          "",
		"badJSON":        `["test": 123]`,
		"noBalancerName": `{"node": {"id": "ENVOY_NODE_ID"}}`,
		"emptyNodeProto": `
		{
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443"
			}]
		}`,
		"emptyXdsServer": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			}
		}`,
		"unknownTopLevelFieldInFile": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "not-google-default" }
				]
			}],
			"unknownField": "foobar"
		}`,
		"unknownFieldInNodeProto": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"unknownField": "foobar",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443"
			}]
		}`,
		"unknownFieldInXdsServer": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "not-google-default" }
				],
				"unknownField": "foobar"
			}]
		}`,
		"emptyChannelCreds": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443"
			}]
		}`,
		"nonGoogleDefaultCreds": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "not-google-default" }
				]
			}]
		}`,
		"multipleChannelCreds": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "not-google-default" },
					{ "type": "google_default" }
				]
			}]
		}`,
		"goodBootstrap": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "google_default" }
				]
			}]
		}`,
		"multipleXDSServers": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [
				{
					"server_uri": "trafficdirector.googleapis.com:443",
					"channel_creds": [{ "type": "google_default" }]
				},
				{
					"server_uri": "backup.never.use.com:1234",
					"channel_creds": [{ "type": "not-google-default" }]
				}
			]
		}`,
	}

	oldFileReadFunc := fileReadFunc
	fileReadFunc = func(name string) ([]byte, error) {
		if b, ok := bootstrapFileMap[name]; ok {
			return []byte(b), nil
		}
		return nil, os.ErrNotExist
	}
	defer func() {
		fileReadFunc = oldFileReadFunc
		os.Unsetenv(fileEnv)
	}()

	tests := []struct {
		name       string
		wantConfig *Config
		wantError  bool
	}{
		{"nonExistentBootstrapFile", nil, true},
		{"empty", nil, true},
		{"badJSON", nil, true},
		{"emptyNodeProto", &Config{
			BalancerName: "trafficdirector.googleapis.com:443",
			NodeProto:    &corepb.Node{BuildVersion: gRPCVersion},
		}, false},
		{"noBalancerName", nil, true},
		{"emptyXdsServer", nil, true},
		{"unknownTopLevelFieldInFile", nilCredsConfig, false},
		{"unknownFieldInNodeProto", nilCredsConfig, false},
		{"unknownFieldInXdsServer", nilCredsConfig, false},
		{"emptyChannelCreds", nilCredsConfig, false},
		{"nonGoogleDefaultCreds", nilCredsConfig, false},
		{"multipleChannelCreds", nonNilCredsConfig, false},
		{"goodBootstrap", nonNilCredsConfig, false},
		{"multipleXDSServers", nonNilCredsConfig, false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := os.Setenv(fileEnv, test.name); err != nil {
				t.Fatalf("os.Setenv(%s, %s) failed with error: %v", fileEnv, test.name, err)
			}
			config, err := NewConfig()
			if err != nil {
				if !test.wantError {
					t.Fatalf("unexpected error %v", err)
				}
				return
			}
			if test.wantError {
				t.Fatalf("wantError: %v, got error %v", test.wantError, err)
			}
			if config.BalancerName != test.wantConfig.BalancerName {
				t.Errorf("config.BalancerName is %s, want %s", config.BalancerName, test.wantConfig.BalancerName)
			}
			if !proto.Equal(config.NodeProto, test.wantConfig.NodeProto) {
				t.Errorf("config.NodeProto is %#v, want %#v", config.NodeProto, test.wantConfig.NodeProto)
			}
			if (config.Creds != nil) != (test.wantConfig.Creds != nil) {
				t.Errorf("config.Creds is %#v, want %#v", config.Creds, test.wantConfig.Creds)
			}
		})
	}
}

func TestNewConfigEnvNotSet(t *testing.T) {
	os.Unsetenv(fileEnv)
	config, err := NewConfig()
	if err == nil {
		t.Errorf("NewConfig() returned: %#v, <nil>, wanted non-nil error", config)
	}
}
