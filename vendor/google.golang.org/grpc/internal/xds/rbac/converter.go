/*
 * Copyright 2023 gRPC authors.
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
 */

package rbac

import (
	"encoding/json"
	"fmt"
	"strings"

	v1xdsudpatypepb "github.com/cncf/xds/go/udpa/type/v1"
	v3xdsxdstypepb "github.com/cncf/xds/go/xds/type/v3"
	v3rbacpb "github.com/envoyproxy/go-control-plane/envoy/config/rbac/v3"
	v3auditloggersstreampb "github.com/envoyproxy/go-control-plane/envoy/extensions/rbac/audit_loggers/stream/v3"
	"google.golang.org/grpc/authz/audit"
	"google.golang.org/grpc/authz/audit/stdout"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/structpb"
)

func buildLogger(loggerConfig *v3rbacpb.RBAC_AuditLoggingOptions_AuditLoggerConfig) (audit.Logger, error) {
	if loggerConfig.GetAuditLogger().GetTypedConfig() == nil {
		return nil, fmt.Errorf("missing required field: TypedConfig")
	}
	customConfig, loggerName, err := getCustomConfig(loggerConfig.AuditLogger.TypedConfig)
	if err != nil {
		return nil, err
	}
	if loggerName == "" {
		return nil, fmt.Errorf("field TypedConfig.TypeURL cannot be an empty string")
	}
	factory := audit.GetLoggerBuilder(loggerName)
	if factory == nil {
		if loggerConfig.IsOptional {
			return nil, nil
		}
		return nil, fmt.Errorf("no builder registered for %v", loggerName)
	}
	auditLoggerConfig, err := factory.ParseLoggerConfig(customConfig)
	if err != nil {
		return nil, fmt.Errorf("custom config could not be parsed by registered factory. error: %v", err)
	}
	auditLogger := factory.Build(auditLoggerConfig)
	return auditLogger, nil
}

func getCustomConfig(config *anypb.Any) (json.RawMessage, string, error) {
	c, err := config.UnmarshalNew()
	if err != nil {
		return nil, "", err
	}
	switch m := c.(type) {
	case *v1xdsudpatypepb.TypedStruct:
		return convertCustomConfig(m.TypeUrl, m.Value)
	case *v3xdsxdstypepb.TypedStruct:
		return convertCustomConfig(m.TypeUrl, m.Value)
	case *v3auditloggersstreampb.StdoutAuditLog:
		return convertStdoutConfig(m)
	}
	return nil, "", fmt.Errorf("custom config not implemented for type [%v]", config.GetTypeUrl())
}

func convertStdoutConfig(config *v3auditloggersstreampb.StdoutAuditLog) (json.RawMessage, string, error) {
	json, err := protojson.Marshal(config)
	return json, stdout.Name, err
}

func convertCustomConfig(typeURL string, s *structpb.Struct) (json.RawMessage, string, error) {
	// The gRPC policy name will be the "type name" part of the value of the
	// type_url field in the TypedStruct. We get this by using the part after
	// the last / character. Can assume a valid type_url from the control plane.
	urls := strings.Split(typeURL, "/")
	if len(urls) == 0 {
		return nil, "", fmt.Errorf("error converting custom audit logger %v for %v: typeURL must have a url-like format with the typeName being the value after the last /", typeURL, s)
	}
	name := urls[len(urls)-1]

	rawJSON := []byte("{}")
	var err error
	if s != nil {
		rawJSON, err = json.Marshal(s)
		if err != nil {
			return nil, "", fmt.Errorf("error converting custom audit logger %v for %v: %v", typeURL, s, err)
		}
	}
	return rawJSON, name, nil
}
