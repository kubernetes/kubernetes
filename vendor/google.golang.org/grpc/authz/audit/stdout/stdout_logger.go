/*
 *
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
 *
 */

// Package stdout defines an stdout audit logger.
package stdout

import (
	"encoding/json"
	"log"
	"os"
	"time"

	"google.golang.org/grpc/authz/audit"
	"google.golang.org/grpc/grpclog"
)

var grpcLogger = grpclog.Component("authz-audit")

// Name is the string to identify this logger type in the registry
const Name = "stdout_logger"

func init() {
	audit.RegisterLoggerBuilder(&loggerBuilder{
		goLogger: log.New(os.Stdout, "", 0),
	})
}

type event struct {
	FullMethodName string `json:"rpc_method"`
	Principal      string `json:"principal"`
	PolicyName     string `json:"policy_name"`
	MatchedRule    string `json:"matched_rule"`
	Authorized     bool   `json:"authorized"`
	Timestamp      string `json:"timestamp"` // Time when the audit event is logged via Log method
}

// logger implements the audit.logger interface by logging to standard output.
type logger struct {
	goLogger *log.Logger
}

// Log marshals the audit.Event to json and prints it to standard output.
func (l *logger) Log(event *audit.Event) {
	jsonContainer := map[string]any{
		"grpc_audit_log": convertEvent(event),
	}
	jsonBytes, err := json.Marshal(jsonContainer)
	if err != nil {
		grpcLogger.Errorf("failed to marshal AuditEvent data to JSON: %v", err)
		return
	}
	l.goLogger.Println(string(jsonBytes))
}

// loggerConfig represents the configuration for the stdout logger.
// It is currently empty and implements the audit.Logger interface by embedding it.
type loggerConfig struct {
	audit.LoggerConfig
}

type loggerBuilder struct {
	goLogger *log.Logger
}

func (loggerBuilder) Name() string {
	return Name
}

// Build returns a new instance of the stdout logger.
// Passed in configuration is ignored as the stdout logger does not
// expect any configuration to be provided.
func (lb *loggerBuilder) Build(audit.LoggerConfig) audit.Logger {
	return &logger{
		goLogger: lb.goLogger,
	}
}

// ParseLoggerConfig is a no-op since the stdout logger does not accept any configuration.
func (*loggerBuilder) ParseLoggerConfig(config json.RawMessage) (audit.LoggerConfig, error) {
	if len(config) != 0 && string(config) != "{}" {
		grpcLogger.Warningf("Stdout logger doesn't support custom configs. Ignoring:\n%s", string(config))
	}
	return &loggerConfig{}, nil
}

func convertEvent(auditEvent *audit.Event) *event {
	return &event{
		FullMethodName: auditEvent.FullMethodName,
		Principal:      auditEvent.Principal,
		PolicyName:     auditEvent.PolicyName,
		MatchedRule:    auditEvent.MatchedRule,
		Authorized:     auditEvent.Authorized,
		Timestamp:      time.Now().Format(time.RFC3339Nano),
	}
}
