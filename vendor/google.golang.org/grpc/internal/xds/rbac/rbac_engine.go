/*
 * Copyright 2021 gRPC authors.
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

// Package rbac provides service-level and method-level access control for a
// service. See
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/rbac/v3/rbac.proto#role-based-access-control-rbac
// for documentation.
package rbac

import (
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"strconv"

	"google.golang.org/grpc"
	"google.golang.org/grpc/authz/audit"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	v3rbacpb "github.com/envoyproxy/go-control-plane/envoy/config/rbac/v3"
)

var logger = grpclog.Component("rbac")

var getConnection = transport.GetConnection

// ChainEngine represents a chain of RBAC Engines, used to make authorization
// decisions on incoming RPCs.
type ChainEngine struct {
	chainedEngines []*engine
}

// NewChainEngine returns a chain of RBAC engines, used to make authorization
// decisions on incoming RPCs. Returns a non-nil error for invalid policies.
func NewChainEngine(policies []*v3rbacpb.RBAC, policyName string) (*ChainEngine, error) {
	engines := make([]*engine, 0, len(policies))
	for _, policy := range policies {
		engine, err := newEngine(policy, policyName)
		if err != nil {
			return nil, err
		}
		engines = append(engines, engine)
	}
	return &ChainEngine{chainedEngines: engines}, nil
}

func (cre *ChainEngine) logRequestDetails(rpcData *rpcData) {
	if logger.V(2) {
		logger.Infof("checking request: url path=%s", rpcData.fullMethod)
		if len(rpcData.certs) > 0 {
			cert := rpcData.certs[0]
			logger.Infof("uri sans=%q, dns sans=%q, subject=%v", cert.URIs, cert.DNSNames, cert.Subject)
		}
	}
}

// IsAuthorized determines if an incoming RPC is authorized based on the chain of RBAC
// engines and their associated actions.
//
// Errors returned by this function are compatible with the status package.
func (cre *ChainEngine) IsAuthorized(ctx context.Context) error {
	// This conversion step (i.e. pulling things out of ctx) can be done once,
	// and then be used for the whole chain of RBAC Engines.
	rpcData, err := newRPCData(ctx)
	if err != nil {
		logger.Errorf("newRPCData: %v", err)
		return status.Errorf(codes.Internal, "gRPC RBAC: %v", err)
	}
	for _, engine := range cre.chainedEngines {
		matchingPolicyName, ok := engine.findMatchingPolicy(rpcData)
		if logger.V(2) && ok {
			logger.Infof("incoming RPC matched to policy %v in engine with action %v", matchingPolicyName, engine.action)
		}

		switch {
		case engine.action == v3rbacpb.RBAC_ALLOW && !ok:
			cre.logRequestDetails(rpcData)
			engine.doAuditLogging(rpcData, matchingPolicyName, false)
			return status.Errorf(codes.PermissionDenied, "incoming RPC did not match an allow policy")
		case engine.action == v3rbacpb.RBAC_DENY && ok:
			cre.logRequestDetails(rpcData)
			engine.doAuditLogging(rpcData, matchingPolicyName, false)
			return status.Errorf(codes.PermissionDenied, "incoming RPC matched a deny policy %q", matchingPolicyName)
		}
		// Every policy in the engine list must be queried. Thus, iterate to the
		// next policy.
		engine.doAuditLogging(rpcData, matchingPolicyName, true)
	}
	// If the incoming RPC gets through all of the engines successfully (i.e.
	// doesn't not match an allow or match a deny engine), the RPC is authorized
	// to proceed.
	return nil
}

// engine is used for matching incoming RPCs to policies.
type engine struct {
	// TODO(gtcooke94) - differentiate between `policyName`, `policies`, and `rules`
	policyName string
	policies   map[string]*policyMatcher
	// action must be ALLOW or DENY.
	action         v3rbacpb.RBAC_Action
	auditLoggers   []audit.Logger
	auditCondition v3rbacpb.RBAC_AuditLoggingOptions_AuditCondition
}

// newEngine creates an RBAC Engine based on the contents of a policy. Returns a
// non-nil error if the policy is invalid.
func newEngine(config *v3rbacpb.RBAC, policyName string) (*engine, error) {
	a := config.GetAction()
	if a != v3rbacpb.RBAC_ALLOW && a != v3rbacpb.RBAC_DENY {
		return nil, fmt.Errorf("unsupported action %s", config.Action)
	}

	policies := make(map[string]*policyMatcher, len(config.GetPolicies()))
	for name, policy := range config.GetPolicies() {
		matcher, err := newPolicyMatcher(policy)
		if err != nil {
			return nil, err
		}
		policies[name] = matcher
	}

	auditLoggers, auditCondition, err := parseAuditOptions(config.GetAuditLoggingOptions())
	if err != nil {
		return nil, err
	}
	return &engine{
		policyName:     policyName,
		policies:       policies,
		action:         a,
		auditLoggers:   auditLoggers,
		auditCondition: auditCondition,
	}, nil
}

func parseAuditOptions(opts *v3rbacpb.RBAC_AuditLoggingOptions) ([]audit.Logger, v3rbacpb.RBAC_AuditLoggingOptions_AuditCondition, error) {
	if opts == nil {
		return nil, v3rbacpb.RBAC_AuditLoggingOptions_NONE, nil
	}
	var auditLoggers []audit.Logger
	for _, logger := range opts.LoggerConfigs {
		auditLogger, err := buildLogger(logger)
		if err != nil {
			return nil, v3rbacpb.RBAC_AuditLoggingOptions_NONE, err
		}
		if auditLogger == nil {
			// This occurs when the audit logger is not registered but also
			// marked optional.
			continue
		}
		auditLoggers = append(auditLoggers, auditLogger)
	}
	return auditLoggers, opts.GetAuditCondition(), nil

}

// findMatchingPolicy determines if an incoming RPC matches a policy. On a
// successful match, it returns the name of the matching policy and a true bool
// to specify that there was a matching policy found.  It returns false in
// the case of not finding a matching policy.
func (e *engine) findMatchingPolicy(rpcData *rpcData) (string, bool) {
	for policy, matcher := range e.policies {
		if matcher.match(rpcData) {
			return policy, true
		}
	}
	return "", false
}

// newRPCData takes an incoming context (should be a context representing state
// needed for server RPC Call with metadata, peer info (used for source ip/port
// and TLS information) and connection (used for destination ip/port) piped into
// it) and the method name of the Service being called server side and populates
// an rpcData struct ready to be passed to the RBAC Engine to find a matching
// policy.
func newRPCData(ctx context.Context) (*rpcData, error) {
	// The caller should populate all of these fields (i.e. for empty headers,
	// pipe an empty md into context).
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, errors.New("missing metadata in incoming context")
	}
	// ":method can be hard-coded to POST if unavailable" - A41
	md[":method"] = []string{"POST"}
	// "If the transport exposes TE in Metadata, then RBAC must special-case the
	// header to treat it as not present." - A41
	delete(md, "TE")

	pi, ok := peer.FromContext(ctx)
	if !ok {
		return nil, errors.New("missing peer info in incoming context")
	}

	// The methodName will be available in the passed in ctx from a unary or streaming
	// interceptor, as grpc.Server pipes in a transport stream which contains the methodName
	// into contexts available in both unary or streaming interceptors.
	mn, ok := grpc.Method(ctx)
	if !ok {
		return nil, errors.New("missing method in incoming context")
	}
	// gRPC-Go strips :path from the headers given to the application, but RBAC should be
	// able to match against it.
	md[":path"] = []string{mn}

	// The connection is needed in order to find the destination address and
	// port of the incoming RPC Call.
	conn := getConnection(ctx)
	if conn == nil {
		return nil, errors.New("missing connection in incoming context")
	}
	_, dPort, err := net.SplitHostPort(conn.LocalAddr().String())
	if err != nil {
		return nil, fmt.Errorf("error parsing local address: %v", err)
	}
	dp, err := strconv.ParseUint(dPort, 10, 32)
	if err != nil {
		return nil, fmt.Errorf("error parsing local address: %v", err)
	}

	var authType string
	var peerCertificates []*x509.Certificate
	if tlsInfo, ok := pi.AuthInfo.(credentials.TLSInfo); ok {
		authType = pi.AuthInfo.AuthType()
		peerCertificates = tlsInfo.State.PeerCertificates
	}

	return &rpcData{
		md:              md,
		peerInfo:        pi,
		fullMethod:      mn,
		destinationPort: uint32(dp),
		localAddr:       conn.LocalAddr(),
		authType:        authType,
		certs:           peerCertificates,
	}, nil
}

// rpcData wraps data pulled from an incoming RPC that the RBAC engine needs to
// find a matching policy.
type rpcData struct {
	// md is the HTTP Headers that are present in the incoming RPC.
	md metadata.MD
	// peerInfo is information about the downstream peer.
	peerInfo *peer.Peer
	// fullMethod is the method name being called on the upstream service.
	fullMethod string
	// destinationPort is the port that the RPC is being sent to on the
	// server.
	destinationPort uint32
	// localAddr is the address that the RPC is being sent to.
	localAddr net.Addr
	// authType is the type of authentication e.g. "tls".
	authType string
	// certs are the certificates presented by the peer during a TLS
	// handshake.
	certs []*x509.Certificate
}

func (e *engine) doAuditLogging(rpcData *rpcData, rule string, authorized bool) {
	// In the RBAC world, we need to have a SPIFFE ID as the principal for this
	// to be meaningful
	principal := ""
	if rpcData.peerInfo != nil {
		// If AuthType = tls, then we can cast AuthInfo to TLSInfo.
		if tlsInfo, ok := rpcData.peerInfo.AuthInfo.(credentials.TLSInfo); ok {
			if tlsInfo.SPIFFEID != nil {
				principal = tlsInfo.SPIFFEID.String()
			}
		}
	}

	// TODO(gtcooke94) check if we need to log before creating the event
	event := &audit.Event{
		FullMethodName: rpcData.fullMethod,
		Principal:      principal,
		PolicyName:     e.policyName,
		MatchedRule:    rule,
		Authorized:     authorized,
	}
	for _, logger := range e.auditLoggers {
		switch e.auditCondition {
		case v3rbacpb.RBAC_AuditLoggingOptions_ON_DENY:
			if !authorized {
				logger.Log(event)
			}
		case v3rbacpb.RBAC_AuditLoggingOptions_ON_ALLOW:
			if authorized {
				logger.Log(event)
			}
		case v3rbacpb.RBAC_AuditLoggingOptions_ON_DENY_AND_ALLOW:
			logger.Log(event)
		}
	}
}

// This is used when converting a custom config from raw JSON to a TypedStruct.
// The TypeURL of the TypeStruct will be "grpc.authz.audit_logging/<name>".
const typeURLPrefix = "grpc.authz.audit_logging/"
