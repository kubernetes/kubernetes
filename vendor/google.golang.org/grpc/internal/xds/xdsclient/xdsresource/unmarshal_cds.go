/*
 *
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

package xdsresource

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	v3clusterpb "github.com/envoyproxy/go-control-plane/envoy/config/cluster/v3"
	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3aggregateclusterpb "github.com/envoyproxy/go-control-plane/envoy/extensions/clusters/aggregate/v3"
	v3tlspb "github.com/envoyproxy/go-control-plane/envoy/extensions/transport_sockets/tls/v3"

	"google.golang.org/grpc/internal/envconfig"
	"google.golang.org/grpc/internal/pretty"
	iserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/internal/xds/matcher"
	"google.golang.org/grpc/internal/xds/xdsclient/xdslbregistry"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource/version"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/structpb"
)

// ValidateClusterAndConstructClusterUpdateForTesting exports the
// validateClusterAndConstructClusterUpdate function for testing purposes.
var ValidateClusterAndConstructClusterUpdateForTesting = validateClusterAndConstructClusterUpdate

// TransportSocket proto message has a `name` field which is expected to be set
// to this value by the management server.
const transportSocketName = "envoy.transport_sockets.tls"

func unmarshalClusterResource(r *anypb.Any, serverCfg *bootstrap.ServerConfig) (string, ClusterUpdate, error) {
	r, err := UnwrapResource(r)
	if err != nil {
		return "", ClusterUpdate{}, fmt.Errorf("failed to unwrap resource: %v", err)
	}

	if !IsClusterResource(r.GetTypeUrl()) {
		return "", ClusterUpdate{}, fmt.Errorf("unexpected resource type: %q ", r.GetTypeUrl())
	}

	cluster := &v3clusterpb.Cluster{}
	if err := proto.Unmarshal(r.GetValue(), cluster); err != nil {
		return "", ClusterUpdate{}, fmt.Errorf("failed to unmarshal resource: %v", err)
	}
	cu, err := validateClusterAndConstructClusterUpdate(cluster, serverCfg)
	if err != nil {
		return cluster.GetName(), ClusterUpdate{}, err
	}
	cu.Raw = r

	return cluster.GetName(), cu, nil
}

const (
	defaultRingHashMinSize = 1024
	defaultRingHashMaxSize = 8 * 1024 * 1024 // 8M
	ringHashSizeUpperBound = 8 * 1024 * 1024 // 8M

	defaultLeastRequestChoiceCount = 2
)

func validateClusterAndConstructClusterUpdate(cluster *v3clusterpb.Cluster, serverCfg *bootstrap.ServerConfig) (ClusterUpdate, error) {
	telemetryLabels := make(map[string]string)
	if fmd := cluster.GetMetadata().GetFilterMetadata(); fmd != nil {
		if val, ok := fmd["com.google.csm.telemetry_labels"]; ok {
			if fields := val.GetFields(); fields != nil {
				if val, ok := fields["service_name"]; ok {
					if _, ok := val.GetKind().(*structpb.Value_StringValue); ok {
						telemetryLabels["csm.service_name"] = val.GetStringValue()
					}
				}
				if val, ok := fields["service_namespace"]; ok {
					if _, ok := val.GetKind().(*structpb.Value_StringValue); ok {
						telemetryLabels["csm.service_namespace_name"] = val.GetStringValue()
					}
				}
			}
		}
	}
	// "The values for the service labels csm.service_name and
	// csm.service_namespace_name come from xDS, “unknown” if not present." -
	// CSM Design.
	if _, ok := telemetryLabels["csm.service_name"]; !ok {
		telemetryLabels["csm.service_name"] = "unknown"
	}
	if _, ok := telemetryLabels["csm.service_namespace_name"]; !ok {
		telemetryLabels["csm.service_namespace_name"] = "unknown"
	}

	var lbPolicy json.RawMessage
	var err error
	switch cluster.GetLbPolicy() {
	case v3clusterpb.Cluster_ROUND_ROBIN:
		lbPolicy = []byte(`[{"xds_wrr_locality_experimental": {"childPolicy": [{"round_robin": {}}]}}]`)
	case v3clusterpb.Cluster_RING_HASH:
		rhc := cluster.GetRingHashLbConfig()
		if rhc.GetHashFunction() != v3clusterpb.Cluster_RingHashLbConfig_XX_HASH {
			return ClusterUpdate{}, fmt.Errorf("unsupported ring_hash hash function %v in response: %+v", rhc.GetHashFunction(), cluster)
		}
		// Minimum defaults to 1024 entries, and limited to 8M entries Maximum
		// defaults to 8M entries, and limited to 8M entries
		var minSize, maxSize uint64 = defaultRingHashMinSize, defaultRingHashMaxSize
		if min := rhc.GetMinimumRingSize(); min != nil {
			minSize = min.GetValue()
		}
		if max := rhc.GetMaximumRingSize(); max != nil {
			maxSize = max.GetValue()
		}

		rhLBCfg := []byte(fmt.Sprintf("{\"minRingSize\": %d, \"maxRingSize\": %d}", minSize, maxSize))
		lbPolicy = []byte(fmt.Sprintf(`[{"ring_hash_experimental": %s}]`, rhLBCfg))
	case v3clusterpb.Cluster_LEAST_REQUEST:
		// "The configuration for the Least Request LB policy is the
		// least_request_lb_config field. The field is optional; if not present,
		// defaults will be assumed for all of its values." - A48
		lr := cluster.GetLeastRequestLbConfig()
		var choiceCount uint32 = defaultLeastRequestChoiceCount
		if cc := lr.GetChoiceCount(); cc != nil {
			choiceCount = cc.GetValue()
		}
		// "If choice_count < 2, the config will be rejected." - A48
		if choiceCount < 2 {
			return ClusterUpdate{}, fmt.Errorf("Cluster_LeastRequestLbConfig.ChoiceCount must be >= 2, got: %v", choiceCount)
		}

		lrLBCfg := []byte(fmt.Sprintf("{\"choiceCount\": %d}", choiceCount))
		lbPolicy = []byte(fmt.Sprintf(`[{"least_request_experimental": %s}]`, lrLBCfg))
	default:
		return ClusterUpdate{}, fmt.Errorf("unexpected lbPolicy %v in response: %+v", cluster.GetLbPolicy(), cluster)
	}
	// Process security configuration received from the control plane iff the
	// corresponding environment variable is set.
	var sc *SecurityConfig
	if sc, err = securityConfigFromCluster(cluster); err != nil {
		return ClusterUpdate{}, err
	}

	// Process outlier detection received from the control plane iff the
	// corresponding environment variable is set.
	var od json.RawMessage
	if od, err = outlierConfigFromCluster(cluster); err != nil {
		return ClusterUpdate{}, err
	}

	if cluster.GetLoadBalancingPolicy() != nil {
		lbPolicy, err = xdslbregistry.ConvertToServiceConfig(cluster.GetLoadBalancingPolicy(), 0)
		if err != nil {
			return ClusterUpdate{}, fmt.Errorf("error converting LoadBalancingPolicy %v in response: %+v: %v", cluster.GetLoadBalancingPolicy(), cluster, err)
		}
		// "It will be the responsibility of the XdsClient to validate the
		// converted configuration. It will do this by having the gRPC LB policy
		// registry parse the configuration." - A52
		bc := &iserviceconfig.BalancerConfig{}
		if err := json.Unmarshal(lbPolicy, bc); err != nil {
			return ClusterUpdate{}, fmt.Errorf("JSON generated from xDS LB policy registry: %s is invalid: %v", pretty.FormatJSON(lbPolicy), err)
		}
	}

	ret := ClusterUpdate{
		ClusterName:      cluster.GetName(),
		SecurityCfg:      sc,
		MaxRequests:      circuitBreakersFromCluster(cluster),
		LBPolicy:         lbPolicy,
		OutlierDetection: od,
		TelemetryLabels:  telemetryLabels,
	}

	if lrs := cluster.GetLrsServer(); lrs != nil {
		if lrs.GetSelf() == nil {
			return ClusterUpdate{}, fmt.Errorf("unsupported config_source_specifier %T in lrs_server field", lrs.ConfigSourceSpecifier)
		}
		ret.LRSServerConfig = serverCfg
	}

	// Validate and set cluster type from the response.
	switch {
	case cluster.GetType() == v3clusterpb.Cluster_EDS:
		if configsource := cluster.GetEdsClusterConfig().GetEdsConfig(); configsource.GetAds() == nil && configsource.GetSelf() == nil {
			return ClusterUpdate{}, fmt.Errorf("CDS's EDS config source is not ADS or Self: %+v", cluster)
		}
		ret.ClusterType = ClusterTypeEDS
		ret.EDSServiceName = cluster.GetEdsClusterConfig().GetServiceName()
		if strings.HasPrefix(ret.ClusterName, "xdstp:") && ret.EDSServiceName == "" {
			return ClusterUpdate{}, fmt.Errorf("CDS's EDS service name is not set with a new-style cluster name: %+v", cluster)
		}
		return ret, nil
	case cluster.GetType() == v3clusterpb.Cluster_LOGICAL_DNS:
		ret.ClusterType = ClusterTypeLogicalDNS
		dnsHN, err := dnsHostNameFromCluster(cluster)
		if err != nil {
			return ClusterUpdate{}, err
		}
		ret.DNSHostName = dnsHN
		return ret, nil
	case cluster.GetClusterType() != nil && cluster.GetClusterType().Name == "envoy.clusters.aggregate":
		clusters := &v3aggregateclusterpb.ClusterConfig{}
		if err := proto.Unmarshal(cluster.GetClusterType().GetTypedConfig().GetValue(), clusters); err != nil {
			return ClusterUpdate{}, fmt.Errorf("failed to unmarshal resource: %v", err)
		}
		if len(clusters.Clusters) == 0 {
			return ClusterUpdate{}, fmt.Errorf("xds: aggregate cluster has empty clusters field in response: %+v", cluster)
		}
		ret.ClusterType = ClusterTypeAggregate
		ret.PrioritizedClusterNames = clusters.Clusters
		return ret, nil
	default:
		return ClusterUpdate{}, fmt.Errorf("unsupported cluster type (%v, %v) in response: %+v", cluster.GetType(), cluster.GetClusterType(), cluster)
	}
}

// dnsHostNameFromCluster extracts the DNS host name from the cluster's load
// assignment.
//
// There should be exactly one locality, with one endpoint, whose address
// contains the address and port.
func dnsHostNameFromCluster(cluster *v3clusterpb.Cluster) (string, error) {
	loadAssignment := cluster.GetLoadAssignment()
	if loadAssignment == nil {
		return "", fmt.Errorf("load_assignment not present for LOGICAL_DNS cluster")
	}
	if len(loadAssignment.GetEndpoints()) != 1 {
		return "", fmt.Errorf("load_assignment for LOGICAL_DNS cluster must have exactly one locality, got: %+v", loadAssignment)
	}
	endpoints := loadAssignment.GetEndpoints()[0].GetLbEndpoints()
	if len(endpoints) != 1 {
		return "", fmt.Errorf("locality for LOGICAL_DNS cluster must have exactly one endpoint, got: %+v", endpoints)
	}
	endpoint := endpoints[0].GetEndpoint()
	if endpoint == nil {
		return "", fmt.Errorf("endpoint for LOGICAL_DNS cluster not set")
	}
	socketAddr := endpoint.GetAddress().GetSocketAddress()
	if socketAddr == nil {
		return "", fmt.Errorf("socket address for endpoint for LOGICAL_DNS cluster not set")
	}
	if socketAddr.GetResolverName() != "" {
		return "", fmt.Errorf("socket address for endpoint for LOGICAL_DNS cluster not set has unexpected custom resolver name: %v", socketAddr.GetResolverName())
	}
	host := socketAddr.GetAddress()
	if host == "" {
		return "", fmt.Errorf("host for endpoint for LOGICAL_DNS cluster not set")
	}
	port := socketAddr.GetPortValue()
	if port == 0 {
		return "", fmt.Errorf("port for endpoint for LOGICAL_DNS cluster not set")
	}
	return net.JoinHostPort(host, strconv.Itoa(int(port))), nil
}

// securityConfigFromCluster extracts the relevant security configuration from
// the received Cluster resource.
func securityConfigFromCluster(cluster *v3clusterpb.Cluster) (*SecurityConfig, error) {
	if tsm := cluster.GetTransportSocketMatches(); len(tsm) != 0 {
		return nil, fmt.Errorf("unsupported transport_socket_matches field is non-empty: %+v", tsm)
	}
	// The Cluster resource contains a `transport_socket` field, which contains
	// a oneof `typed_config` field of type `protobuf.Any`. The any proto
	// contains a marshaled representation of an `UpstreamTlsContext` message.
	ts := cluster.GetTransportSocket()
	if ts == nil {
		return nil, nil
	}
	if name := ts.GetName(); name != transportSocketName {
		return nil, fmt.Errorf("transport_socket field has unexpected name: %s", name)
	}
	tc := ts.GetTypedConfig()
	if typeURL := tc.GetTypeUrl(); typeURL != version.V3UpstreamTLSContextURL {
		return nil, fmt.Errorf("transport_socket missing typed_config or wrong type_url: %q", typeURL)
	}
	upstreamCtx := &v3tlspb.UpstreamTlsContext{}
	if err := proto.Unmarshal(tc.GetValue(), upstreamCtx); err != nil {
		return nil, fmt.Errorf("failed to unmarshal UpstreamTlsContext in CDS response: %v", err)
	}
	// The following fields from `UpstreamTlsContext` are ignored:
	// - sni
	// - allow_renegotiation
	// - max_session_keys
	if upstreamCtx.GetCommonTlsContext() == nil {
		return nil, errors.New("UpstreamTlsContext in CDS response does not contain a CommonTlsContext")
	}

	return securityConfigFromCommonTLSContext(upstreamCtx.GetCommonTlsContext(), false)
}

// common is expected to be not nil.
// The `alpn_protocols` field is ignored.
func securityConfigFromCommonTLSContext(common *v3tlspb.CommonTlsContext, server bool) (*SecurityConfig, error) {
	if common.GetTlsParams() != nil {
		return nil, fmt.Errorf("unsupported tls_params field in CommonTlsContext message: %+v", common)
	}
	if common.GetCustomHandshaker() != nil {
		return nil, fmt.Errorf("unsupported custom_handshaker field in CommonTlsContext message: %+v", common)
	}

	// For now, if we can't get a valid security config from the new fields, we
	// fallback to the old deprecated fields.
	// TODO: Drop support for deprecated fields. NACK if err != nil here.
	sc, err1 := securityConfigFromCommonTLSContextUsingNewFields(common, server)
	if sc == nil || sc.Equal(&SecurityConfig{}) {
		var err error
		sc, err = securityConfigFromCommonTLSContextWithDeprecatedFields(common, server)
		if err != nil {
			// Retain the validation error from using the new fields.
			return nil, errors.Join(err1, fmt.Errorf("failed to parse config using deprecated fields: %v", err))
		}
	}
	if sc != nil {
		// sc == nil is a valid case where the control plane has not sent us any
		// security configuration. xDS creds will use fallback creds.
		if server {
			if sc.IdentityInstanceName == "" {
				return nil, errors.New("security configuration on the server-side does not contain identity certificate provider instance name")
			}
		} else {
			if !sc.UseSystemRootCerts && sc.RootInstanceName == "" {
				return nil, errors.New("security configuration on the client-side does not contain root certificate provider instance name")
			}
		}
	}
	return sc, nil
}

func securityConfigFromCommonTLSContextWithDeprecatedFields(common *v3tlspb.CommonTlsContext, server bool) (*SecurityConfig, error) {
	// The `CommonTlsContext` contains a
	// `tls_certificate_certificate_provider_instance` field of type
	// `CertificateProviderInstance`, which contains the provider instance name
	// and the certificate name to fetch identity certs.
	sc := &SecurityConfig{}
	if identity := common.GetTlsCertificateCertificateProviderInstance(); identity != nil {
		sc.IdentityInstanceName = identity.GetInstanceName()
		sc.IdentityCertName = identity.GetCertificateName()
	}

	// The `CommonTlsContext` contains a `validation_context_type` field which
	// is a oneof. We can get the values that we are interested in from two of
	// those possible values:
	//  - combined validation context:
	//    - contains a default validation context which holds the list of
	//      matchers for accepted SANs.
	//    - contains certificate provider instance configuration
	//  - certificate provider instance configuration
	//    - in this case, we do not get a list of accepted SANs.
	switch t := common.GetValidationContextType().(type) {
	case *v3tlspb.CommonTlsContext_CombinedValidationContext:
		combined := common.GetCombinedValidationContext()
		var matchers []matcher.StringMatcher
		if def := combined.GetDefaultValidationContext(); def != nil {
			for _, m := range def.GetMatchSubjectAltNames() {
				matcher, err := matcher.StringMatcherFromProto(m)
				if err != nil {
					return nil, err
				}
				matchers = append(matchers, matcher)
			}
		}
		if server && len(matchers) != 0 {
			return nil, fmt.Errorf("match_subject_alt_names field in validation context is not supported on the server: %v", common)
		}
		sc.SubjectAltNameMatchers = matchers
		if pi := combined.GetValidationContextCertificateProviderInstance(); pi != nil {
			sc.RootInstanceName = pi.GetInstanceName()
			sc.RootCertName = pi.GetCertificateName()
		}
	case *v3tlspb.CommonTlsContext_ValidationContextCertificateProviderInstance:
		pi := common.GetValidationContextCertificateProviderInstance()
		sc.RootInstanceName = pi.GetInstanceName()
		sc.RootCertName = pi.GetCertificateName()
	case nil:
		// It is valid for the validation context to be nil on the server side.
	default:
		return nil, fmt.Errorf("validation context contains unexpected type: %T", t)
	}
	return sc, nil
}

// gRFC A29 https://github.com/grpc/proposal/blob/master/A29-xds-tls-security.md
// specifies the new way to fetch security configuration and says the following:
//
// Although there are various ways to obtain certificates as per this proto
// (which are supported by Envoy), gRPC supports only one of them and that is
// the `CertificateProviderPluginInstance` proto.
//
// This helper function attempts to fetch security configuration from the
// `CertificateProviderPluginInstance` message, given a CommonTlsContext.
func securityConfigFromCommonTLSContextUsingNewFields(common *v3tlspb.CommonTlsContext, server bool) (*SecurityConfig, error) {
	// The `tls_certificate_provider_instance` field of type
	// `CertificateProviderPluginInstance` is used to fetch the identity
	// certificate provider.
	sc := &SecurityConfig{}
	identity := common.GetTlsCertificateProviderInstance()
	if identity == nil && len(common.GetTlsCertificates()) != 0 {
		return nil, fmt.Errorf("expected field tls_certificate_provider_instance is not set, while unsupported field tls_certificates is set in CommonTlsContext message: %+v", common)
	}
	if identity == nil && common.GetTlsCertificateSdsSecretConfigs() != nil {
		return nil, fmt.Errorf("expected field tls_certificate_provider_instance is not set, while unsupported field tls_certificate_sds_secret_configs is set in CommonTlsContext message: %+v", common)
	}
	sc.IdentityInstanceName = identity.GetInstanceName()
	sc.IdentityCertName = identity.GetCertificateName()

	// The `CommonTlsContext` contains a oneof field `validation_context_type`,
	// which contains the `CertificateValidationContext` message in one of the
	// following ways:
	//  - `validation_context` field
	//    - this is directly of type `CertificateValidationContext`
	//  - `combined_validation_context` field
	//    - this is of type `CombinedCertificateValidationContext` and contains
	//      a `default validation context` field of type
	//      `CertificateValidationContext`
	//
	// The `CertificateValidationContext` message has the following fields that
	// we are interested in:
	//  - `ca_certificate_provider_instance`
	//    - this is of type `CertificateProviderPluginInstance`
	//  - `system_root_certs`:
	//    - This indicates the usage of system root certs for validation.
	//  - `match_subject_alt_names`
	//    - this is a list of string matchers
	//
	// The `CertificateProviderPluginInstance` message contains two fields
	//  - instance_name
	//    - this is the certificate provider instance name to be looked up in
	//      the bootstrap configuration
	//  - certificate_name
	//    -  this is an opaque name passed to the certificate provider
	var validationCtx *v3tlspb.CertificateValidationContext
	switch typ := common.GetValidationContextType().(type) {
	case *v3tlspb.CommonTlsContext_ValidationContext:
		validationCtx = common.GetValidationContext()
	case *v3tlspb.CommonTlsContext_CombinedValidationContext:
		validationCtx = common.GetCombinedValidationContext().GetDefaultValidationContext()
	case nil:
		// It is valid for the validation context to be nil on the server side.
		return sc, nil
	default:
		return nil, fmt.Errorf("validation context contains unexpected type: %T", typ)
	}
	// If we get here, it means that the `CertificateValidationContext` message
	// was found through one of the supported ways. It is an error if the
	// validation context is specified, but it does not specify a way to
	// validate TLS certificates. Peer TLS certs can be verified in the
	// following ways:
	// 1. If the ca_certificate_provider_instance field is set, it contains
	//    information about the certificate provider to be used for the root
	//    certificates, else
	// 2. If the system_root_certs field is set, and the config is for a client,
	//    use the system default root certs.
	useSystemRootCerts := false
	if validationCtx.GetCaCertificateProviderInstance() == nil && envconfig.XDSSystemRootCertsEnabled {
		if server {
			if validationCtx.GetSystemRootCerts() != nil {
				// The `system_root_certs` field will not be supported on the
				// gRPC server side. If `ca_certificate_provider_instance` is
				// unset and `system_root_certs` is set, the LDS resource will
				// be NACKed.
				// - A82
				return nil, fmt.Errorf("expected field ca_certificate_provider_instance is missing and unexpected field system_root_certs is set for server in CommonTlsContext message: %+v", common)
			}
		} else {
			if validationCtx.GetSystemRootCerts() != nil {
				useSystemRootCerts = true
			}
		}
	}

	// The following fields are ignored:
	// - trusted_ca
	// - watched_directory
	// - allow_expired_certificate
	// - trust_chain_verification
	switch {
	case len(validationCtx.GetVerifyCertificateSpki()) != 0:
		return nil, fmt.Errorf("unsupported verify_certificate_spki field in CommonTlsContext message: %+v", common)
	case len(validationCtx.GetVerifyCertificateHash()) != 0:
		return nil, fmt.Errorf("unsupported verify_certificate_hash field in CommonTlsContext message: %+v", common)
	case validationCtx.GetRequireSignedCertificateTimestamp().GetValue():
		return nil, fmt.Errorf("unsupported require_signed_certificate_timestamp field in CommonTlsContext message: %+v", common)
	case validationCtx.GetCrl() != nil:
		return nil, fmt.Errorf("unsupported crl field in CommonTlsContext message: %+v", common)
	case validationCtx.GetCustomValidatorConfig() != nil:
		return nil, fmt.Errorf("unsupported custom_validator_config field in CommonTlsContext message: %+v", common)
	}

	if rootProvider := validationCtx.GetCaCertificateProviderInstance(); rootProvider != nil {
		sc.RootInstanceName = rootProvider.GetInstanceName()
		sc.RootCertName = rootProvider.GetCertificateName()
	} else if useSystemRootCerts {
		sc.UseSystemRootCerts = true
	} else if !server && envconfig.XDSSystemRootCertsEnabled {
		return nil, fmt.Errorf("expected fields ca_certificate_provider_instance and system_root_certs are missing in CommonTlsContext message: %+v", common)
	} else {
		// Don't mention the system_root_certs field if it was not checked.
		return nil, fmt.Errorf("expected field ca_certificate_provider_instance is missing in CommonTlsContext message: %+v", common)
	}

	var matchers []matcher.StringMatcher
	for _, m := range validationCtx.GetMatchSubjectAltNames() {
		matcher, err := matcher.StringMatcherFromProto(m)
		if err != nil {
			return nil, err
		}
		matchers = append(matchers, matcher)
	}
	if server && len(matchers) != 0 {
		return nil, fmt.Errorf("match_subject_alt_names field in validation context is not supported on the server: %v", common)
	}
	sc.SubjectAltNameMatchers = matchers
	return sc, nil
}

// circuitBreakersFromCluster extracts the circuit breakers configuration from
// the received cluster resource. Returns nil if no CircuitBreakers or no
// Thresholds in CircuitBreakers.
func circuitBreakersFromCluster(cluster *v3clusterpb.Cluster) *uint32 {
	for _, threshold := range cluster.GetCircuitBreakers().GetThresholds() {
		if threshold.GetPriority() != v3corepb.RoutingPriority_DEFAULT {
			continue
		}
		maxRequestsPb := threshold.GetMaxRequests()
		if maxRequestsPb == nil {
			return nil
		}
		maxRequests := maxRequestsPb.GetValue()
		return &maxRequests
	}
	return nil
}

// idurationp takes a time.Duration and converts it to an internal duration, and
// returns a pointer to that internal duration.
func idurationp(d time.Duration) *iserviceconfig.Duration {
	id := iserviceconfig.Duration(d)
	return &id
}

func uint32p(i uint32) *uint32 {
	return &i
}

// Helper types to prepare Outlier Detection JSON. Pointer types to distinguish
// between unset and a zero value.
type successRateEjection struct {
	StdevFactor           *uint32 `json:"stdevFactor,omitempty"`
	EnforcementPercentage *uint32 `json:"enforcementPercentage,omitempty"`
	MinimumHosts          *uint32 `json:"minimumHosts,omitempty"`
	RequestVolume         *uint32 `json:"requestVolume,omitempty"`
}

type failurePercentageEjection struct {
	Threshold             *uint32 `json:"threshold,omitempty"`
	EnforcementPercentage *uint32 `json:"enforcementPercentage,omitempty"`
	MinimumHosts          *uint32 `json:"minimumHosts,omitempty"`
	RequestVolume         *uint32 `json:"requestVolume,omitempty"`
}

type odLBConfig struct {
	Interval                  *iserviceconfig.Duration   `json:"interval,omitempty"`
	BaseEjectionTime          *iserviceconfig.Duration   `json:"baseEjectionTime,omitempty"`
	MaxEjectionTime           *iserviceconfig.Duration   `json:"maxEjectionTime,omitempty"`
	MaxEjectionPercent        *uint32                    `json:"maxEjectionPercent,omitempty"`
	SuccessRateEjection       *successRateEjection       `json:"successRateEjection,omitempty"`
	FailurePercentageEjection *failurePercentageEjection `json:"failurePercentageEjection,omitempty"`
}

// outlierConfigFromCluster converts the received Outlier Detection
// configuration into JSON configuration for Outlier Detection, taking into
// account xDS Defaults. Returns nil if no OutlierDetection field set in the
// cluster resource.
func outlierConfigFromCluster(cluster *v3clusterpb.Cluster) (json.RawMessage, error) {
	od := cluster.GetOutlierDetection()
	if od == nil {
		return nil, nil
	}

	// "The outlier_detection field of the Cluster resource should have its fields
	//	validated according to the rules for the corresponding LB policy config
	//	fields in the above "Validation" section. If any of these requirements is
	//	violated, the Cluster resource should be NACKed." - A50
	// "The google.protobuf.Duration fields interval, base_ejection_time, and
	// max_ejection_time must obey the restrictions in the
	// google.protobuf.Duration documentation and they must have non-negative
	// values." - A50
	var interval *iserviceconfig.Duration
	if i := od.GetInterval(); i != nil {
		if err := i.CheckValid(); err != nil {
			return nil, fmt.Errorf("outlier_detection.interval is invalid with error: %v", err)
		}
		if interval = idurationp(i.AsDuration()); *interval < 0 {
			return nil, fmt.Errorf("outlier_detection.interval = %v; must be a valid duration and >= 0", *interval)
		}
	}

	var baseEjectionTime *iserviceconfig.Duration
	if bet := od.GetBaseEjectionTime(); bet != nil {
		if err := bet.CheckValid(); err != nil {
			return nil, fmt.Errorf("outlier_detection.base_ejection_time is invalid with error: %v", err)
		}
		if baseEjectionTime = idurationp(bet.AsDuration()); *baseEjectionTime < 0 {
			return nil, fmt.Errorf("outlier_detection.base_ejection_time = %v; must be >= 0", *baseEjectionTime)
		}
	}

	var maxEjectionTime *iserviceconfig.Duration
	if met := od.GetMaxEjectionTime(); met != nil {
		if err := met.CheckValid(); err != nil {
			return nil, fmt.Errorf("outlier_detection.max_ejection_time is invalid: %v", err)
		}
		if maxEjectionTime = idurationp(met.AsDuration()); *maxEjectionTime < 0 {
			return nil, fmt.Errorf("outlier_detection.max_ejection_time = %v; must be >= 0", *maxEjectionTime)
		}
	}

	// "The fields max_ejection_percent, enforcing_success_rate,
	// failure_percentage_threshold, and enforcing_failure_percentage must have
	// values less than or equal to 100. If any of these requirements is
	// violated, the Cluster resource should be NACKed." - A50
	var maxEjectionPercent *uint32
	if mep := od.GetMaxEjectionPercent(); mep != nil {
		if maxEjectionPercent = uint32p(mep.GetValue()); *maxEjectionPercent > 100 {
			return nil, fmt.Errorf("outlier_detection.max_ejection_percent = %v; must be <= 100", *maxEjectionPercent)
		}
	}
	// "if the enforcing_success_rate field is set to 0, the config
	// success_rate_ejection field will be null and all success_rate_* fields
	// will be ignored." - A50
	var enforcingSuccessRate *uint32
	if esr := od.GetEnforcingSuccessRate(); esr != nil {
		if enforcingSuccessRate = uint32p(esr.GetValue()); *enforcingSuccessRate > 100 {
			return nil, fmt.Errorf("outlier_detection.enforcing_success_rate = %v; must be <= 100", *enforcingSuccessRate)
		}
	}
	var failurePercentageThreshold *uint32
	if fpt := od.GetFailurePercentageThreshold(); fpt != nil {
		if failurePercentageThreshold = uint32p(fpt.GetValue()); *failurePercentageThreshold > 100 {
			return nil, fmt.Errorf("outlier_detection.failure_percentage_threshold = %v; must be <= 100", *failurePercentageThreshold)
		}
	}
	// "If the enforcing_failure_percent field is set to 0 or null, the config
	// failure_percent_ejection field will be null and all failure_percent_*
	// fields will be ignored." - A50
	var enforcingFailurePercentage *uint32
	if efp := od.GetEnforcingFailurePercentage(); efp != nil {
		if enforcingFailurePercentage = uint32p(efp.GetValue()); *enforcingFailurePercentage > 100 {
			return nil, fmt.Errorf("outlier_detection.enforcing_failure_percentage = %v; must be <= 100", *enforcingFailurePercentage)
		}
	}

	var successRateStdevFactor *uint32
	if srsf := od.GetSuccessRateStdevFactor(); srsf != nil {
		successRateStdevFactor = uint32p(srsf.GetValue())
	}
	var successRateMinimumHosts *uint32
	if srmh := od.GetSuccessRateMinimumHosts(); srmh != nil {
		successRateMinimumHosts = uint32p(srmh.GetValue())
	}
	var successRateRequestVolume *uint32
	if srrv := od.GetSuccessRateRequestVolume(); srrv != nil {
		successRateRequestVolume = uint32p(srrv.GetValue())
	}
	var failurePercentageMinimumHosts *uint32
	if fpmh := od.GetFailurePercentageMinimumHosts(); fpmh != nil {
		failurePercentageMinimumHosts = uint32p(fpmh.GetValue())
	}
	var failurePercentageRequestVolume *uint32
	if fprv := od.GetFailurePercentageRequestVolume(); fprv != nil {
		failurePercentageRequestVolume = uint32p(fprv.GetValue())
	}

	// "if the enforcing_success_rate field is set to 0, the config
	// success_rate_ejection field will be null and all success_rate_* fields
	// will be ignored." - A50
	var sre *successRateEjection
	if enforcingSuccessRate == nil || *enforcingSuccessRate != 0 {
		sre = &successRateEjection{
			StdevFactor:           successRateStdevFactor,
			EnforcementPercentage: enforcingSuccessRate,
			MinimumHosts:          successRateMinimumHosts,
			RequestVolume:         successRateRequestVolume,
		}
	}

	// "If the enforcing_failure_percent field is set to 0 or null, the config
	// failure_percent_ejection field will be null and all failure_percent_*
	// fields will be ignored." - A50
	var fpe *failurePercentageEjection
	if enforcingFailurePercentage != nil && *enforcingFailurePercentage != 0 {
		fpe = &failurePercentageEjection{
			Threshold:             failurePercentageThreshold,
			EnforcementPercentage: enforcingFailurePercentage,
			MinimumHosts:          failurePercentageMinimumHosts,
			RequestVolume:         failurePercentageRequestVolume,
		}
	}

	odLBCfg := &odLBConfig{
		Interval:                  interval,
		BaseEjectionTime:          baseEjectionTime,
		MaxEjectionTime:           maxEjectionTime,
		MaxEjectionPercent:        maxEjectionPercent,
		SuccessRateEjection:       sre,
		FailurePercentageEjection: fpe,
	}
	return json.Marshal(odLBCfg)
}
