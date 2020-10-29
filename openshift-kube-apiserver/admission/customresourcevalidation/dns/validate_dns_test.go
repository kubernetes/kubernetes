package dns

import (
	"testing"

	operatorv1 "github.com/openshift/api/operator/v1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// TestFailValidateDNSSpec verifies that validateDNSSpec rejects invalid specs.
func TestFailValidateDNSSpecCreate(t *testing.T) {
	errorCases := map[string]struct {
		spec       operatorv1.DNSSpec
		errorType  field.ErrorType
		errorField string
	}{
		"invalid toleration": {
			spec: operatorv1.DNSSpec{
				NodePlacement: operatorv1.DNSNodePlacement{
					Tolerations: []corev1.Toleration{{
						Key:      "x",
						Operator: corev1.TolerationOpExists,
						Effect:   "NoExcute",
					}},
				},
			},
			errorType:  field.ErrorTypeNotSupported,
			errorField: "spec.nodePlacement.tolerations[0].effect",
		},
		"invalid node selector": {
			spec: operatorv1.DNSSpec{
				NodePlacement: operatorv1.DNSNodePlacement{
					NodeSelector: map[string]string{
						"-": "foo",
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.nodePlacement.nodeSelector",
		},
		"SystemResolveConfType Upstream with TLS configured": {
			spec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.upstreams[0].type",
		},
		"Mixed Upstream types with TLS configured": {
			spec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
						{
							Type:    operatorv1.NetworkResolverType,
							Address: "1.1.1.1",
							Port:    7777,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.upstreams[0].type",
		},
		"Unknown Transport configured": {
			spec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "random",
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.transportConfig.transport",
		},
		"ForwardPlugin configured with TLS and without ServerName": {
			spec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeRequired,
			errorField: "spec.servers[0].forwardPlugin.transportConfig.tls.serverName",
		},
		"ForwardPlugin configured with Cleartext and TLS configuration": {
			spec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.CleartextTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.example.com",
								},
							},
						}},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.servers[0].forwardPlugin.transportConfig.tls",
		},
	}

	for tcName, tc := range errorCases {
		errs := validateDNSSpecCreate(tc.spec)
		if len(errs) == 0 {
			t.Errorf("%q: should have failed but did not", tcName)
		}

		for _, e := range errs {
			if e.Type != tc.errorType {
				t.Errorf("%q: expected errors of type '%s', got %v:", tcName, tc.errorType, e)
			}

			if e.Field != tc.errorField {
				t.Errorf("%q: expected errors in field '%s', got %v:", tcName, tc.errorField, e)
			}
		}
	}
}

func TestFailValidateDNSSpecUpdate(t *testing.T) {
	errorCases := map[string]struct {
		oldSpec    operatorv1.DNSSpec
		newSpec    operatorv1.DNSSpec
		errorType  field.ErrorType
		errorField string
	}{
		"UpstreamResolvers configured with unknown transport and updated to invalid cleartext config": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "oldtransport",
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type: "foo",
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.upstreams[0].type",
		},
		"SystemResolveConfType Upstream with TLS configured": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type:    operatorv1.SystemResolveConfType,
							Address: "2.2.2.2",
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.upstreams[0].type",
		},
		"UpstreamResolvers configured with unknown transport and updated to invalid TLS configuration": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type:    operatorv1.NetworkResolverType,
							Address: "1.1.1.1",
							Port:    7777,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "unknown",
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type:    operatorv1.NetworkResolverType,
							Address: "1.1.1.1",
							Port:    7777,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
					},
				},
			},
			errorType:  field.ErrorTypeRequired,
			errorField: "spec.upstreamResolvers.transportConfig.tls.serverName",
		},
		"ForwardPlugin configured with unknown transport and updated to invalid TLS configuration": {
			oldSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: "unknown",
							},
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeRequired,
			errorField: "spec.servers[0].forwardPlugin.transportConfig.tls.serverName",
		},
		"UpstreamResolvers TransportConfig has not changed but Upstreams has changed": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.NetworkResolverType,
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.upstreams[0].type",
		},
		"Servers Transport changed from known (TLS) to unknown type": {
			oldSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "unknown-transport-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.example.com",
								},
							},
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "unknown-transport-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: "unknown",
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.example.com",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.servers[0].forwardPlugin.transportConfig.transport",
		},
		"UpstreamResolvers Transport changed from known (TLS) to unknown type": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "unknown",
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.transportConfig.transport",
		},
		"Uniform Upstream types to mixed Upstream types with TLS configured": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.NetworkResolverType,
						},
						{
							Type:    operatorv1.NetworkResolverType,
							Address: "1.1.1.1",
							Port:    7777,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
						{
							Type:    operatorv1.NetworkResolverType,
							Address: "1.1.1.1",
							Port:    7777,
						},
					},
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.upstreamResolvers.upstreams[0].type",
		},
		"UpstreamResolvers TLS configured without ServerName": {
			oldSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "",
						},
					},
				},
			},
			errorType:  field.ErrorTypeRequired,
			errorField: "spec.upstreamResolvers.transportConfig.tls.serverName",
		},
		"ForwardPlugin configured with TLS and without ServerName": {
			oldSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "has-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.tls-server.com",
								},
							},
						},
					},
					{
						Name: "no-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.tls-server.com",
								},
							},
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "has-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.tls-server.com",
								},
							},
						},
					},
					{
						Name: "no-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeRequired,
			errorField: "spec.servers[1].forwardPlugin.transportConfig.tls.serverName",
		},
		"ForwardPlugin configured with Cleartext and TLS configuration": {
			oldSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.example.com",
								},
							},
						},
					},
				},
			},
			newSpec: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.CleartextTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.example.com",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.servers[0].forwardPlugin.transportConfig.tls",
		},
	}

	for tcName, tc := range errorCases {
		errs := validateDNSSpecUpdate(tc.newSpec, tc.oldSpec)
		if len(errs) == 0 {
			t.Errorf("%q: should have failed but did not", tcName)
		}

		for _, e := range errs {
			if e.Type != tc.errorType {
				t.Errorf("%q: expected errors of type '%s', got %v:", tcName, tc.errorType, e)
			}

			if e.Field != tc.errorField {
				t.Errorf("%q: expected errors in field '%s', got %v:", tcName, tc.errorField, e)
			}
		}
	}
}

// TestSucceedValidateDNSSpec verifies that validateDNSSpec accepts valid specs.
func TestSucceedValidateDNSSpecCreate(t *testing.T) {
	successCases := map[string]operatorv1.DNSSpec{
		"empty": {},
		"toleration + node selector": {
			NodePlacement: operatorv1.DNSNodePlacement{
				NodeSelector: map[string]string{
					"node-role.kubernetes.io/master": "",
				},
				Tolerations: []corev1.Toleration{{
					Key:      "node-role.kubernetes.io/master",
					Operator: corev1.TolerationOpExists,
					Effect:   corev1.TaintEffectNoExecute,
				}},
			},
		},
		"NetworkResolverType Upstream with TLS configured": {
			UpstreamResolvers: operatorv1.UpstreamResolvers{
				Upstreams: []operatorv1.Upstream{
					{
						Type:    operatorv1.NetworkResolverType,
						Address: "1.1.1.1",
					},
				},
				TransportConfig: operatorv1.DNSTransportConfig{
					Transport: operatorv1.TLSTransport,
					TLS: &operatorv1.DNSOverTLSConfig{
						ServerName: "dns.example.com",
					},
				},
			},
		},
		"Mixed Upstream types without TLS configured": {
			UpstreamResolvers: operatorv1.UpstreamResolvers{
				Upstreams: []operatorv1.Upstream{
					{
						Type: operatorv1.SystemResolveConfType,
					},
					{
						Type:    operatorv1.NetworkResolverType,
						Address: "1.1.1.1",
						Port:    7777,
					},
				},
			},
		},
		"Mixed Upstream types with Cleartext configured": {
			UpstreamResolvers: operatorv1.UpstreamResolvers{
				TransportConfig: operatorv1.DNSTransportConfig{
					Transport: operatorv1.CleartextTransport,
				},
				Upstreams: []operatorv1.Upstream{
					{
						Type: operatorv1.SystemResolveConfType,
					},
					{
						Type:    operatorv1.NetworkResolverType,
						Address: "1.1.1.1",
						Port:    7777,
					},
				},
			},
		},
		"Mixed Upstream types with nil TransportConfig configured": {
			UpstreamResolvers: operatorv1.UpstreamResolvers{
				TransportConfig: operatorv1.DNSTransportConfig{},
				Upstreams: []operatorv1.Upstream{
					{
						Type: operatorv1.SystemResolveConfType,
					},
					{
						Type:    operatorv1.NetworkResolverType,
						Address: "1.1.1.1",
						Port:    7777,
					},
				},
			},
		},
		"Mixed Upstream types with empty Transport configured": {
			UpstreamResolvers: operatorv1.UpstreamResolvers{
				TransportConfig: operatorv1.DNSTransportConfig{
					Transport: "",
				},
				Upstreams: []operatorv1.Upstream{
					{
						Type: operatorv1.SystemResolveConfType,
					},
					{
						Type:    operatorv1.NetworkResolverType,
						Address: "1.1.1.1",
						Port:    7777,
					},
				},
			},
		},
	}

	for tcName, s := range successCases {
		errs := validateDNSSpecCreate(s)
		if len(errs) != 0 {
			t.Errorf("%q: expected success, but failed: %v", tcName, errs.ToAggregate().Error())
		}
	}
}

func TestSucceedValidateDNSSpecUpdate(t *testing.T) {
	testCases := []struct {
		description string
		new         operatorv1.DNSSpec
		old         operatorv1.DNSSpec
	}{
		{
			description: "UpstreamResolvers TransportConfig has not changed but Upstreams have changed",
			old: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.CleartextTransport,
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type:    operatorv1.SystemResolveConfType,
							Address: "1.1.1.1",
						},
					},
				},
			},
			new: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.CleartextTransport,
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type:    operatorv1.NetworkResolverType,
							Address: "1.1.1.1",
						},
					},
				},
			},
		},
		{
			description: "UpstreamResolvers unknown old transport matches unknown new transport",
			old: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "oldtransport",
					},
				},
			},
			new: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "oldtransport",
					},
				},
			},
		},
		{
			description: "UpstreamResolvers unknown old transport matches unknown new transport with Upstream changes",
			old: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "oldtransport",
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type: operatorv1.SystemResolveConfType,
						},
					},
				},
			},
			new: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: "oldtransport",
					},
					Upstreams: []operatorv1.Upstream{
						{
							Type: "random",
						},
					},
				},
			},
		},
		{
			description: "UpstreamResolvers TransportConfig has changed",
			old: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.CleartextTransport,
					},
				},
			},
			new: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.TLSTransport,
						TLS: &operatorv1.DNSOverTLSConfig{
							ServerName: "dns.example.com",
						},
					},
				},
			},
		},
		{
			description: "UpstreamResolvers known transport to empty",
			old: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{
						Transport: operatorv1.CleartextTransport,
					},
				},
			},
			new: operatorv1.DNSSpec{
				UpstreamResolvers: operatorv1.UpstreamResolvers{
					TransportConfig: operatorv1.DNSTransportConfig{},
				},
			},
		},
		{
			description: "Servers TransportConfig has not changed",
			old: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.CleartextTransport,
							},
						},
					},
				},
			},
			new: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.CleartextTransport,
							},
						},
					},
				},
			},
		},
		{
			description: "Compare configuration by server name",
			old: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "cleartext-transport-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.CleartextTransport,
							},
						},
					},
					{
						Name: "unknown-transport-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: "unknown",
							},
						},
					},
				},
			},
			new: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "unknown-transport-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: "unknown",
							},
						},
					},
					{
						Name: "cleartext-transport-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.2"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.CleartextTransport,
							},
						},
					},
				},
			},
		},
		{
			description: "Servers TransportConfig has changed",
			old: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams:       []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{},
						},
					},
				},
			},
			new: operatorv1.DNSSpec{
				Servers: []operatorv1.Server{
					{
						Name: "tls-server",
						ForwardPlugin: operatorv1.ForwardPlugin{
							Upstreams: []string{"1.1.1.1"},
							TransportConfig: operatorv1.DNSTransportConfig{
								Transport: operatorv1.TLSTransport,
								TLS: &operatorv1.DNSOverTLSConfig{
									ServerName: "dns.example.com",
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		errs := validateDNSSpecUpdate(tc.new, tc.old)
		if len(errs) != 0 {
			t.Errorf("%q: expected success, but failed: %v", tc.description, errs.ToAggregate().Error())
		}
	}
}
