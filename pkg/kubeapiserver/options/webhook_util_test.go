package options

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"testing"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/util/webhook"
)

type fakeServiceResolver struct {
	host string
	port string
}

// ResolveEndpoint ...
func (i *fakeServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	if len(name) == 0 || len(namespace) == 0 || port == 0 {
		return nil, errors.New("cannot resolve an empty service name or namespace or port")
	}
	return &url.URL{Scheme: "https", Host: fmt.Sprintf("%s:%s", i.host, i.port)}, nil
}

func newFakeServiceResolver(host, port string) *fakeServiceResolver {
	return &fakeServiceResolver{
		host: host,
		port: port,
	}
}

func Test_newWebhookDialer(t *testing.T) {
	var tt = []struct {
		name       string
		dialerAddr string

		withBaseDialer bool
		withResolver   bool
		expectedError  bool
	}{
		{
			name:           "resolving a kubernetes service addr",
			dialerAddr:     "servicename.servicenamespace.svc:777",
			withResolver:   true,
			withBaseDialer: true,
		},
		{
			name:           "dialer's addr is not a kubernetes service reference",
			dialerAddr:     "example.org:777",
			withResolver:   true,
			expectedError:  true,
			withBaseDialer: true,
		},
		{
			name:           "resolving a kubernetes service addr without a port",
			dialerAddr:     "servicename.servicenamespace.svc",
			withResolver:   true,
			withBaseDialer: true,
		},
		{
			name:           "resolving a kubernetes service addr",
			dialerAddr:     "servicename.servicenamespace.pod:777",
			withResolver:   true,
			expectedError:  true,
			withBaseDialer: true,
		},
		{
			name:           "resolving an addr length less than 3",
			dialerAddr:     "example.org",
			withResolver:   true,
			expectedError:  true,
			withBaseDialer: true,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			var defaultResvoler webhook.ServiceResolver
			if tc.withResolver {
				defaultResvoler = newFakeServiceResolver(fakeDefaultEndpointHost, fakeDefaultEndpointPort)
			}
			var defaultBaseDialer utilnet.DialFunc
			if tc.withBaseDialer {
				defaultBaseDialer = func(ctx context.Context, net, addr string) (net.Conn, error) {
					if !tc.expectedError && tc.withResolver {
						return &fakeConn{ip: fakeDefaultEndpointHost, port: fakeDefaultEndpointPort}, nil
					}
					return &fakeConn{ip: fakeDelegatedEndpointHost, port: fakeDelegatedEndpointPort}, nil
				}
			}

			dialFunc, err := newWebhookDialer(defaultResvoler, defaultBaseDialer)(context.TODO(), "tcp", tc.dialerAddr)
			if err != nil {
				t.Fatalf("error got: %v", err)
			}

			remoteAddr := dialFunc.RemoteAddr().String()
			if tc.withResolver && !tc.expectedError {
				if remoteAddr != fmt.Sprintf("%s:%s", fakeDefaultEndpointHost, fakeDefaultEndpointPort) {
					t.Errorf("remoteAddr %s is not expected, should be resolved by serviceResolver", remoteAddr)
				}
			}
			if tc.withBaseDialer && tc.expectedError {
				if remoteAddr != fmt.Sprintf("%s:%s", fakeDelegatedEndpointHost, fakeDelegatedEndpointPort) {
					t.Errorf("remoteAddr %s is not expected, should be resolved by customDialer", remoteAddr)
				}
			}
			if !tc.withResolver && tc.withBaseDialer {
				if remoteAddr != fmt.Sprintf("%s:%s", fakeDelegatedEndpointHost, fakeDelegatedEndpointPort) {
					t.Errorf("remoteAddr %s is not expected, should be resolved by customDialer", remoteAddr)
				}
			}
		})
	}
}

type fakeConn struct {
	ip, port string
}

func (f *fakeConn) Read([]byte) (int, error)  { return 0, nil }
func (f *fakeConn) Write([]byte) (int, error) { return 0, nil }
func (f *fakeConn) Close() error              { return nil }
func (fakeConn) LocalAddr() net.Addr          { return nil }
func (f *fakeConn) RemoteAddr() net.Addr {
	return &fakeAddr{
		ip:   f.ip,
		port: f.port,
	}
}

func (fakeConn) SetDeadline(t time.Time) error      { return nil }
func (fakeConn) SetReadDeadline(t time.Time) error  { return nil }
func (fakeConn) SetWriteDeadline(t time.Time) error { return nil }

type fakeAddr struct {
	ip, port string
}

func (f fakeAddr) Network() string { return "" }

func (f fakeAddr) String() string {
	return fmt.Sprintf("%s:%s", f.ip, f.port)
}

var (
	fakeDefaultEndpointHost   = "1.1.1.1"
	fakeDefaultEndpointPort   = "443"
	fakeDelegatedEndpointHost = "2.2.2.2"
	fakeDelegatedEndpointPort = "8443"
)
