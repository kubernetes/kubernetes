/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package apiserver

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"golang.org/x/net/websocket"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiregistration "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/utils/pointer"
)

type targetHTTPHandler struct {
	called  bool
	headers map[string][]string
	path    string
	host    string
}

func (d *targetHTTPHandler) Reset() {
	d.path = ""
	d.called = false
	d.headers = nil
	d.host = ""
}

func (d *targetHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	d.path = r.URL.Path
	d.called = true
	d.headers = r.Header
	d.host = r.Host
	w.WriteHeader(http.StatusOK)
}

func contextHandler(handler http.Handler, user user.Info) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		if user != nil {
			ctx = genericapirequest.WithUser(ctx, user)
		}
		resolver := &genericapirequest.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		}
		info, err := resolver.NewRequestInfo(req)
		if err == nil {
			ctx = genericapirequest.WithRequestInfo(ctx, info)
		}
		req = req.WithContext(ctx)
		handler.ServeHTTP(w, req)
	})
}

type mockedRouter struct {
	destinationHost string
	err             error
}

func (r *mockedRouter) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return &url.URL{Scheme: "https", Host: r.destinationHost}, r.err
}

func emptyCert() []byte {
	return []byte{}
}

func TestProxyHandler(t *testing.T) {
	target := &targetHTTPHandler{}
	targetServer := httptest.NewUnstartedServer(target)
	if cert, err := tls.X509KeyPair(svcCrt, svcKey); err != nil {
		t.Fatal(err)
	} else {
		targetServer.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	}
	targetServer.StartTLS()
	defer targetServer.Close()

	tests := map[string]struct {
		user       user.Info
		path       string
		apiService *apiregistration.APIService

		serviceResolver ServiceResolver

		expectedStatusCode int
		expectedBody       string
		expectedCalled     bool
		expectedHeaders    map[string][]string
	}{
		"no target": {
			expectedStatusCode: http.StatusNotFound,
		},
		"no user": {
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{Port: pointer.Int32Ptr(443)},
					Group:   "foo",
					Version: "v1",
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			expectedStatusCode: http.StatusInternalServerError,
			expectedBody:       "missing user",
		},
		"proxy with user, insecure": {
			user: &user.DefaultInfo{
				Name:   "username",
				Groups: []string{"one", "two"},
			},
			path: "/request/path",
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
				Spec: apiregistration.APIServiceSpec{
					Service:               &apiregistration.ServiceReference{Port: pointer.Int32Ptr(443)},
					Group:                 "foo",
					Version:               "v1",
					InsecureSkipTLSVerify: true,
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			expectedStatusCode: http.StatusOK,
			expectedCalled:     true,
			expectedHeaders: map[string][]string{
				"X-Forwarded-Proto": {"https"},
				"X-Forwarded-Uri":   {"/request/path"},
				"X-Forwarded-For":   {"127.0.0.1"},
				"X-Remote-User":     {"username"},
				"User-Agent":        {"Go-http-client/1.1"},
				"Accept-Encoding":   {"gzip"},
				"X-Remote-Group":    {"one", "two"},
			},
		},
		"proxy with user, cabundle": {
			user: &user.DefaultInfo{
				Name:   "username",
				Groups: []string{"one", "two"},
			},
			path: "/request/path",
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
				Spec: apiregistration.APIServiceSpec{
					Service:  &apiregistration.ServiceReference{Name: "test-service", Namespace: "test-ns", Port: pointer.Int32Ptr(443)},
					Group:    "foo",
					Version:  "v1",
					CABundle: testCACrt,
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			expectedStatusCode: http.StatusOK,
			expectedCalled:     true,
			expectedHeaders: map[string][]string{
				"X-Forwarded-Proto": {"https"},
				"X-Forwarded-Uri":   {"/request/path"},
				"X-Forwarded-For":   {"127.0.0.1"},
				"X-Remote-User":     {"username"},
				"User-Agent":        {"Go-http-client/1.1"},
				"Accept-Encoding":   {"gzip"},
				"X-Remote-Group":    {"one", "two"},
			},
		},
		"service unavailable": {
			user: &user.DefaultInfo{
				Name:   "username",
				Groups: []string{"one", "two"},
			},
			path: "/request/path",
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
				Spec: apiregistration.APIServiceSpec{
					Service:  &apiregistration.ServiceReference{Name: "test-service", Namespace: "test-ns", Port: pointer.Int32Ptr(443)},
					Group:    "foo",
					Version:  "v1",
					CABundle: testCACrt,
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionFalse},
					},
				},
			},
			expectedStatusCode: http.StatusServiceUnavailable,
		},
		"service unresolveable": {
			user: &user.DefaultInfo{
				Name:   "username",
				Groups: []string{"one", "two"},
			},
			path:            "/request/path",
			serviceResolver: &mockedRouter{err: fmt.Errorf("unresolveable")},
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
				Spec: apiregistration.APIServiceSpec{
					Service:  &apiregistration.ServiceReference{Name: "bad-service", Namespace: "test-ns", Port: pointer.Int32Ptr(443)},
					Group:    "foo",
					Version:  "v1",
					CABundle: testCACrt,
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			expectedStatusCode: http.StatusServiceUnavailable,
		},
		"fail on bad serving cert": {
			user: &user.DefaultInfo{
				Name:   "username",
				Groups: []string{"one", "two"},
			},
			path: "/request/path",
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{Port: pointer.Int32Ptr(443)},
					Group:   "foo",
					Version: "v1",
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			expectedStatusCode: http.StatusServiceUnavailable,
		},
	}

	for name, tc := range tests {
		target.Reset()

		func() {
			serviceResolver := tc.serviceResolver
			if serviceResolver == nil {
				serviceResolver = &mockedRouter{destinationHost: targetServer.Listener.Addr().String()}
			}
			handler := &proxyHandler{
				localDelegate:              http.NewServeMux(),
				serviceResolver:            serviceResolver,
				proxyTransport:             &http.Transport{},
				proxyCurrentCertKeyContent: func() ([]byte, []byte) { return emptyCert(), emptyCert() },
			}
			server := httptest.NewServer(contextHandler(handler, tc.user))
			defer server.Close()

			if tc.apiService != nil {
				handler.updateAPIService(tc.apiService)
				curr := handler.handlingInfo.Load().(proxyHandlingInfo)
				handler.handlingInfo.Store(curr)
			}

			resp, err := http.Get(server.URL + tc.path)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			if e, a := tc.expectedStatusCode, resp.StatusCode; e != a {
				body, _ := httputil.DumpResponse(resp, true)
				t.Logf("%s: %v", name, string(body))
				t.Errorf("%s: expected %v, got %v", name, e, a)
				return
			}
			bytes, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			if !strings.Contains(string(bytes), tc.expectedBody) {
				t.Errorf("%s: expected %q, got %q", name, tc.expectedBody, string(bytes))
				return
			}

			if e, a := tc.expectedCalled, target.called; e != a {
				t.Errorf("%s: expected %v, got %v", name, e, a)
				return
			}
			// this varies every test
			delete(target.headers, "X-Forwarded-Host")
			if e, a := tc.expectedHeaders, target.headers; !reflect.DeepEqual(e, a) {
				t.Errorf("%s: expected %v, got %v", name, e, a)
				return
			}
			if e, a := targetServer.Listener.Addr().String(), target.host; tc.expectedCalled && !reflect.DeepEqual(e, a) {
				t.Errorf("%s: expected %v, got %v", name, e, a)
				return
			}
		}()
	}
}

func TestProxyUpgrade(t *testing.T) {
	testcases := map[string]struct {
		APIService   *apiregistration.APIService
		ExpectError  bool
		ExpectCalled bool
	}{
		"valid hostname + CABundle": {
			APIService: &apiregistration.APIService{
				Spec: apiregistration.APIServiceSpec{
					CABundle: testCACrt,
					Group:    "mygroup",
					Version:  "v1",
					Service:  &apiregistration.ServiceReference{Name: "test-service", Namespace: "test-ns", Port: pointer.Int32Ptr(443)},
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			ExpectError:  false,
			ExpectCalled: true,
		},
		"invalid hostname + insecure": {
			APIService: &apiregistration.APIService{
				Spec: apiregistration.APIServiceSpec{
					InsecureSkipTLSVerify: true,
					Group:                 "mygroup",
					Version:               "v1",
					Service:               &apiregistration.ServiceReference{Name: "invalid-service", Namespace: "invalid-ns", Port: pointer.Int32Ptr(443)},
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			ExpectError:  false,
			ExpectCalled: true,
		},
		"invalid hostname + CABundle": {
			APIService: &apiregistration.APIService{
				Spec: apiregistration.APIServiceSpec{
					CABundle: testCACrt,
					Group:    "mygroup",
					Version:  "v1",
					Service:  &apiregistration.ServiceReference{Name: "invalid-service", Namespace: "invalid-ns", Port: pointer.Int32Ptr(443)},
				},
				Status: apiregistration.APIServiceStatus{
					Conditions: []apiregistration.APIServiceCondition{
						{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
					},
				},
			},
			ExpectError:  true,
			ExpectCalled: false,
		},
	}

	for k, tc := range testcases {
		tcName := k
		path := "/apis/" + tc.APIService.Spec.Group + "/" + tc.APIService.Spec.Version + "/foo"
		timesCalled := int32(0)

		func() { // Cleanup after each test case.
			backendHandler := http.NewServeMux()
			backendHandler.Handle(path, websocket.Handler(func(ws *websocket.Conn) {
				atomic.AddInt32(&timesCalled, 1)
				defer ws.Close()
				body := make([]byte, 5)
				ws.Read(body)
				ws.Write([]byte("hello " + string(body)))
			}))

			backendServer := httptest.NewUnstartedServer(backendHandler)
			cert, err := tls.X509KeyPair(svcCrt, svcKey)
			if err != nil {
				t.Errorf("https (valid hostname): %v", err)
				return
			}
			backendServer.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
			backendServer.StartTLS()
			defer backendServer.Close()

			defer func() {
				if called := atomic.LoadInt32(&timesCalled) > 0; called != tc.ExpectCalled {
					t.Errorf("%s: expected called=%v, got %v", tcName, tc.ExpectCalled, called)
				}
			}()

			serverURL, _ := url.Parse(backendServer.URL)
			proxyHandler := &proxyHandler{
				serviceResolver:            &mockedRouter{destinationHost: serverURL.Host},
				proxyTransport:             &http.Transport{},
				proxyCurrentCertKeyContent: func() ([]byte, []byte) { return emptyCert(), emptyCert() },
			}
			proxyHandler.updateAPIService(tc.APIService)
			aggregator := httptest.NewServer(contextHandler(proxyHandler, &user.DefaultInfo{Name: "username"}))
			defer aggregator.Close()

			ws, err := websocket.Dial("ws://"+aggregator.Listener.Addr().String()+path, "", "http://127.0.0.1/")
			if err != nil {
				if !tc.ExpectError {
					t.Errorf("%s: websocket dial err: %s", tcName, err)
				}
				return
			}
			defer ws.Close()
			if tc.ExpectError {
				t.Errorf("%s: expected websocket error, got none", tcName)
				return
			}

			if _, err := ws.Write([]byte("world")); err != nil {
				t.Errorf("%s: write err: %s", tcName, err)
				return
			}

			response := make([]byte, 20)
			n, err := ws.Read(response)
			if err != nil {
				t.Errorf("%s: read err: %s", tcName, err)
				return
			}
			if e, a := "hello world", string(response[0:n]); e != a {
				t.Errorf("%s: expected '%#v', got '%#v'", tcName, e, a)
				return
			}
		}()
	}
}

var testCACrt = []byte(`-----BEGIN CERTIFICATE-----
MIICxDCCAaygAwIBAgIBATANBgkqhkiG9w0BAQsFADASMRAwDgYDVQQDEwd0ZXN0
LWNhMCAXDTE3MDcyMDIxMTc1MloYDzIxMTcwNjI2MjExNzUzWjASMRAwDgYDVQQD
Ewd0ZXN0LWNhMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuv/sT2xH
VS1/uXVNAEIwvEb2yTMbXwP6FD38LWkc37Ri7YMB9xiXEDBrbr6K1JThsqyitBxU
22QIl53LUm6I7c/vej1tdYtE2rDVuviiiRgy6omR8imVSv9vU024rgDe+nC9zTT1
3aNKR03olCG6fkygdcZOghzlyQLhyh8LG75XdnLNksnakum2dNxQ5QIFmBKAuev3
A069oRMNjudot+t/nFP9UDZ8dL80PNTNPF22bPsnxiau7KLZ4I0Lf7gt6yHlNcue
Fd5sqzqsw/LUFJR5Xuo1+0e7NV3SwCH5CymG6hkboM4Rf5S3EDDyXTxPbXzbQHf1
7ksW6gjAxh4x/wIDAQABoyMwITAOBgNVHQ8BAf8EBAMCAqQwDwYDVR0TAQH/BAUw
AwEB/zANBgkqhkiG9w0BAQsFAAOCAQEATgmDrW1BjFp+Vmw6T+ojVK4lJuIoerGw
TCCqabHs6O1iWkNi5KsY6vV86tofBIEXsf6S3mV2jcBn87+CIbNHlHFKrXwmcydA
WOc0LWVqqoeqIvEcMNoWQskzmOOUDTanX9mXkirm8d8BljC351TH17rSjLGzFuNh
Cy48xyKFM7kPauNZGfCyaZsGbNJP3Keeu35dOLZMDdBJw7ZvYEUqX7MLOO+d7vlO
JGNA5jsU2uBnSo6qsjxfsbGemk2uRO0nLhplWurw+4qzA79D0lKNLtH9yTn12KZb
/kUpsOSCtLomjWwp67lQyA/yFXf897pSKMXbnIfZfIlDg51CI3U2Sw==
-----END CERTIFICATE-----`)

// valid for hostname test-service.test-ns.svc
// signed by testCACrt
var svcCrt = []byte(`-----BEGIN CERTIFICATE-----
MIIDDDCCAfSgAwIBAgIBBDANBgkqhkiG9w0BAQsFADASMRAwDgYDVQQDEwd0ZXN0
LWNhMCAXDTE3MDcyMDIxMjAzN1oYDzIxMTcwNjI2MjEyMDM4WjAjMSEwHwYDVQQD
Exh0ZXN0LXNlcnZpY2UudGVzdC1ucy5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQDOKgoTmlVeDhImiBLBccxdniKkS+FZSaoAEtoTvJG1wjk0ewzF
vKhjbHolJ+/qEANiQ6CpTz4hU3m/Iad6IrnmKd1jnkh9yKEaU32B2Xbh6VaV7Sca
Hv4cKWTe50sBvufZinTT8hlFcGufFlJIOLXya5t6HH1Ld7Xf2qwNqusHdmFlJko7
0By8jhTtD7+2OAJsIPQDWfAsXxFa6LeQ/lqS2DCFnp45DirTNetXoIH8ZJvTBjak
bQuAAA3H+61gRm1blIu8/JjHYTDOcUe5pFyrFLFPgA+eIcpIbzTD61UTNhVlusV2
eRrBr5BlRM13Zj6ZMcWp0Iiw5QI/W9QU7O4jAgMBAAGjWjBYMA4GA1UdDwEB/wQE
AwIFoDATBgNVHSUEDDAKBggrBgEFBQcDATAMBgNVHRMBAf8EAjAAMCMGA1UdEQQc
MBqCGHRlc3Qtc2VydmljZS50ZXN0LW5zLnN2YzANBgkqhkiG9w0BAQsFAAOCAQEA
kpULlml6Ct0cjOuHgDKUnTboFTUm2FHJY27p4NXUvXUMoipg/eSxk0r5JynzXrPa
jaJfY2bC45ixLjJv9irp9ER/lGYUcBQ8OHouXy+uKtA5YKHI3wYf8ITZrQwzRpsK
C5v7qDW+iyb9dn4T6qgpZvylKtkH5cH31hQooNgjZd5aEq3JSFnCM12IVqs/5kjL
NnbPXzeBQ9CHbC+mx7Mm6eSQVtGcQOy4yXFrh2/vrIB2t4gNeWaI1b+7l4MaJjV/
kRrOirhZaJ90ow/PdYrILtEAdpeC/2Varpr3l4rIKhkkop4gfPwaFeWhG38shH3E
eG5PW2waPpxJzEnGBoAWxQ==
-----END CERTIFICATE-----`)

var svcKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAzioKE5pVXg4SJogSwXHMXZ4ipEvhWUmqABLaE7yRtcI5NHsM
xbyoY2x6JSfv6hADYkOgqU8+IVN5vyGneiK55indY55IfcihGlN9gdl24elWle0n
Gh7+HClk3udLAb7n2Yp00/IZRXBrnxZSSDi18mubehx9S3e139qsDarrB3ZhZSZK
O9AcvI4U7Q+/tjgCbCD0A1nwLF8RWui3kP5aktgwhZ6eOQ4q0zXrV6CB/GSb0wY2
pG0LgAANx/utYEZtW5SLvPyYx2EwznFHuaRcqxSxT4APniHKSG80w+tVEzYVZbrF
dnkawa+QZUTNd2Y+mTHFqdCIsOUCP1vUFOzuIwIDAQABAoIBABiX9z/DZ2+i6hNi
pCojcyev154V1zoZiYgct5snIZK3Kq/SBgIIsWW66Q9Jplsbseuk+aN46oZ7OMjO
MPZm8ho84EYj+a3XozBKyWwWDxKADW4xLjr1e4bMgVX97Xq11V6kH6+w78bS1GPT
+9jVuw7CO3fjsiawjye3JFM1Enh/NeRLEpT/oaQoWIV8b0IQB0VyqrdxWOO0rQhd
xA5w39tAZPDQ79MbMQyNWtPgBy0FuulP0GB12PrEbE+SXxsFhWViEwdB5Qx6Gqsx
KGn9vB1oaeSuuKIAjyBV0rXszrGektorDchsOY9UQi1mQsPSvvRFTM9T3qqSFIpu
oPNQLvECgYEA3ox3WJGjEve6VI4RMRt0l6ZFswNbNaHcTMPVsayqsl9KfebG+uyn
Z7TyyoCRzZZQa+3Z9jjW3hAGM9e7MG8jkeHbZpJpZv9X7eB3dgq3eZ1Zt5dyoDrU
PTdIPA2efFAf6V1ejyqH9h6RPQMeAb4uFU9nbI4rPagMxRdp5qIveIUCgYEA7Scb
0zWplDit4EUo+Fq80wzItwJZv64my8KIkEPpW3Fu6UPQvY74qyhE2fCSCwHqRpYJ
jVylyE0GIMx42kjwBgOpi4yEg8M3uMTal+Iy9SgrxZ5cPetaFpEF3Wk7/tz6ppr+
wnZQTO2WH3YLzv7JIWVrOKuBNVfNEbguVFWw4IcCgYB54mp2uoSancySBKDLyWKo
r6raqQrqK7TQ4iyGO6/dMy1EGQF/ad8hgEu8tn+kHh/7jG/kVyruwc3z1MIze5r6
ib00xxktDMnmgRpMLwBffdsmHq7rrGyS/lT0du0G3ocrszRXqo5+MC2RQcTMZZEt
oKhfHtn10bT0uKcKZmcjVQKBgEls2WWccMOuhM8yOowic+IYTDC1bpo1Tle6BFQ+
YoroZQGd+IwoLv+3ORINNPppfmKaY5y7+aw5hNM025oiCQajraPCPukY0TDI6jEq
XMKgzGSkMkUNkFf6UMmLooK3Yneg94232gbnbJqTDvbo1dccMoVaPGgKpjh9QQLl
gR0TAoGACFOvhl8txfbkwLeuNeunyOPL7J4nIccthgd2ioFOr3HTou6wzN++vYTa
a3OF9jH5Z7m6X1rrwn6J1+Gw9sBme38/GeGXHigsBI/8WaTvyuppyVIXOVPoTvVf
VYsTwo5YgV1HzDkV+BNmBCw1GYcGXAElhJI+dCsgQuuU6TKzgl8=
-----END RSA PRIVATE KEY-----`)

func TestGetContextForNewRequest(t *testing.T) {
	done := make(chan struct{})
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		<-done // never return so that we're certain to return base on timeout
	}))
	defer server.Close()
	defer close(done)

	proxyServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		location, err := url.Parse(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		location.Path = req.URL.Path

		nestedReq := req.WithContext(genericapirequest.WithRequestInfo(req.Context(), &genericapirequest.RequestInfo{Path: req.URL.Path}))
		newReq, cancelFn := newRequestForProxy(location, nestedReq)
		defer cancelFn()

		theproxy := proxy.NewUpgradeAwareHandler(location, server.Client().Transport, true, false, &responder{w: w})
		theproxy.ServeHTTP(w, newReq)
	}))
	defer proxyServer.Close()

	// normal clients will not be setting a timeout, don't set one here.  Our proxy logic should construct this for us
	resp, err := proxyServer.Client().Get(proxyServer.URL + "/apis/group/version")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Error(err)
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "context deadline exceeded") {
		t.Error(string(body))
	}

}

// TestProxyCertReload verifies that the proxy reloading of certificates work
// to be able to test the reloading it starts a server with client auth enabled
// it first uses certs that does not match the client CA so the verification fails - expecting HTTP 503
// then we write correct client certs to the disk, expecting the proxy to reload the cert and use it for the next request
//
// Note: this test doesn't use apiserviceRegistrationController nor it doesn't start DynamicServingContentFromFiles controller
// instead it manually calls to updateAPIService and RunOnce to reload the certificate
func TestProxyCertReload(t *testing.T) {
	// STEP 1: set up a backend server that will require the client certificate
	//         this server uses clientCaCrt to validate the client certificate
	backendHandler := &targetHTTPHandler{}
	backendServer := httptest.NewUnstartedServer(backendHandler)
	if cert, err := tls.X509KeyPair(backendCertificate, backendKey); err != nil {
		t.Fatal(err)
	} else {
		caCertPool := x509.NewCertPool()
		// we're testing this while enabling MTLS
		caCertPool.AppendCertsFromPEM(clientCaCrt)
		backendServer.TLS = &tls.Config{Certificates: []tls.Certificate{cert}, ClientAuth: tls.RequireAndVerifyClientCert, ClientCAs: caCertPool}
	}
	backendServer.StartTLS()
	defer backendServer.Close()

	// STEP 2: set up the aggregator that will use an invalid certificate (it won't be validated by the clientCA) to auth against the backend server
	aggregatorHandler := &proxyHandler{
		localDelegate:   http.NewServeMux(),
		serviceResolver: &mockedRouter{destinationHost: backendServer.Listener.Addr().String()},
	}
	certFile, keyFile, dir := getCertAndKeyPaths(t)
	writeCerts(certFile, keyFile, backendCertificate, backendKey, t)

	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("Unable to clean up test directory %q: %v", dir, err)
		}
	}()

	certProvider, err := dynamiccertificates.NewDynamicServingContentFromFiles("test", certFile, keyFile)
	if err != nil {
		t.Fatalf("Unable to create dynamic certificates: %v", err)
	}
	err = certProvider.RunOnce()
	if err != nil {
		t.Fatalf("Unable to load dynamic certificates: %v", err)
	}
	aggregatorHandler.proxyCurrentCertKeyContent = certProvider.CurrentCertKeyContent

	apiService := &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1.foo"},
		Spec: apiregistration.APIServiceSpec{
			Service:  &apiregistration.ServiceReference{Name: "test-service2", Namespace: "test-ns", Port: pointer.Int32Ptr(443)},
			Group:    "foo",
			Version:  "v1",
			CABundle: backendCaCertificate, // used to validate backendCertificate
		},
		Status: apiregistration.APIServiceStatus{
			Conditions: []apiregistration.APIServiceCondition{
				{Type: apiregistration.Available, Status: apiregistration.ConditionTrue},
			},
		},
	}
	aggregatorHandler.updateAPIService(apiService)

	server := httptest.NewServer(contextHandler(aggregatorHandler, &user.DefaultInfo{
		Name:   "username",
		Groups: []string{"one", "two"},
	}))
	defer server.Close()

	resp, err := http.Get(server.URL + "/request/path")
	if err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("Expected status code 503 but got %d", resp.StatusCode)
	}

	// STEP 3: swap the certificate used by the aggregator to auth against the backend server and verify the request passes
	//         note that this step uses the certificate that can be validated by the backend server with clientCaCrt
	writeCerts(certFile, keyFile, clientCert, clientKey, t)
	err = certProvider.RunOnce()
	if err != nil {
		t.Fatalf("Expected no error when refreshing dynamic certs, got %v", err)
	}
	aggregatorHandler.updateAPIService(apiService)

	resp, err = http.Get(server.URL + "/request/path")
	if err != nil {
		t.Errorf("%v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected status code 200 but got %d", resp.StatusCode)
	}
}

func getCertAndKeyPaths(t *testing.T) (string, string, string) {
	dir, err := ioutil.TempDir(os.TempDir(), "k8s-test-handler-proxy-cert")
	if err != nil {
		t.Fatalf("Unable to create the test directory %q: %v", dir, err)
	}
	certFile := filepath.Join(dir, "certfile.pem")
	keyFile := filepath.Join(dir, "keytfile.pem")
	return certFile, keyFile, dir
}

func writeCerts(certFile, keyFile string, certContent, keyContent []byte, t *testing.T) {
	if err := ioutil.WriteFile(certFile, certContent, 0600); err != nil {
		t.Fatalf("Unable to create the file %q: %v", certFile, err)
	}
	if err := ioutil.WriteFile(keyFile, keyContent, 0600); err != nil {
		t.Fatalf("Unable to create the file %q: %v", keyFile, err)
	}
}

// cert and ca for client auth
var clientCert = []byte(`-----BEGIN CERTIFICATE-----
MIIFaDCCA1ACAWUwDQYJKoZIhvcNAQEFBQAwejELMAkGA1UEBhMCVVMxEzARBgNV
BAgMCkNhbGlmb3JuaWExFjAUBgNVBAcMDU1vdW50YWluIFZpZXcxGDAWBgNVBAoM
D015IG9yZ2FuaXphdGlvbjEQMA4GA1UECwwHTXkgdW5pdDESMBAGA1UEAwwJbG9j
YWxob3N0MB4XDTIwMDUyMjA4MTA1MVoXDTIxMDUyMjA4MTA1MVowejELMAkGA1UE
BhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExFjAUBgNVBAcMDU1vdW50YWluIFZp
ZXcxGDAWBgNVBAoMD015IG9yZ2FuaXphdGlvbjEQMA4GA1UECwwHTXkgdW5pdDES
MBAGA1UEAwwJbG9jYWxob3N0MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKC
AgEAwdDdguS2eVb950cmuyK/fTEBy+I1OFwPSg6S2zF5v/98Sva87Y/qFBrv1EzY
usU+OWuH0nnyk14bOGl+imbvk+tdiXr4i8tIY8QnBrUbyNvPwemcRejQQb1P5YX0
An3BS8vckt1e1zahhyb+Uch/ApLFzv3nOEGg7OTA5vfyNs/OUcaz7XuKrFQipxLA
wEpPbukI8ThH2uLwiRxWUrLGmOeWocM4JFCk6LaQLWkTzl9WgKTYwzrI24LaUgb6
0urlUi0bmE8AJRZBdmVCiEapxiHDre8c3CaLh8aF1LQ95ZraF8NZAvMxJvSK0R7I
05V+eZH+xdBH2n5naLjVuvm96VPbDGlcWRwi+ZKZXAvi6YMNJ5g564u2Nl+eACtd
9Kg6C9AIU8vSX9WrX4UcwaohQVjxUmHNL6YqHXhltyPdN3coFxDSPyp46x8Y2BIW
s1x1qnlor5xOOQhYPoIQzMgrgJw6wRLWdIkyP/NOazSwet2i4cpeLD3wgXpuylQp
Of06WChGN7NRx9JQSA7y6JKJq38jyB4+iNpU7NfkCQQndwvowPUBOSXNAUOgv2Qt
QEiODhNPsHhSHM6L4xSpwFzh7dDywpPCeb6Fzyp/EslaLiFoEQr2Wc0xM/Xssqa6
yBjSpATBqP1exQVr7LQn50lf9penN4FOQRZ9k/49DLX1RFUCAwEAATANBgkqhkiG
9w0BAQUFAAOCAgEAVyFuPhtyDMi8FxD00fqnAxwnr7IyNBwYuQivu7gXKwQ2U9v1
LSqDxvUft6sDWNUl/2f+Lga3CaVJ7FJL/rOwU5APkD4lcc43UcUv8pN2QAVFUs2h
8MPEZnM2oHEA3M77Yr1RZUHE24pHsv3Bi0u7w8kPhFb7ebAbfXAHIWkekPejroso
fOC2W8PXGqCJcpuIrAzIRvu/Ia0Cu4bmSZp4pK4lilgmUCr5LTc3YeNuAvbqco8f
mhXJ+qR4PYWkldgOdhz7eajKF0JP6R8pQacCTZ5OM1y9tg3yN6BEKus3EojpDtqs
5cTegj914lnNXI/bod6kqnuMT1sfnt2y8AmUcgD+NMhw6dG6zJI1Jf+01G2q3HCn
wtB0jPntk1hRepVkLfSvxoMofkjESHSVstYiGRQWQziFq98ei59uW1ZNpP/yVJGb
I7eM/b3vnFUBX2eypfVyY7+vBCxvgRjmpKnOuhCgm2bla1Ho7XUz1OvGkYfnHM3u
lUiTnAdNXQEf1Y2OjWeHeQeoeJ7gJiwJhMH8yZIierLHDP7FbBSLZ+VZW4Wfe6vT
WJ4no8kkD5ROWBNf0c0dt2uip6dZ5L2zMrqeUrhpy59ZhoZoMP5cmY/sfTzpRzNO
KitvR2SwVL12T6pAkwq3ItdiGZ16x5XrYv22H0jP8R6MCd59Sfnz9wWdY1Q=
-----END CERTIFICATE-----`)
var clientKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
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
-----END RSA PRIVATE KEY-----`)
var backendCertificate = []byte(`-----BEGIN CERTIFICATE-----
MIIDiDCCAnCgAwIBAgIUJgFO0eypsogvehekMVrJ/eXj1MYwDQYJKoZIhvcNAQEL
BQAwXDELMAkGA1UEBhMCeHgxCjAIBgNVBAgMAXgxCjAIBgNVBAcMAXgxCjAIBgNV
BAoMAXgxCjAIBgNVBAsMAXgxCzAJBgNVBAMMAmNhMRAwDgYJKoZIhvcNAQkBFgF4
MB4XDTIwMDcyMzE5NTEwMFoXDTI1MDcyMjE5NTEwMFowJDEiMCAGA1UEAxMZdGVz
dC1zZXJ2aWNlMi50ZXN0LW5zLnN2YzCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAPSmCdoH7RzBeGaGBGqBOV1I4Ex2Da2kUCPVeNfW3mPpJTUVi+QLwSDS
YTLnyw9tHRQgwV+rU1GTJSpcEk6CpiYdMavGnyH0C0iXKqXeJDfbU19ioUIInMxG
OkfcL98fWgj/mih52zjBIh5f9Q7gCmzH6di4zXMQODTiDhrcjPzmMtMPvRJs+kol
4Hh+tWH3s/hOeqiaWpw01UKis181SdEgX2uwNJYdHBbKF390vVIx/qpcFKUAw9to
CviyRMKv+DAK0jBoAsQVIU1Kt4reUrWyzonyO2wUrJmmFs997O04exkNlmFKa+bV
cA8DtBhX4hTMKRFIAaYb4Kh5v5Pg0l0CAwEAAaN6MHgwDgYDVR0PAQH/BAQDAgWg
MBMGA1UdJQQMMAoGCCsGAQUFBwMBMAwGA1UdEwEB/wQCMAAwHQYDVR0OBBYEFIe3
Cry9ZA6zIWMvikdBZwBVprNzMCQGA1UdEQQdMBuCGXRlc3Qtc2VydmljZTIudGVz
dC1ucy5zdmMwDQYJKoZIhvcNAQELBQADggEBACg/8So7bv3e2UxL6TDAK43IV7lR
N+fIdkrxboiJY9XH7lPK4Cm7gNmxjzzlBeCbBRBNRrcbk4BoBRrDXMi2W13dtLE4
jmGPke7MFu6C9J26GrfiIchMyZAgFTGOucs1SOXr5hoaOnLkm9H3ZlkhWgIf/EUX
B4WEHdxKZCYTlUoPFsfcZ3vImo2zhelo5RyG+P8aACc1V7cSaDbZ6CHEdTsP2E70
9DKQHfkRr4MgrngoYiIZyj3IHK2kWnavLo0/XxBeoNVeenOrfmZAJ6QDSFAvTpMN
wWcx3Aj9jkGT+Cam2dvHFA+QaCni2uzOXlTyjLWwTjhc+Ml7FAL2Lc7U07c=
-----END CERTIFICATE-----`)
var backendKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA9KYJ2gftHMF4ZoYEaoE5XUjgTHYNraRQI9V419beY+klNRWL
5AvBINJhMufLD20dFCDBX6tTUZMlKlwSToKmJh0xq8afIfQLSJcqpd4kN9tTX2Kh
QgiczEY6R9wv3x9aCP+aKHnbOMEiHl/1DuAKbMfp2LjNcxA4NOIOGtyM/OYy0w+9
Emz6SiXgeH61Yfez+E56qJpanDTVQqKzXzVJ0SBfa7A0lh0cFsoXf3S9UjH+qlwU
pQDD22gK+LJEwq/4MArSMGgCxBUhTUq3it5StbLOifI7bBSsmaYWz33s7Th7GQ2W
YUpr5tVwDwO0GFfiFMwpEUgBphvgqHm/k+DSXQIDAQABAoIBAGzU2BkX4ZEjN85T
2+8NIVmwK6eX9KnEKKpoMmPCABhuBNFCjoKaAAX70KV2m8x2+7KSh7NpYZ0uWiAn
6TTnxcW6wvfpWa0fBU37gUtcMLxwYvxRwe7AKhBtRUvmVZ1qMwFBw3AyFSWANQ9S
HI/LdpfBrvNr8mk3U+mijifA6S8u0co/QwlHmh1fRzLruP6VrTIAVs67+JvkKMBw
O3hxF/ImTIR8YwlPx4ckP4OXSftLTYKFVxDZBHtxyT5ED5GLx7nCPossL9mRpAYU
XLje+5K4UNoLSFu9SaSZbBUDqbsSUsyJTWX1J+AYEThPUywV9lVBBtUj8JKOQ9kr
i+Nt8HkCgYEA9o0WH97Orn/iyxe6KgbIGKPS46tcFGYAIgNTMEaeegfBIrg7kah3
NV84d/Im3lYShCjGrnuoOHY2Wz4/a0DCbf+bgJWB/ZHpE00z+gBjfPE94as7wxC2
TO4HYg5kiy3b1RKaXWvOBrQ5fpZvdYo5WjWweNF6rTCanVPH5g7fenMCgYEA/gZJ
THt54MJdUOTBR1GS3l3da4yYJPNgRAFBdp8FRc8u0CTYTfLo0oNFfJHu+F/Ph5dj
VWxhA+as+4rqJi+w8KZCCp/8LKjlJKzcCpv93E2UxM7e6WTa7Z/TmLi97i8FI39c
62B8XJTVW/IRTqojW0noY62FqYrIWZ8ymrWnO+8CgYBVp044ZD+JgARaajPSxehe
Jwvs7Gtg6s7BAka0TtRfsLH4TejkAZLoh9wmT4oRU/W61C+yDmOyud7IdCe0Kxtg
+5waX9Z5MWe3vOqBwADQNz84VzS73+J1d3w5JKbpc1UcAQp/yiQZUCNpRvoR66Nh
I6XbU2s7H9eXMLQRyLj64QKBgQCSZfkUdQ0Wta2mE1A41BB6y0ny08JTeVf/mWGr
BZa6Vt854iIvOlFoEXOYiVpaFo26LUt4Tc/Tubvz9GlhvJaS+p6RFQb2jhgRfPYL
vz8dGjElA7yAcjmiPTxrhf0gKkUh4iMhHChQCw6zwNyso21hDUU7PSQNRAiXbiJx
+0L4TQKBgQDyAry0K7dTbEmsacFpHsxqE/F0O2tmFE0WzrDkKkjVu38jshMhDu5D
1X179FWkKL6dYrFdig5SHBM2T3Yjha6VF7o1apYqj5HoVhS/mz80xXCqUBVrg88v
aOz9qqvSZQDZYwbOfr/vLMvJMp4M5gWWdxgaqoteLo1dQU20cYwlqA==
-----END RSA PRIVATE KEY-----`)
var backendCaCertificate = []byte(`-----BEGIN CERTIFICATE-----
MIIDNDCCAhwCCQD9J4txHjsBLTANBgkqhkiG9w0BAQsFADBcMQswCQYDVQQGEwJ4
eDEKMAgGA1UECAwBeDEKMAgGA1UEBwwBeDEKMAgGA1UECgwBeDEKMAgGA1UECwwB
eDELMAkGA1UEAwwCY2ExEDAOBgkqhkiG9w0BCQEWAXgwHhcNMjAwNzIzMTk1NjA3
WhcNMjEwNzIzMTk1NjA3WjBcMQswCQYDVQQGEwJ4eDEKMAgGA1UECAwBeDEKMAgG
A1UEBwwBeDEKMAgGA1UECgwBeDEKMAgGA1UECwwBeDELMAkGA1UEAwwCY2ExEDAO
BgkqhkiG9w0BCQEWAXgwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDu
lMNXqY4D9EhgkDrKYcQD+Qai0rSWXSx2u28NCsQ36oR+J6UocSA1+0aFnZHo2s2P
sRndP1/AqEELpYl4XtAqrDUrhgH0KuvlIIp0LLDGLoJaOvv89VnNyuqSg4KtkGNZ
leiEBOUk7vITQkWtt3+QNVZPx/lMWUjI8QCvtaVKNcd7C9P6HCTuSbfkkHUdLLwM
Ud1zp6T/YHFxGGNtN0XDMapQJid4pfQF4vj89H5JT4GArOgUTEDfkVy7Go5/1F8I
X5sG9WbCLcClfPAHFZNM1igTMVEau0uF6wkL3UIBImyExFEwgN3HT88kIVN+tZSZ
n7bEnx9uWQKExZNOwf6TAgMBAAEwDQYJKoZIhvcNAQELBQADggEBAH5dU7u4+RRD
C3nodTMJjd4UD7kdO2Stp9sLsPsbFhWQGpW10J0v+m7+ISgxOfbpNU9NI3dlDsCo
h4sG4MYfJio28r7ohkbzgBc3xKpLKK54XvPFhmrUiHccJT0PV6F3MJyBCn1Bxdya
+phcQapwRda/ytrqV5Xf55Od1n9plPnl+eV89teBV8qpd/cufIiFPeO8zhHI3wfh
AUbPo2yBwdFXKZxLo5rR3yTlJBkRjfodHNTcJffio2fEzPQumP+qCkHWx37aR3kW
9iRvhus3UcCluc76CrV2XJvXzgbXjU0YBDqRmiShVCGm+eTftq1v9wDLRhgadxPu
RzFJLb91brg=
-----END CERTIFICATE-----`)
var clientCaCrt = []byte(`-----BEGIN CERTIFICATE-----
MIIFcDCCA1gCCQDgTBDe5gjLSDANBgkqhkiG9w0BAQsFADB6MQswCQYDVQQGEwJV
UzETMBEGA1UECAwKQ2FsaWZvcm5pYTEWMBQGA1UEBwwNTW91bnRhaW4gVmlldzEY
MBYGA1UECgwPTXkgb3JnYW5pemF0aW9uMRAwDgYDVQQLDAdNeSB1bml0MRIwEAYD
VQQDDAlsb2NhbGhvc3QwHhcNMjAwNTIyMDczNTQxWhcNMzAwNTIwMDczNTQxWjB6
MQswCQYDVQQGEwJVUzETMBEGA1UECAwKQ2FsaWZvcm5pYTEWMBQGA1UEBwwNTW91
bnRhaW4gVmlldzEYMBYGA1UECgwPTXkgb3JnYW5pemF0aW9uMRAwDgYDVQQLDAdN
eSB1bml0MRIwEAYDVQQDDAlsb2NhbGhvc3QwggIiMA0GCSqGSIb3DQEBAQUAA4IC
DwAwggIKAoICAQCj89Np0QeBHn6pyDUrzd45Ow9oHTBgvrDAmhND0i+WkcoDAOrX
V4W6aNLibM/5stR7PRwl93cwkLawE84YHevH7/69EeTjYqIUUTF/Otxh+qTZMDUu
Z3hcW7Pu/JnfHbmliR+ci4kr7KkVAYHJtT9DcyWAs5KUudPGKpQprVKtnJ04J/hV
gDrZbBVKU/N7Ik0ta0MWy97LegbRaGrcY/h7ICoaeMDL0UGU8b61tUCVObmhAnM6
jK6xk/PtMk2d4we3yIWhowrGbp8vxN25WtFXIvJfyrrLFvpsl1f/dLwOzxU8RIt0
soXkF5ig6BkjzXtG+WM8ZHBGgL1salP6B0IhLjIjsyZVNORyRJEn0SxDnVKtYLuO
tjcDZb1Ij/KzWdyXCMD8uJECO9z1Zt2kCfsZDjCal+nyas9Otn3djERaGaaQZd1q
oL/ioQSTgRhHO3Jx721YaetfM5Bf4h/xGIZlR0wsUPM86rN3s5LcN01C8MLMt3op
l5ECQE4zlCq2j7EZwlTcq7B5onwUDqQYImD/AHIaOMAeAxHCfeGAl9t+84pnd9iU
BG3XnaSdrhJJApK7Pa7peu7FDaeAkl71VQW0URHjCedCHNdqk1pbsCJMKfpMuRWp
LldTG83/bCyuNsku8rkKmkY25MSt80EpyYxg0ZfP2GqSX9+wbH67EJlEfQIDAQAB
MA0GCSqGSIb3DQEBCwUAA4ICAQAqaCc/LkDdJq/QS27qhCKEI885ZYOHuk8N64G6
7Mfk6YhkSf5/Ln4qwP0f4HJCgupRMRLFs96qIh2HeEvytQk/xd8j111BHBUmjx3E
tS271x6PTkwkHa5j7kxE85b/wnUjVZ58NKccstp/Ub/ajssPdS7Ohzm0DGTjktja
Bavju5Q3fyBl4OmICOVDqIVBqNUfszesBtW9QcSgW7VcL2X+5/H/tu2YYnJG8IXp
v4uJRZ2rimhQZFFvcihCMN6wR7M5hqDPyffloHy+tFYFNd+Wc+RHU/DU2i83ySa/
BwRD5J8iTHplDFosCo1u6EoALWQx/WM/l4E9P895LFFoF/8tvHUeLAQXjUbqEPUq
sbHlhZK18vxYUu/n+OtRdHDimjjoEWZHgoUNnNardukcLdGvk2dbmWltd8NA+kjh
e88NQn5x5mKUfENtK/GYKN4duguR6mOKlKBuobLcjeplnrHcRoWsvYOPJr0L9Ki3
F1XEUPu0NgZyx5kTX3znm+7UV/W1rZeRppHSeqVfwHE+N2FEds65rMF1sEvw3fZv
mwAA1eyVJXIGum9MHf9XAgjjyubtwzPdCE6NQ9nYBuXr6sAqZx6irTHrtHl7zmbJ
St3GLAs3qHVMa6Va1imhvInbV6m9CauCbt4vAs6xVtR/jIaq1NKHP63f+bHp8hhK
4ulSKQ==
-----END CERTIFICATE-----`)
