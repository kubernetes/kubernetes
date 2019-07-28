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
	"fmt"
	"io/ioutil"
	"k8s.io/utils/pointer"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"golang.org/x/net/websocket"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiregistration "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
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
				localDelegate:   http.NewServeMux(),
				serviceResolver: serviceResolver,
				proxyTransport:  &http.Transport{},
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
				serviceResolver: &mockedRouter{destinationHost: serverURL.Host},
				proxyTransport:  &http.Transport{},
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
