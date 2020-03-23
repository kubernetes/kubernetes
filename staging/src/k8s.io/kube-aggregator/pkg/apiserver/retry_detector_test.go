package apiserver

import (
	"crypto/tls"
	"golang.org/x/net/websocket"
	"k8s.io/apiserver/pkg/authentication/user"
	apiregistration "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/utils/pointer"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
)

// TODO: test proxy with an HTTP Client as this would allow to test "NewSingleHostReverseProxy"
//func TestProxyRetriesHTTPClient(t *testing.T) { }

func TestProxyRetries(t *testing.T) {
	testcases := map[string]struct {
		APIService                   *apiregistration.APIService
		StartBackend                 bool
		BackendCustomHandler         func(http.ResponseWriter, *http.Request)
		ExpectError                  bool
		ExpectCalled                 bool
		ExpectServiceResolverCounter int
	}{
		"retry when connection was refused": {
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
			ExpectError:                  true,
			ExpectCalled:                 false,
			ExpectServiceResolverCounter: 4,
		},
		"no retry on proxy upgrade error (hijacked connections are not retriable)": {
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
			BackendCustomHandler: func(w http.ResponseWriter, req *http.Request) {
				// this handler will cause proxy upgrade error
				w.WriteHeader(http.StatusInternalServerError)
				return
			},
			StartBackend:                 true,
			ExpectError:                  true,
			ExpectCalled:                 false,
			ExpectServiceResolverCounter: 1,
		},
		/*
		"TODO: fix me rety on io.EOF when connecting to proxy": {
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
			BackendCustomHandler: func(w http.ResponseWriter, req *http.Request) {
				// TODO: this handler causes IO.EOF error which in not consider as retriable ATM
				w.WriteHeader(50)
				return
			},
			StartBackend:                 true,
			ExpectCalled:                 false,
			ExpectServiceResolverCounter: 1,
		},*/
		// TODO: happy path - no retries
	}

	for k, tc := range testcases {
		tcName := k
		path := "/apis/" + tc.APIService.Spec.Group + "/" + tc.APIService.Spec.Version + "/foo"
		timesCalled := int32(0)

		func() { // Cleanup after each test case.
			backendHandler := http.NewServeMux()
			if tc.BackendCustomHandler != nil {
				backendHandler.HandleFunc(path, tc.BackendCustomHandler)
			} else {
				backendHandler.Handle(path, websocket.Handler(func(ws *websocket.Conn) {
					atomic.AddInt32(&timesCalled, 1)
					defer ws.Close()
					body := make([]byte, 5)
					ws.Read(body)
					ws.Write([]byte("hello " + string(body)))
				}))
			}

			backendServer := httptest.NewUnstartedServer(backendHandler)
			cert, err := tls.X509KeyPair(svcCrt, svcKey)
			if err != nil {
				t.Errorf("https (valid hostname): %v", err)
				return
			}
			backendServer.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
			if tc.StartBackend {
				backendServer.StartTLS()
				defer backendServer.Close()
			}

			defer func() {
				if called := atomic.LoadInt32(&timesCalled) > 0; called != tc.ExpectCalled {
					t.Errorf("%s: expected called=%v, got %v", tcName, tc.ExpectCalled, called)
				}
			}()

			serverURL, _ := url.Parse(backendServer.URL)
			proxyHandler := &proxyHandler{
				serviceResolver: &mockedRouterWithCounter{&mockedRouter{destinationHost: serverURL.Host}, 0},
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
				actualRetries := proxyHandler.serviceResolver.(*mockedRouterWithCounter).counter
				if tc.ExpectServiceResolverCounter != actualRetries {
					t.Errorf("expected %d retries but got %d", tc.ExpectServiceResolverCounter, actualRetries)
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

type mockedRouterWithCounter struct {
	delegate *mockedRouter
	counter  int
}

func (r *mockedRouterWithCounter) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	r.counter++
	return r.delegate.ResolveEndpoint(name, name, port)
}
