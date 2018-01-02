package storageos

// import (
// 	"net/http"
// 	"strings"
// 	"testing"
//
// 	"github.com/storageos/api/httpclient"
// 	"github.com/storageos/api/registry"
// 	"github.com/storageos/api/testutil"
// )
//
// func TestGetConnectionString(t *testing.T) {
// 	var tests = []struct {
// 		address  string
// 		expected string
// 	}{
// 		{"", "http://localhost:8000/v1"},
// 		{"localhost", "http://localhost:8000/v1"},
// 		{"localhost:9999", "http://localhost:9999/v1"},
// 		{":8888", "http://localhost:8888/v1"},
// 		{"xxx:9999", "http://xxx:9999/v1"},
// 		{"a.b.c:9999", "http://a.b.c:9999/v1"},
// 		{"1.2.3.4:9999", "http://1.2.3.4:9999/v1"},
// 	}
//
// 	for _, tt := range tests {
// 		actual := getConnectionString(tt.address)
// 		testutil.Expect(t, actual, tt.expected)
// 	}
// }
//
// func TestErrNotFound(t *testing.T) {
// 	server, teardown := testutil.NewTestingServer(404, "")
// 	defer teardown()
//
// 	reg := registry.NewDefaultRegistry()
// 	reg.Add(ControllerEndpoint, APIVersion, server.Addr)
// 	haClient := httpclient.NewHAHTTPClient(reg, &http.Client{}, ControllerEndpoint, APIVersion)
//
// 	manager := NewHTTPClientManager(server.Addr, haClient)
//
// 	resp, err := manager.request(http.MethodGet, "/", nil)
// 	testutil.Expect(t, err, nil)
//
// 	var obj interface{}
// 	err = manager.decode(resp, &obj)
// 	testutil.Refute(t, err, nil)
// 	testutil.Expect(t, err.Error(), ErrNotFound)
// }
//
// func TestErrHTTPOther(t *testing.T) {
// 	server, teardown := testutil.NewTestingServer(500, "")
// 	defer teardown()
//
// 	reg := registry.NewDefaultRegistry()
// 	reg.Add(ControllerEndpoint, APIVersion, server.Addr)
// 	haClient := httpclient.NewHAHTTPClient(reg, &http.Client{}, ControllerEndpoint, APIVersion)
//
// 	manager := NewHTTPClientManager(server.Addr, haClient)
//
// 	resp, err := manager.request(http.MethodGet, "/", nil)
// 	testutil.Expect(t, err, nil)
//
// 	var obj interface{}
// 	err = manager.decode(resp, &obj)
// 	testutil.Refute(t, err, nil)
// 	testutil.Expect(t, strings.HasPrefix(err.Error(), ErrHTTPOther), true)
// }
//
// func TestErrJSONUnMarshall(t *testing.T) {
// 	var data = `
//     {
//       xxx
//     }
//   `
// 	server, teardown := testutil.NewTestingServer(200, data)
// 	defer teardown()
//
// 	reg := registry.NewDefaultRegistry()
// 	reg.Add(ControllerEndpoint, APIVersion, server.Addr)
// 	haClient := httpclient.NewHAHTTPClient(reg, &http.Client{}, ControllerEndpoint, APIVersion)
//
// 	manager := NewHTTPClientManager(server.Addr, haClient)
//
// 	resp, err := manager.request(http.MethodGet, "/", nil)
// 	testutil.Expect(t, err, nil)
//
// 	var obj interface{}
// 	err = manager.decode(resp, &obj)
// 	testutil.Refute(t, err, nil)
// 	testutil.Expect(t, strings.HasPrefix(err.Error(), ErrJSONUnMarshall), true)
// }
