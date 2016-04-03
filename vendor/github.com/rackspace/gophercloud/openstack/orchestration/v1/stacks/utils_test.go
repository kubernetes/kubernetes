package stacks

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestTEFixFileRefs(t *testing.T) {
	te := TE{
		Bin: []byte(`string_to_replace: my fair lady`),
		fileMaps: map[string]string{
			"string_to_replace": "london bridge is falling down",
		},
	}
	te.fixFileRefs()
	th.AssertEquals(t, string(te.Bin), `london bridge is falling down: my fair lady`)
}

func TesttoStringKeys(t *testing.T) {
	var test1 interface{} = map[interface{}]interface{}{
		"Adam":  "Smith",
		"Isaac": "Newton",
	}
	result1, err := toStringKeys(test1)
	th.AssertNoErr(t, err)

	expected := map[string]interface{}{
		"Adam":  "Smith",
		"Isaac": "Newton",
	}
	th.AssertDeepEquals(t, result1, expected)
}

func TestGetBasePath(t *testing.T) {
	_, err := getBasePath()
	th.AssertNoErr(t, err)
}

// test if HTTP client can read file type URLS. Read the URL of this file
// because if this test is running, it means this file _must_ exist
func TestGetHTTPClient(t *testing.T) {
	client := getHTTPClient()
	baseurl, err := getBasePath()
	th.AssertNoErr(t, err)
	resp, err := client.Get(baseurl)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, resp.StatusCode, 200)
}

// Implement a fakeclient that can be used to mock out HTTP requests
type fakeClient struct {
	BaseClient Client
}

// this client's Get method first changes the URL given to point to
// testhelper's (th) endpoints. This is done because the http Mux does not seem
// to work for fqdns with the `file` scheme
func (c fakeClient) Get(url string) (*http.Response, error) {
	newurl := strings.Replace(url, "file://", th.Endpoint(), 1)
	return c.BaseClient.Get(newurl)
}

// test the fetch function
func TestFetch(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	baseurl, err := getBasePath()
	th.AssertNoErr(t, err)
	fakeURL := strings.Join([]string{baseurl, "file.yaml"}, "/")
	urlparsed, err := url.Parse(fakeURL)
	th.AssertNoErr(t, err)

	th.Mux.HandleFunc(urlparsed.Path, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		w.Header().Set("Content-Type", "application/jason")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Fee-fi-fo-fum")
	})

	client := fakeClient{BaseClient: getHTTPClient()}
	te := TE{
		URL:    "file.yaml",
		client: client,
	}
	err = te.Fetch()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, fakeURL, te.URL)
	th.AssertEquals(t, "Fee-fi-fo-fum", string(te.Bin))
}
