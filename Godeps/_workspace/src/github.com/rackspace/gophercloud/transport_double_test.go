package gophercloud

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
)

type transport struct {
	called         int
	response       string
	expectTenantId bool
	tenantIdFound  bool
	status         int
}

func (t *transport) RoundTrip(req *http.Request) (rsp *http.Response, err error) {
	var authContainer *AuthContainer

	t.called++

	headers := make(http.Header)
	headers.Add("Content-Type", "application/xml; charset=UTF-8")

	body := ioutil.NopCloser(strings.NewReader(t.response))

	if t.status == 0 {
		t.status = 200
	}
	statusMsg := "OK"
	if (t.status < 200) || (299 < t.status) {
		statusMsg = "Error"
	}

	rsp = &http.Response{
		Status:           fmt.Sprintf("%d %s", t.status, statusMsg),
		StatusCode:       t.status,
		Proto:            "HTTP/1.1",
		ProtoMajor:       1,
		ProtoMinor:       1,
		Header:           headers,
		Body:             body,
		ContentLength:    -1,
		TransferEncoding: nil,
		Close:            true,
		Trailer:          nil,
		Request:          req,
	}

	bytes, err := ioutil.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(bytes, &authContainer)
	if err != nil {
		return nil, err
	}
	t.tenantIdFound = (authContainer.Auth.TenantId != "")

	if t.tenantIdFound != t.expectTenantId {
		rsp.Status = "500 Internal Server Error"
		rsp.StatusCode = 500
	}
	return
}

func newTransport() *transport {
	return &transport{}
}

func (t *transport) IgnoreTenantId() *transport {
	t.expectTenantId = false
	return t
}

func (t *transport) ExpectTenantId() *transport {
	t.expectTenantId = true
	return t
}

func (t *transport) WithResponse(r string) *transport {
	t.response = r
	t.status = 200
	return t
}

func (t *transport) WithError(code int) *transport {
	t.response = fmt.Sprintf("Error %d", code)
	t.status = code
	return t
}

func (t *transport) VerifyCalls(test *testing.T, n int) error {
	if t.called != n {
		err := fmt.Errorf("Expected Transport to be called %d times; found %d instead", n, t.called)
		test.Error(err)
		return err
	}
	return nil
}
