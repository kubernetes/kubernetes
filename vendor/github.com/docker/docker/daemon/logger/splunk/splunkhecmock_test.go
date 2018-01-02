package splunk

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"testing"
)

func (message *splunkMessage) EventAsString() (string, error) {
	if val, ok := message.Event.(string); ok {
		return val, nil
	}
	return "", fmt.Errorf("Cannot cast Event %v to string", message.Event)
}

func (message *splunkMessage) EventAsMap() (map[string]interface{}, error) {
	if val, ok := message.Event.(map[string]interface{}); ok {
		return val, nil
	}
	return nil, fmt.Errorf("Cannot cast Event %v to map", message.Event)
}

type HTTPEventCollectorMock struct {
	tcpAddr     *net.TCPAddr
	tcpListener *net.TCPListener

	token               string
	simulateServerError bool

	test *testing.T

	connectionVerified bool
	gzipEnabled        *bool
	messages           []*splunkMessage
	numOfRequests      int
}

func NewHTTPEventCollectorMock(t *testing.T) *HTTPEventCollectorMock {
	tcpAddr := &net.TCPAddr{IP: []byte{127, 0, 0, 1}, Port: 0, Zone: ""}
	tcpListener, err := net.ListenTCP("tcp", tcpAddr)
	if err != nil {
		t.Fatal(err)
	}
	return &HTTPEventCollectorMock{
		tcpAddr:             tcpAddr,
		tcpListener:         tcpListener,
		token:               "4642492F-D8BD-47F1-A005-0C08AE4657DF",
		simulateServerError: false,
		test:                t,
		connectionVerified:  false}
}

func (hec *HTTPEventCollectorMock) URL() string {
	return "http://" + hec.tcpListener.Addr().String()
}

func (hec *HTTPEventCollectorMock) Serve() error {
	return http.Serve(hec.tcpListener, hec)
}

func (hec *HTTPEventCollectorMock) Close() error {
	return hec.tcpListener.Close()
}

func (hec *HTTPEventCollectorMock) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	var err error

	hec.numOfRequests++

	if hec.simulateServerError {
		if request.Body != nil {
			defer request.Body.Close()
		}
		writer.WriteHeader(http.StatusInternalServerError)
		return
	}

	switch request.Method {
	case http.MethodOptions:
		// Verify that options method is getting called only once
		if hec.connectionVerified {
			hec.test.Errorf("Connection should not be verified more than once. Got second request with %s method.", request.Method)
		}
		hec.connectionVerified = true
		writer.WriteHeader(http.StatusOK)
	case http.MethodPost:
		// Always verify that Driver is using correct path to HEC
		if request.URL.String() != "/services/collector/event/1.0" {
			hec.test.Errorf("Unexpected path %v", request.URL)
		}
		defer request.Body.Close()

		if authorization, ok := request.Header["Authorization"]; !ok || authorization[0] != ("Splunk "+hec.token) {
			hec.test.Error("Authorization header is invalid.")
		}

		gzipEnabled := false
		if contentEncoding, ok := request.Header["Content-Encoding"]; ok && contentEncoding[0] == "gzip" {
			gzipEnabled = true
		}

		if hec.gzipEnabled == nil {
			hec.gzipEnabled = &gzipEnabled
		} else if gzipEnabled != *hec.gzipEnabled {
			// Nothing wrong with that, but we just know that Splunk Logging Driver does not do that
			hec.test.Error("Driver should not change Content Encoding.")
		}

		var gzipReader *gzip.Reader
		var reader io.Reader
		if gzipEnabled {
			gzipReader, err = gzip.NewReader(request.Body)
			if err != nil {
				hec.test.Fatal(err)
			}
			reader = gzipReader
		} else {
			reader = request.Body
		}

		// Read body
		var body []byte
		body, err = ioutil.ReadAll(reader)
		if err != nil {
			hec.test.Fatal(err)
		}

		// Parse message
		messageStart := 0
		for i := 0; i < len(body); i++ {
			if i == len(body)-1 || (body[i] == '}' && body[i+1] == '{') {
				var message splunkMessage
				err = json.Unmarshal(body[messageStart:i+1], &message)
				if err != nil {
					hec.test.Log(string(body[messageStart : i+1]))
					hec.test.Fatal(err)
				}
				hec.messages = append(hec.messages, &message)
				messageStart = i + 1
			}
		}

		if gzipEnabled {
			gzipReader.Close()
		}

		writer.WriteHeader(http.StatusOK)
	default:
		hec.test.Errorf("Unexpected HTTP method %s", http.MethodOptions)
		writer.WriteHeader(http.StatusBadRequest)
	}
}
