package fixchain

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/google/certificate-transparency/go/x509"
)

type testRoundTripper struct {
	t         *testing.T
	test      *fixAndLogTest
	testIndex int
	seen      []bool
}

func (rt testRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	url := fmt.Sprintf("%s://%s%s", request.URL.Scheme, request.URL.Host, request.URL.Path)
	switch url {
	case "https://ct.googleapis.com/pilot/ct/v1/get-roots":
		b := stringRootsToJSON([]string{verisignRoot, testRoot})
		return &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         request.Proto,
			ProtoMajor:    request.ProtoMajor,
			ProtoMinor:    request.ProtoMinor,
			Body:          &bytesReadCloser{bytes.NewReader(b)},
			ContentLength: int64(len(b)),
			Request:       request,
		}, nil
	case "https://ct.googleapis.com/pilot/ct/v1/add-chain":
		body, err := ioutil.ReadAll(request.Body)
		request.Body.Close()
		if err != nil {
			errStr := fmt.Sprintf("#%d: Could not read request body: %s", rt.testIndex, err.Error())
			rt.t.Error(errStr)
			return nil, errors.New(errStr)
		}

		type Chain struct {
			Chain [][]byte
		}
		var chainBytes Chain
		err = json.Unmarshal(body, &chainBytes)
		if err != nil {
			errStr := fmt.Sprintf("#%d: Could not unmarshal json: %s", rt.testIndex, err.Error())
			rt.t.Error(errStr)
			return nil, errors.New(errStr)
		}
		var chain []*x509.Certificate
		for _, certBytes := range chainBytes.Chain {
			cert, err := x509.ParseCertificate(certBytes)
			if err != nil {
				errStr := fmt.Sprintf("#%d: Could not parse certificate: %s", rt.testIndex, err.Error())
				rt.t.Error(errStr)
				return nil, errors.New(errStr)
			}
			chain = append(chain, cert)
		}

	TryNextExpected:
		for i, expChain := range rt.test.expLoggedChains {
			if rt.seen[i] || len(chain) != len(expChain) {
				continue
			}
			for j, cert := range chain {
				if !strings.Contains(nameToKey(&cert.Subject), expChain[j]) {
					continue TryNextExpected
				}
			}
			rt.seen[i] = true
			goto Return
		}
		rt.t.Errorf("#%d: Logged chain was not expected: %s", rt.testIndex, chainToDebugString(chain))
	Return:
		return &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         request.Proto,
			ProtoMajor:    request.ProtoMajor,
			ProtoMinor:    request.ProtoMinor,
			Body:          &bytesReadCloser{bytes.NewReader([]byte(""))},
			ContentLength: 0,
			Request:       request,
		}, nil
	default:
		var cert string
		switch url {
		case "http://www.thawte.com/repository/Thawte_SGC_CA.crt":
			cert = thawteIntermediate
		case "http://crt.comodoca.com/EssentialSSLCA_2.crt":
			cert = comodoIntermediate
		case "http://crt.comodoca.com/ComodoUTNSGCCA.crt":
			cert = comodoRoot
		case "http://www.example.com/intermediate2.crt":
			cert = testIntermediate2
		case "http://www.example.com/intermediate1.crt":
			cert = testIntermediate1
		case "http://www.example.com/ca.crt":
			cert = testRoot
		case "http://www.example.com/a.crt":
			cert = testA
		case "http://www.example.com/b.crt":
			cert = testB
		default:
			return nil, fmt.Errorf("can't reach url %s", url)
		}

		return &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         request.Proto,
			ProtoMajor:    request.ProtoMajor,
			ProtoMinor:    request.ProtoMinor,
			Body:          &bytesReadCloser{bytes.NewReader([]byte(cert))},
			ContentLength: int64(len([]byte(cert))),
			Request:       request,
		}, nil
	}
}

// The round tripper used during testing of PostChainToLog() is used to check
// that the http requests sent by PostChainToLog() contain the right information
// for a Certificate Transparency log to be able to log the given chain
// (assuming the chain is valid).
type postTestRoundTripper struct {
	t         *testing.T
	test      *postTest
	testIndex int
}

func (rt postTestRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	// For tests that are checking the correct FixError type is returned:
	if rt.test.ferr.Type == PostFailed {
		return nil, errors.New("")
	}

	if rt.test.ferr.Type == LogPostFailed {
		return &http.Response{
			Status:        "501 Not Implemented",
			StatusCode:    501,
			Proto:         request.Proto,
			ProtoMajor:    request.ProtoMajor,
			ProtoMinor:    request.ProtoMinor,
			Body:          &bytesReadCloser{bytes.NewReader([]byte(""))},
			ContentLength: 0,
			Request:       request,
		}, nil
	}

	// For tests to check request sent to log looks right:
	// Check method used
	if request.Method != "POST" {
		rt.t.Errorf("#%d: expected request method to be POST, received %s", rt.testIndex, request.Method)
	}

	// Check URL
	if request.URL.Scheme != rt.test.urlScheme {
		rt.t.Errorf("#%d: Scheme: received %s, expected %s", rt.testIndex, request.URL.Scheme, rt.test.urlScheme)
	}
	if request.URL.Host != rt.test.urlHost {
		rt.t.Errorf("#%d: Host: received %s, expected %s", rt.testIndex, request.URL.Host, rt.test.urlHost)
	}
	if request.URL.Path != rt.test.urlPath {
		rt.t.Errorf("#%d: Path: received %s, expected %s", rt.testIndex, request.URL.Path, rt.test.urlPath)
	}

	// Check Body
	body, err := ioutil.ReadAll(request.Body)
	request.Body.Close()
	if err != nil {
		errStr := fmt.Sprintf("#%d: Could not read request body: %s", rt.testIndex, err.Error())
		rt.t.Error(errStr)
		return nil, errors.New(errStr)
	}

	// Create string in the format that the Certificate Transparency logs expect
	// the body of an add-chain request to be in.
	var encode = base64.StdEncoding.EncodeToString
	expStr := "{\"chain\":"
	if rt.test.chain == nil {
		expStr += "null"
	} else {
		expStr += "["
		for i, cert := range rt.test.chain {
			expStr += "\"" + encode(GetTestCertificateFromPEM(rt.t, cert).Raw) + "\""
			if i != len(rt.test.chain)-1 {
				expStr += ","
			}
		}
		expStr += "]"
	}
	expStr += "}"

	if string(body) != expStr {
		rt.t.Errorf("#%d: incorrect format of request body.  Received %s, expected %s", rt.testIndex, string(body), expStr)
	}

	// Return a response
	return &http.Response{
		Status:        "200 OK",
		StatusCode:    200,
		Proto:         request.Proto,
		ProtoMajor:    request.ProtoMajor,
		ProtoMinor:    request.ProtoMinor,
		Body:          &bytesReadCloser{bytes.NewReader([]byte(""))},
		ContentLength: 0,
		Request:       request,
	}, nil
}

type newLoggerTestRoundTripper struct{}

func (rt newLoggerTestRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	// Return a response
	return &http.Response{
		Status:        "200 OK",
		StatusCode:    200,
		Proto:         request.Proto,
		ProtoMajor:    request.ProtoMajor,
		ProtoMinor:    request.ProtoMinor,
		Body:          &bytesReadCloser{bytes.NewReader([]byte(""))},
		ContentLength: 0,
		Request:       request,
	}, nil
}

type rootCertsTestRoundTripper struct{}

func (rt rootCertsTestRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	url := fmt.Sprintf("%s://%s%s", request.URL.Scheme, request.URL.Host, request.URL.Path)
	if url == "https://ct.googleapis.com/pilot/ct/v1/get-roots" {
		b := stringRootsToJSON([]string{verisignRoot, comodoRoot})
		return &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         request.Proto,
			ProtoMajor:    request.ProtoMajor,
			ProtoMinor:    request.ProtoMinor,
			Body:          &bytesReadCloser{bytes.NewReader(b)},
			ContentLength: int64(len(b)),
			Request:       request,
		}, nil
	}
	return nil, errors.New("")
}
