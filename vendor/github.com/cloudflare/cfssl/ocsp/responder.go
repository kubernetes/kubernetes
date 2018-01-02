package ocsp

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"time"

	"github.com/cloudflare/cfssl/log"
	"github.com/jmhodges/clock"
	"golang.org/x/crypto/ocsp"
)

var (
	malformedRequestErrorResponse = []byte{0x30, 0x03, 0x0A, 0x01, 0x01}
	internalErrorErrorResponse    = []byte{0x30, 0x03, 0x0A, 0x01, 0x02}
	tryLaterErrorResponse         = []byte{0x30, 0x03, 0x0A, 0x01, 0x03}
	sigRequredErrorResponse       = []byte{0x30, 0x03, 0x0A, 0x01, 0x05}
	unauthorizedErrorResponse     = []byte{0x30, 0x03, 0x0A, 0x01, 0x06}
)

// Source represents the logical source of OCSP responses, i.e.,
// the logic that actually chooses a response based on a request.  In
// order to create an actual responder, wrap one of these in a Responder
// object and pass it to http.Handle.
type Source interface {
	Response(*ocsp.Request) ([]byte, bool)
}

// An InMemorySource is a map from serialNumber -> der(response)
type InMemorySource map[string][]byte

// Response looks up an OCSP response to provide for a given request.
// InMemorySource looks up a response purely based on serial number,
// without regard to what issuer the request is asking for.
func (src InMemorySource) Response(request *ocsp.Request) (response []byte, present bool) {
	response, present = src[request.SerialNumber.String()]
	return
}

// NewSourceFromFile reads the named file into an InMemorySource.
// The file read by this function must contain whitespace-separated OCSP
// responses. Each OCSP response must be in base64-encoded DER form (i.e.,
// PEM without headers or whitespace).  Invalid responses are ignored.
// This function pulls the entire file into an InMemorySource.
func NewSourceFromFile(responseFile string) (Source, error) {
	fileContents, err := ioutil.ReadFile(responseFile)
	if err != nil {
		return nil, err
	}

	responsesB64 := regexp.MustCompile("\\s").Split(string(fileContents), -1)
	src := InMemorySource{}
	for _, b64 := range responsesB64 {
		// if the line/space is empty just skip
		if b64 == "" {
			continue
		}
		der, tmpErr := base64.StdEncoding.DecodeString(b64)
		if tmpErr != nil {
			log.Errorf("Base64 decode error %s on: %s", tmpErr, b64)
			continue
		}

		response, tmpErr := ocsp.ParseResponse(der, nil)
		if tmpErr != nil {
			log.Errorf("OCSP decode error %s on: %s", tmpErr, b64)
			continue
		}

		src[response.SerialNumber.String()] = der
	}

	log.Infof("Read %d OCSP responses", len(src))
	return src, nil
}

// A Responder object provides the HTTP logic to expose a
// Source of OCSP responses.
type Responder struct {
	Source Source
	clk    clock.Clock
}

// NewResponder instantiates a Responder with the give Source.
func NewResponder(source Source) *Responder {
	return &Responder{
		Source: source,
		clk:    clock.Default(),
	}
}

// A Responder can process both GET and POST requests.  The mapping
// from an OCSP request to an OCSP response is done by the Source;
// the Responder simply decodes the request, and passes back whatever
// response is provided by the source.
// Note: The caller must use http.StripPrefix to strip any path components
// (including '/') on GET requests.
// Do not use this responder in conjunction with http.NewServeMux, because the
// default handler will try to canonicalize path components by changing any
// strings of repeated '/' into a single '/', which will break the base64
// encoding.
func (rs Responder) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	// By default we set a 'max-age=0, no-cache' Cache-Control header, this
	// is only returned to the client if a valid authorized OCSP response
	// is not found or an error is returned. If a response if found the header
	// will be altered to contain the proper max-age and modifiers.
	response.Header().Add("Cache-Control", "max-age=0, no-cache")
	// Read response from request
	var requestBody []byte
	var err error
	switch request.Method {
	case "GET":
		base64Request, err := url.QueryUnescape(request.URL.Path)
		if err != nil {
			log.Errorf("Error decoding URL: %s", request.URL.Path)
			response.WriteHeader(http.StatusBadRequest)
			return
		}
		// url.QueryUnescape not only unescapes %2B escaping, but it additionally
		// turns the resulting '+' into a space, which makes base64 decoding fail.
		// So we go back afterwards and turn ' ' back into '+'. This means we
		// accept some malformed input that includes ' ' or %20, but that's fine.
		base64RequestBytes := []byte(base64Request)
		for i := range base64RequestBytes {
			if base64RequestBytes[i] == ' ' {
				base64RequestBytes[i] = '+'
			}
		}
		requestBody, err = base64.StdEncoding.DecodeString(string(base64RequestBytes))
		if err != nil {
			log.Errorf("Error decoding base64 from URL: %s", base64Request)
			response.WriteHeader(http.StatusBadRequest)
			return
		}
	case "POST":
		requestBody, err = ioutil.ReadAll(request.Body)
		if err != nil {
			log.Errorf("Problem reading body of POST: %s", err)
			response.WriteHeader(http.StatusBadRequest)
			return
		}
	default:
		response.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	// TODO log request
	b64Body := base64.StdEncoding.EncodeToString(requestBody)
	log.Infof("Received OCSP request: %s", b64Body)

	// All responses after this point will be OCSP.
	// We could check for the content type of the request, but that
	// seems unnecessariliy restrictive.
	response.Header().Add("Content-Type", "application/ocsp-response")

	// Parse response as an OCSP request
	// XXX: This fails if the request contains the nonce extension.
	//      We don't intend to support nonces anyway, but maybe we
	//      should return unauthorizedRequest instead of malformed.
	ocspRequest, err := ocsp.ParseRequest(requestBody)
	if err != nil {
		log.Errorf("Error decoding request body: %s", b64Body)
		response.WriteHeader(http.StatusBadRequest)
		response.Write(malformedRequestErrorResponse)
		return
	}

	// Look up OCSP response from source
	ocspResponse, found := rs.Source.Response(ocspRequest)
	if !found {
		log.Errorf("No response found for request: %s", b64Body)
		response.Write(unauthorizedErrorResponse)
		return
	}

	parsedResponse, err := ocsp.ParseResponse(ocspResponse, nil)
	if err != nil {
		log.Errorf("Error parsing response: %s", err)
		response.Write(unauthorizedErrorResponse)
		return
	}

	// Write OCSP response to response
	response.Header().Add("Last-Modified", parsedResponse.ThisUpdate.Format(time.RFC1123))
	response.Header().Add("Expires", parsedResponse.NextUpdate.Format(time.RFC1123))
	now := rs.clk.Now()
	maxAge := 0
	if now.Before(parsedResponse.NextUpdate) {
		maxAge = int(parsedResponse.NextUpdate.Sub(now) / time.Second)
	} else {
		// TODO(#530): we want max-age=0 but this is technically an authorized OCSP response
		//             (despite being stale) and 5019 forbids attaching no-cache
		maxAge = 0
	}
	response.Header().Set(
		"Cache-Control",
		fmt.Sprintf(
			"max-age=%d, public, no-transform, must-revalidate",
			maxAge,
		),
	)
	responseHash := sha256.Sum256(ocspResponse)
	response.Header().Add("ETag", fmt.Sprintf("\"%X\"", responseHash))

	// RFC 7232 says that a 304 response must contain the above
	// headers if they would also be sent for a 200 for the same
	// request, so we have to wait until here to do this
	if etag := request.Header.Get("If-None-Match"); etag != "" {
		if etag == fmt.Sprintf("\"%X\"", responseHash) {
			response.WriteHeader(http.StatusNotModified)
			return
		}
	}
	response.WriteHeader(http.StatusOK)
	response.Write(ocspResponse)
}
