// Package ocsp implements an OCSP responder based on a generic storage backend.
// It provides a couple of sample implementations.
// Because OCSP responders handle high query volumes, we have to be careful
// about how much logging we do. Error-level logs are reserved for problems
// internal to the server, that can be fixed by an administrator. Any type of
// incorrect input from a user should be logged and Info or below. For things
// that are logged on every request, Debug is the appropriate level.
package ocsp

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"time"

	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/certdb/dbconf"
	"github.com/cloudflare/cfssl/certdb/sql"
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

	// ErrNotFound indicates the request OCSP response was not found. It is used to
	// indicate that the responder should reply with unauthorizedErrorResponse.
	ErrNotFound = errors.New("Request OCSP Response not found")
)

// Source represents the logical source of OCSP responses, i.e.,
// the logic that actually chooses a response based on a request.  In
// order to create an actual responder, wrap one of these in a Responder
// object and pass it to http.Handle. By default the Responder will set
// the headers Cache-Control to "max-age=(response.NextUpdate-now), public, no-transform, must-revalidate",
// Last-Modified to response.ThisUpdate, Expires to response.NextUpdate,
// ETag to the SHA256 hash of the response, and Content-Type to
// application/ocsp-response. If you want to override these headers,
// or set extra headers, your source should return a http.Header
// with the headers you wish to set. If you don't want to set any
// extra headers you may return nil instead.
type Source interface {
	Response(*ocsp.Request) ([]byte, http.Header, error)
}

// An InMemorySource is a map from serialNumber -> der(response)
type InMemorySource map[string][]byte

// Response looks up an OCSP response to provide for a given request.
// InMemorySource looks up a response purely based on serial number,
// without regard to what issuer the request is asking for.
func (src InMemorySource) Response(request *ocsp.Request) ([]byte, http.Header, error) {
	response, present := src[request.SerialNumber.String()]
	if !present {
		return nil, nil, ErrNotFound
	}
	return response, nil, nil
}

// DBSource represnts a source of OCSP responses backed by the certdb package.
type DBSource struct {
	Accessor certdb.Accessor
}

// NewDBSource creates a new DBSource type with an associated dbAccessor.
func NewDBSource(dbAccessor certdb.Accessor) Source {
	return DBSource{
		Accessor: dbAccessor,
	}
}

// Response implements cfssl.ocsp.responder.Source, which returns the
// OCSP response in the Database for the given request with the expiration
// date furthest in the future.
func (src DBSource) Response(req *ocsp.Request) ([]byte, http.Header, error) {
	if req == nil {
		return nil, nil, errors.New("called with nil request")
	}

	aki := hex.EncodeToString(req.IssuerKeyHash)
	sn := req.SerialNumber

	if sn == nil {
		return nil, nil, errors.New("request contains no serial")
	}
	strSN := sn.String()

	if src.Accessor == nil {
		log.Errorf("No DB Accessor")
		return nil, nil, errors.New("called with nil DB accessor")
	}
	records, err := src.Accessor.GetOCSP(strSN, aki)

	// Response() logs when there are errors obtaining the OCSP response
	// and returns nil, false.
	if err != nil {
		log.Errorf("Error obtaining OCSP response: %s", err)
		return nil, nil, fmt.Errorf("failed to obtain OCSP response: %s", err)
	}

	if len(records) == 0 {
		return nil, nil, ErrNotFound
	}

	// Response() finds the OCSPRecord with the expiration date furthest in the future.
	cur := records[0]
	for _, rec := range records {
		if rec.Expiry.After(cur.Expiry) {
			cur = rec
		}
	}
	return []byte(cur.Body), nil, nil
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

// NewSourceFromDB reads the given database configuration file
// and creates a database data source for use with the OCSP responder
func NewSourceFromDB(DBConfigFile string) (Source, error) {
	// Load DB from cofiguration file
	db, err := dbconf.DBFromConfig(DBConfigFile)

	if err != nil {
		return nil, err
	}
	// Create accesor
	accessor := sql.NewAccessor(db)
	src := NewDBSource(accessor)

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
		clk:    clock.New(),
	}
}

func overrideHeaders(response http.ResponseWriter, headers http.Header) {
	for k, v := range headers {
		if len(v) == 1 {
			response.Header().Set(k, v[0])
		} else if len(v) > 1 {
			response.Header().Del(k)
			for _, e := range v {
				response.Header().Add(k, e)
			}
		}
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
			log.Debugf("Error decoding URL: %s", request.URL.Path)
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
		// In certain situations a UA may construct a request that has a double
		// slash between the host name and the base64 request body due to naively
		// constructing the request URL. In that case strip the leading slash
		// so that we can still decode the request.
		if len(base64RequestBytes) > 0 && base64RequestBytes[0] == '/' {
			base64RequestBytes = base64RequestBytes[1:]
		}
		requestBody, err = base64.StdEncoding.DecodeString(string(base64RequestBytes))
		if err != nil {
			log.Debugf("Error decoding base64 from URL: %s", string(base64RequestBytes))
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
	b64Body := base64.StdEncoding.EncodeToString(requestBody)
	log.Debugf("Received OCSP request: %s", b64Body)

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
		log.Debugf("Error decoding request body: %s", b64Body)
		response.WriteHeader(http.StatusBadRequest)
		response.Write(malformedRequestErrorResponse)
		return
	}

	// Look up OCSP response from source
	ocspResponse, headers, err := rs.Source.Response(ocspRequest)
	if err != nil {
		if err == ErrNotFound {
			log.Infof("No response found for request: serial %x, request body %s",
				ocspRequest.SerialNumber, b64Body)
			response.Write(unauthorizedErrorResponse)
			return
		}
		log.Infof("Error retrieving response for request: serial %x, request body %s, error: %s",
			ocspRequest.SerialNumber, b64Body, err)
		response.WriteHeader(http.StatusInternalServerError)
		response.Write(internalErrorErrorResponse)
		return
	}

	parsedResponse, err := ocsp.ParseResponse(ocspResponse, nil)
	if err != nil {
		log.Errorf("Error parsing response for serial %x: %s",
			ocspRequest.SerialNumber, err)
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

	if headers != nil {
		overrideHeaders(response, headers)
	}

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
