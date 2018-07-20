package authorization

import (
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
)

const (
	// AuthZApiRequest is the url for daemon request authorization
	AuthZApiRequest = "AuthZPlugin.AuthZReq"

	// AuthZApiResponse is the url for daemon response authorization
	AuthZApiResponse = "AuthZPlugin.AuthZRes"

	// AuthZApiImplements is the name of the interface all AuthZ plugins implement
	AuthZApiImplements = "authz"
)

// PeerCertificate is a wrapper around x509.Certificate which provides a sane
// encoding/decoding to/from PEM format and JSON.
type PeerCertificate x509.Certificate

// MarshalJSON returns the JSON encoded pem bytes of a PeerCertificate.
func (pc *PeerCertificate) MarshalJSON() ([]byte, error) {
	b := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: pc.Raw})
	return json.Marshal(b)
}

// UnmarshalJSON populates a new PeerCertificate struct from JSON data.
func (pc *PeerCertificate) UnmarshalJSON(b []byte) error {
	var buf []byte
	if err := json.Unmarshal(b, &buf); err != nil {
		return err
	}
	derBytes, _ := pem.Decode(buf)
	c, err := x509.ParseCertificate(derBytes.Bytes)
	if err != nil {
		return err
	}
	*pc = PeerCertificate(*c)
	return nil
}

// Request holds data required for authZ plugins
type Request struct {
	// User holds the user extracted by AuthN mechanism
	User string `json:"User,omitempty"`

	// UserAuthNMethod holds the mechanism used to extract user details (e.g., krb)
	UserAuthNMethod string `json:"UserAuthNMethod,omitempty"`

	// RequestMethod holds the HTTP method (GET/POST/PUT)
	RequestMethod string `json:"RequestMethod,omitempty"`

	// RequestUri holds the full HTTP uri (e.g., /v1.21/version)
	RequestURI string `json:"RequestUri,omitempty"`

	// RequestBody stores the raw request body sent to the docker daemon
	RequestBody []byte `json:"RequestBody,omitempty"`

	// RequestHeaders stores the raw request headers sent to the docker daemon
	RequestHeaders map[string]string `json:"RequestHeaders,omitempty"`

	// RequestPeerCertificates stores the request's TLS peer certificates in PEM format
	RequestPeerCertificates []*PeerCertificate `json:"RequestPeerCertificates,omitempty"`

	// ResponseStatusCode stores the status code returned from docker daemon
	ResponseStatusCode int `json:"ResponseStatusCode,omitempty"`

	// ResponseBody stores the raw response body sent from docker daemon
	ResponseBody []byte `json:"ResponseBody,omitempty"`

	// ResponseHeaders stores the response headers sent to the docker daemon
	ResponseHeaders map[string]string `json:"ResponseHeaders,omitempty"`
}

// Response represents authZ plugin response
type Response struct {
	// Allow indicating whether the user is allowed or not
	Allow bool `json:"Allow"`

	// Msg stores the authorization message
	Msg string `json:"Msg,omitempty"`

	// Err stores a message in case there's an error
	Err string `json:"Err,omitempty"`
}
