package fixchain

import (
	"bytes"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"strconv"

	"github.com/google/certificate-transparency/go/x509"
)

type errorType int

// FixError types
const (
	None errorType = iota
	ParseFailure
	CannotFetchURL
	FixFailed
	PostFailed    // Call to http.Client.PostChainToLog() failed
	LogPostFailed // http.Response from Certificate Transparency log contained a status code other than 200
	VerifyFailed
)

// FixError is the struct with which errors in the fixing process are reported
type FixError struct {
	Type  errorType
	Cert  *x509.Certificate   // The supplied leaf certificate
	Chain []*x509.Certificate // The supplied chain
	URL   string              // URL, if a URL is involved
	Bad   []byte              // The offending certificate bytes, if applicable
	Error error               // The error
	Code  int                 // The http response code from the Certificate Transparency log (only for use with LogPostFailed)
}

// Equal tests whether this FixError is equal to another given FixError
func (e FixError) Equal(f *FixError) bool {
	if f == nil || e.Type != f.Type || e.URL != f.URL || !bytes.Equal(e.Bad, f.Bad) || e.Code != f.Code {
		return false
	}
	// Check Cert equality
	if e.Cert != nil {
		if f.Cert == nil || !e.Cert.Equal(f.Cert) {
			return false
		}
	} else if f.Cert != nil {
		return false
	}
	// Check Chain equality
	if len(e.Chain) != len(f.Chain) {
		return false
	}
	for i := range e.Chain {
		if !e.Chain[i].Equal(f.Chain[i]) {
			return false
		}
	}
	// Check Error equality
	if e.Error != nil {
		if f.Error == nil || e.Error.Error() != f.Error.Error() {
			return false
		}
	} else if f.Error != nil {
		return false
	}

	return true
}

// TypeString returns a string describing e.Type
func (e FixError) TypeString() string {
	switch e.Type {
	case None:
		return "None"
	case ParseFailure:
		return "ParseFailure"
	case CannotFetchURL:
		return "CannotFetchURL"
	case FixFailed:
		return "FixFailed"
	case PostFailed:
		return "PostFailed"
	case LogPostFailed:
		return "LogPostFailed"
	case VerifyFailed:
		return "VerifyFailed"
	default:
		return fmt.Sprintf("Type %d", e.Type)
	}
}

// String converts an error to a (mostly) human readable string
func (e FixError) String() string {
	s := e.TypeString() + "\n"
	if e.Code != 0 {
		s += "Status Code: " + strconv.Itoa(e.Code) + "\n"
	}
	if e.Error != nil {
		s += "Error: " + e.Error.Error() + "\n"
	}
	if e.URL != "" {
		s += "URL: " + e.URL + "\n"
	}
	if e.Bad != nil {
		s += "Bad: " + dumpPEM(e.Bad)
	}
	if e.Cert != nil {
		s += "Cert: " + dumpPEM(e.Cert.Raw)
	}
	if e.Chain != nil {
		s += "Chain: " + dumpChainPEM(e.Chain)
	}
	return s
}

// MarshalJSON converts a FixError to JSON
func (e FixError) MarshalJSON() ([]byte, error) {
	var m struct {
		Type  string
		Cert  []byte
		Chain [][]byte
		URL   string
		Bad   []byte
		Error string
		Code  int
	}
	m.Type = e.TypeString()
	if e.Cert != nil {
		m.Cert = e.Cert.Raw
	}
	for _, c := range e.Chain {
		m.Chain = append(m.Chain, c.Raw)
	}
	m.URL = e.URL
	m.Bad = e.Bad
	if e.Error != nil {
		m.Error = e.Error.Error()
	}
	m.Code = e.Code

	return json.Marshal(m)
}

// UnmarshalJSON converts the JSON representation of a FixError back to a FixError
func UnmarshalJSON(b []byte) (*FixError, error) {
	var u struct {
		Type  string
		Cert  []byte
		Chain [][]byte
		URL   string
		Bad   []byte
		Error string
		Code  int
	}
	err := json.Unmarshal(b, &u)
	if err != nil {
		return nil, err
	}

	ferr := &FixError{}
	switch u.Type {
	case "None":
		ferr.Type = None
	case "ParseFailure":
		ferr.Type = ParseFailure
	case "CannotFetchURL":
		ferr.Type = CannotFetchURL
	case "FixFailed":
		ferr.Type = FixFailed
	case "PostFailed":
		ferr.Type = PostFailed
	case "LogPostFailed":
		ferr.Type = LogPostFailed
	case "VerifyFailed":
		ferr.Type = VerifyFailed
	default:
		return nil, errors.New("cannot parse FixError Type")
	}

	if u.Cert != nil {
		cert, err := x509.ParseCertificate(u.Cert)
		if err != nil {
			return nil, fmt.Errorf("cannot parse FixError Cert: %s", err)
		}
		ferr.Cert = cert
	}

	for _, c := range u.Chain {
		cert, err := x509.ParseCertificate(c)
		if err != nil {
			return nil, fmt.Errorf("cannot parse FixError Chain: %s", err)
		}
		ferr.Chain = append(ferr.Chain, cert)
	}

	ferr.URL = u.URL
	ferr.Bad = u.Bad
	if u.Error != "" {
		ferr.Error = errors.New(u.Error)
	}
	ferr.Code = u.Code

	return ferr, nil
}

func dumpChainPEM(chain []*x509.Certificate) string {
	var p string
	for _, cert := range chain {
		p += dumpPEM(cert.Raw)
	}
	return p
}

func dumpPEM(cert []byte) string {
	b := pem.Block{Type: "CERTIFICATE", Bytes: cert}
	return string(pem.EncodeToMemory(&b))
}
