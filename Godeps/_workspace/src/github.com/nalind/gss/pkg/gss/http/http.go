package http

import (
	"encoding/asn1"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/nalind/gss/pkg/gss"
)

type NegotiateRoundTripper struct {
	Transport http.RoundTripper
	Flags     gss.Flags
	Mech      asn1.ObjectIdentifier
}

func NewNegotiateRoundTripper(rt http.RoundTripper) http.RoundTripper {
	return &NegotiateRoundTripper{Transport: rt}
}

func (rt *NegotiateRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = cloneRequest(req)

	resp, err := rt.Transport.RoundTrip(req)
	if err != nil {
		return resp, err
	}

	// TODO: handle multiple WWW-Authenticate headers, or a single header containing multiple challenges

	// enter the negotiate flow
	if isInitialChallenge(resp) {
		// fmt.Println("RoundTrip: Started negotiate loop")

		name, err := importName(req.Host)
		if err != nil {
			return nil, err
		}
		defer gss.ReleaseName(name)

		var ctx gss.ContextHandle
		defer gss.DeleteSecContext(ctx)

		var major, minor uint32

		// Local copy of flags
		flags := rt.Flags

		// TODO: guard against infinite loops, set some max

		// Loop as long as we get back negotiate challenges, or we don't think we've completed the auth
		for i := 0; isNegotiateResponse(resp) || major != gss.S_COMPLETE; i++ {
			// fmt.Printf("RoundTrip: Got Status=%v, WWW-Authenticate=%#v\n", resp.StatusCode, resp.Header.Get("WWW-Authenticate"))

			// require incoming token for continued responses
			var incomingToken []byte
			if i > 0 {
				incomingToken, err = gssapiData(resp)
				if err != nil {
					return nil, err
				}
				if len(incomingToken) == 0 {
					return nil, errors.New("Continued server response was missing gssapi-data")
				}
			}

			// call gss_init_sec_context to validate the incoming token (if given), and get our outgoing token (if needed)
			var outgoingToken []byte
			major, minor, _, outgoingToken, flags, _, _, _ = gss.InitSecContext(nil, &ctx, name, rt.Mech, flags, gss.C_INDEFINITE, nil, incomingToken)
			if major != gss.S_COMPLETE && major != gss.S_CONTINUE_NEEDED {
				return nil, gss.NewGSSError(fmt.Sprintf("initializing security context (step %d)", i+1), major, minor, &rt.Mech)
			}

			// fmt.Printf("Complete: %v, Continue: %v\n", major == gss.S_COMPLETE, major == gss.S_CONTINUE_NEEDED)

			// the remote server is unhappy, or we don't think we've finished the auth
			if resp.StatusCode == http.StatusUnauthorized || major == gss.S_CONTINUE_NEEDED {
				// retry the request with our new token, and restart the loop
				outgoingTokenBase64 := base64.StdEncoding.EncodeToString(outgoingToken)
				// fmt.Println("Re-sending request with Authorization token")
				req.Header.Set("Authorization", "Negotiate "+outgoingTokenBase64)
				resp, err = rt.Transport.RoundTrip(req)
				if err != nil {
					return nil, err
				}
			} else {
				return resp, nil
			}
		}
	}
	return resp, nil
}

func isInitialChallenge(resp *http.Response) bool {
	return resp.StatusCode == http.StatusUnauthorized && resp.Header.Get("WWW-Authenticate") == "Negotiate"
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}

// isNegotiateResponse returns true if the response contains a WWW-Authenticate header with a Negotiate challenge
func isNegotiateResponse(resp *http.Response) bool {
	return resp.Header.Get("WWW-Authenticate") == "Negotiate" || strings.HasPrefix(resp.Header.Get("WWW-Authenticate"), "Negotiate ")
}

// gssapiData returns base64-decoded gssapi-data in any Negotiate challenge header
// An empty string is returned if no Negotiate challenge header is present, or if no
// gssapi-data is present. An error is returned if malformed gssapi-data is present.
func gssapiData(resp *http.Response) ([]byte, error) {
	authHeader := resp.Header.Get("WWW-Authenticate")

	parts := strings.SplitN(authHeader, " ", 2)
	if len(parts) < 2 || parts[0] != "Negotiate" {
		return nil, nil
	}

	// Remove whitespace
	gssapiData := strings.Replace(parts[1], " ", "", -1)

	// Decode
	decodedData, err := base64.StdEncoding.DecodeString(gssapiData)
	if err != nil {
		return nil, err
	}

	return decodedData, nil
}

// importName returns a gss.InternalName for a given hostname, or an error.
// The caller is responsible for releasing the returned name using gss.ReleaseName.
func importName(hostname string) (gss.InternalName, error) {
	major, minor, name := gss.ImportName("HTTP@"+hostname, gss.C_NT_HOSTBASED_SERVICE)
	if major != gss.S_COMPLETE {
		return nil, gss.NewGSSError("importing remote service name", major, minor, nil)
	}
	return name, nil
}
