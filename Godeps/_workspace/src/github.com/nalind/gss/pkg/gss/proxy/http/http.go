package http

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"

	"github.com/nalind/gss/pkg/gss/proxy"
)

type negotiateRoundTripper struct {
	proxySocket string
	rt          http.RoundTripper
}

func NewNegotiateRoundTripper(proxySocket string, rt http.RoundTripper) http.RoundTripper {
	return &negotiateRoundTripper{proxySocket, rt}
}

func (rt *negotiateRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	var proxyConn net.Conn
	var proxyCall proxy.CallCtx
	var cred proxy.Cred
	req = cloneRequest(req)

	resp, err := rt.rt.RoundTrip(req)
	if err != nil {
		return resp, err
	}

	// TODO: handle multiple WWW-Authenticate headers, or a single header containing multiple challenges

	// enter the negotiate flow
	if isInitialChallenge(resp) {
		var ctx proxy.SecCtx
		var iscr proxy.InitSecContextResults
		var hostname string

		// fmt.Println("RoundTrip: Started negotiate loop")

		if req.Host != "" {
			hostname = req.Host
		} else {
			hostname = req.URL.Host
		}
		name := proxy.Name{DisplayName: "HTTP@" + hostname, NameType: proxy.NT_HOSTBASED_SERVICE}

		// TODO: guard against infinite loops, set some max

		// Loop as long as we get back negotiate challenges, or we don't think we've completed the auth
		for i := 0; isNegotiateResponse(resp) || iscr.Status.MajorStatus != proxy.S_COMPLETE; i++ {
			// fmt.Printf("RoundTrip: Got Status=%v, WWW-Authenticate=%#v\n", resp.StatusCode, resp.Header.Get("WWW-Authenticate"))

			// require incoming token for continued responses
			var incomingToken []byte
			var incomingTokenPtr *[]byte
			if i > 0 {
				incomingToken, err = gssapiData(resp)
				if err != nil {
					return nil, err
				}
				if len(incomingToken) == 0 {
					return nil, errors.New("Continued server response was missing gssapi-data")
				}
				incomingTokenPtr = &incomingToken
			} else {
				proxyConn, err = net.Dial("unix", rt.proxySocket)
				if err != nil {
					return nil, err
				}
				defer proxyConn.Close()
				gcr, err := proxy.GetCallContext(&proxyConn, &proxyCall, nil)
				if err != nil {
					return nil, err
				}
				if gcr.Status.MajorStatus != proxy.S_COMPLETE {
					return nil, errors.New("Error getting gss-proxy call context")
				}
				acr, err := proxy.AcquireCred(&proxyConn, &proxyCall, nil, false, nil, proxy.C_INDEFINITE, nil, proxy.C_INITIATE, proxy.C_INDEFINITE, 0, nil)
				if err != nil {
					return nil, err
				}
				if acr.Status.MajorStatus != proxy.S_COMPLETE {
					return nil, errors.New("Error getting gss-proxy creds")
				}
				cred = *acr.OutputCredHandle
				if cred.NeedsRelease {
					defer proxy.ReleaseCred(&proxyConn, &proxyCall, &cred)
				}
			}

			// call gss_init_sec_context to validate the incoming token (if given), and get our outgoing token (if needed)
			iscr, err = proxy.InitSecContext(&proxyConn, &proxyCall, &ctx, &cred, &name, proxy.MechSPNEGO, proxy.Flags{Mutual: true}, proxy.C_INDEFINITE, nil, incomingTokenPtr, nil)
			if iscr.Status.MajorStatus != proxy.S_COMPLETE && iscr.Status.MajorStatus != proxy.S_CONTINUE_NEEDED {
				if iscr.Status.MinorStatusString != "" {
					return nil, errors.New(fmt.Sprintf("%s while initializing security context (%s) (step %d)", iscr.Status.MajorStatusString, iscr.Status.MinorStatusString, i+1))
				} else {
					return nil, errors.New(fmt.Sprintf("%s while initializing security context (step %d)", iscr.Status.MajorStatusString, i+1))
				}
			}

			// fmt.Printf("Complete: %v, Continue: %v\n", major == proxy.S_COMPLETE, major == proxy.S_CONTINUE_NEEDED)

			// the remote server is unhappy, or we don't think we've finished the auth
			if resp.StatusCode == http.StatusUnauthorized || iscr.Status.MajorStatus == proxy.S_CONTINUE_NEEDED {
				// retry the request with our new token, and restart the loop
				outgoingTokenBase64 := base64.StdEncoding.EncodeToString(*iscr.OutputToken)
				// fmt.Println("Re-sending request with Authorization token")
				req.Header.Set("Authorization", "Negotiate "+outgoingTokenBase64)
				resp, err = rt.rt.RoundTrip(req)
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
