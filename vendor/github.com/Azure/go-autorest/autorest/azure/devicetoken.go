package azure

/*
  This file is largely based on rjw57/oauth2device's code, with the follow differences:
   * scope -> resource, and only allow a single one
   * receive "Message" in the DeviceCode struct and show it to users as the prompt
   * azure-xplat-cli has the following behavior that this emulates:
     - does not send client_secret during the token exchange
     - sends resource again in the token exchange request
*/

import (
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/Azure/go-autorest/autorest"
)

const (
	logPrefix = "autorest/azure/devicetoken:"
)

var (
	// ErrDeviceGeneric represents an unknown error from the token endpoint when using device flow
	ErrDeviceGeneric = fmt.Errorf("%s Error while retrieving OAuth token: Unknown Error", logPrefix)

	// ErrDeviceAccessDenied represents an access denied error from the token endpoint when using device flow
	ErrDeviceAccessDenied = fmt.Errorf("%s Error while retrieving OAuth token: Access Denied", logPrefix)

	// ErrDeviceAuthorizationPending represents the server waiting on the user to complete the device flow
	ErrDeviceAuthorizationPending = fmt.Errorf("%s Error while retrieving OAuth token: Authorization Pending", logPrefix)

	// ErrDeviceCodeExpired represents the server timing out and expiring the code during device flow
	ErrDeviceCodeExpired = fmt.Errorf("%s Error while retrieving OAuth token: Code Expired", logPrefix)

	// ErrDeviceSlowDown represents the service telling us we're polling too often during device flow
	ErrDeviceSlowDown = fmt.Errorf("%s Error while retrieving OAuth token: Slow Down", logPrefix)

	errCodeSendingFails   = "Error occurred while sending request for Device Authorization Code"
	errCodeHandlingFails  = "Error occurred while handling response from the Device Endpoint"
	errTokenSendingFails  = "Error occurred while sending request with device code for a token"
	errTokenHandlingFails = "Error occurred while handling response from the Token Endpoint (during device flow)"
)

// DeviceCode is the object returned by the device auth endpoint
// It contains information to instruct the user to complete the auth flow
type DeviceCode struct {
	DeviceCode      *string `json:"device_code,omitempty"`
	UserCode        *string `json:"user_code,omitempty"`
	VerificationURL *string `json:"verification_url,omitempty"`
	ExpiresIn       *int64  `json:"expires_in,string,omitempty"`
	Interval        *int64  `json:"interval,string,omitempty"`

	Message     *string `json:"message"` // Azure specific
	Resource    string  // store the following, stored when initiating, used when exchanging
	OAuthConfig OAuthConfig
	ClientID    string
}

// TokenError is the object returned by the token exchange endpoint
// when something is amiss
type TokenError struct {
	Error            *string `json:"error,omitempty"`
	ErrorCodes       []int   `json:"error_codes,omitempty"`
	ErrorDescription *string `json:"error_description,omitempty"`
	Timestamp        *string `json:"timestamp,omitempty"`
	TraceID          *string `json:"trace_id,omitempty"`
}

// DeviceToken is the object return by the token exchange endpoint
// It can either look like a Token or an ErrorToken, so put both here
// and check for presence of "Error" to know if we are in error state
type deviceToken struct {
	Token
	TokenError
}

// InitiateDeviceAuth initiates a device auth flow. It returns a DeviceCode
// that can be used with CheckForUserCompletion or WaitForUserCompletion.
func InitiateDeviceAuth(client *autorest.Client, oauthConfig OAuthConfig, clientID, resource string) (*DeviceCode, error) {
	req, _ := autorest.Prepare(
		&http.Request{},
		autorest.AsPost(),
		autorest.AsFormURLEncoded(),
		autorest.WithBaseURL(oauthConfig.DeviceCodeEndpoint.String()),
		autorest.WithFormData(url.Values{
			"client_id": []string{clientID},
			"resource":  []string{resource},
		}),
	)

	resp, err := autorest.SendWithSender(client, req)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeSendingFails, err)
	}

	var code DeviceCode
	err = autorest.Respond(
		resp,
		autorest.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&code),
		autorest.ByClosing())
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeHandlingFails, err)
	}

	code.ClientID = clientID
	code.Resource = resource
	code.OAuthConfig = oauthConfig

	return &code, nil
}

// CheckForUserCompletion takes a DeviceCode and checks with the Azure AD OAuth endpoint
// to see if the device flow has: been completed, timed out, or otherwise failed
func CheckForUserCompletion(client *autorest.Client, code *DeviceCode) (*Token, error) {
	req, _ := autorest.Prepare(
		&http.Request{},
		autorest.AsPost(),
		autorest.AsFormURLEncoded(),
		autorest.WithBaseURL(code.OAuthConfig.TokenEndpoint.String()),
		autorest.WithFormData(url.Values{
			"client_id":  []string{code.ClientID},
			"code":       []string{*code.DeviceCode},
			"grant_type": []string{OAuthGrantTypeDeviceCode},
			"resource":   []string{code.Resource},
		}),
	)

	resp, err := autorest.SendWithSender(client, req)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenSendingFails, err)
	}

	var token deviceToken
	err = autorest.Respond(
		resp,
		autorest.WithErrorUnlessStatusCode(http.StatusOK, http.StatusBadRequest),
		autorest.ByUnmarshallingJSON(&token),
		autorest.ByClosing())
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenHandlingFails, err)
	}

	if token.Error == nil {
		return &token.Token, nil
	}

	switch *token.Error {
	case "authorization_pending":
		return nil, ErrDeviceAuthorizationPending
	case "slow_down":
		return nil, ErrDeviceSlowDown
	case "access_denied":
		return nil, ErrDeviceAccessDenied
	case "code_expired":
		return nil, ErrDeviceCodeExpired
	default:
		return nil, ErrDeviceGeneric
	}
}

// WaitForUserCompletion calls CheckForUserCompletion repeatedly until a token is granted or an error state occurs.
// This prevents the user from looping and checking against 'ErrDeviceAuthorizationPending'.
func WaitForUserCompletion(client *autorest.Client, code *DeviceCode) (*Token, error) {
	intervalDuration := time.Duration(*code.Interval) * time.Second
	waitDuration := intervalDuration

	for {
		token, err := CheckForUserCompletion(client, code)

		if err == nil {
			return token, nil
		}

		switch err {
		case ErrDeviceSlowDown:
			waitDuration += waitDuration
		case ErrDeviceAuthorizationPending:
			// noop
		default: // everything else is "fatal" to us
			return nil, err
		}

		if waitDuration > (intervalDuration * 3) {
			return nil, fmt.Errorf("%s Error waiting for user to complete device flow. Server told us to slow_down too much", logPrefix)
		}

		time.Sleep(waitDuration)
	}
}
