package protocol

import (
	"strings"

	"github.com/aws/aws-sdk-go/aws/request"
)

// ValidateEndpointHostHandler is a request handler that will validate the
// request endpoint's hosts is a valid RFC 3986 host.
var ValidateEndpointHostHandler = request.NamedHandler{
	Name: "awssdk.protocol.ValidateEndpointHostHandler",
	Fn: func(r *request.Request) {
		err := ValidateEndpointHost(r.Operation.Name, r.HTTPRequest.URL.Host)
		if err != nil {
			r.Error = err
		}
	},
}

// ValidateEndpointHost validates that the host string passed in is a valid RFC
// 3986 host. Returns error if the host is not valid.
func ValidateEndpointHost(opName, host string) error {
	paramErrs := request.ErrInvalidParams{Context: opName}
	labels := strings.Split(host, ".")

	for i, label := range labels {
		if i == len(labels)-1 && len(label) == 0 {
			// Allow trailing dot for FQDN hosts.
			continue
		}

		if !ValidHostLabel(label) {
			paramErrs.Add(request.NewErrParamFormat(
				"endpoint host label", "[a-zA-Z0-9-]{1,63}", label))
		}
	}

	if len(host) > 255 {
		paramErrs.Add(request.NewErrParamMaxLen(
			"endpoint host", 255, host,
		))
	}

	if paramErrs.Len() > 0 {
		return paramErrs
	}
	return nil
}

// ValidHostLabel returns if the label is a valid RFC 3986 host label.
func ValidHostLabel(label string) bool {
	if l := len(label); l == 0 || l > 63 {
		return false
	}
	for _, r := range label {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'A' && r <= 'Z':
		case r >= 'a' && r <= 'z':
		case r == '-':
		default:
			return false
		}
	}

	return true
}
