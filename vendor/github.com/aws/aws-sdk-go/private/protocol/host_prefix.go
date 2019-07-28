package protocol

import (
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
)

// HostPrefixHandlerName is the handler name for the host prefix request
// handler.
const HostPrefixHandlerName = "awssdk.endpoint.HostPrefixHandler"

// NewHostPrefixHandler constructs a build handler
func NewHostPrefixHandler(prefix string, labelsFn func() map[string]string) request.NamedHandler {
	builder := HostPrefixBuilder{
		Prefix:   prefix,
		LabelsFn: labelsFn,
	}

	return request.NamedHandler{
		Name: HostPrefixHandlerName,
		Fn:   builder.Build,
	}
}

// HostPrefixBuilder provides the request handler to expand and prepend
// the host prefix into the operation's request endpoint host.
type HostPrefixBuilder struct {
	Prefix   string
	LabelsFn func() map[string]string
}

// Build updates the passed in Request with the HostPrefix template expanded.
func (h HostPrefixBuilder) Build(r *request.Request) {
	if aws.BoolValue(r.Config.DisableEndpointHostPrefix) {
		return
	}

	var labels map[string]string
	if h.LabelsFn != nil {
		labels = h.LabelsFn()
	}

	prefix := h.Prefix
	for name, value := range labels {
		prefix = strings.Replace(prefix, "{"+name+"}", value, -1)
	}

	r.HTTPRequest.URL.Host = prefix + r.HTTPRequest.URL.Host
	if len(r.HTTPRequest.Host) > 0 {
		r.HTTPRequest.Host = prefix + r.HTTPRequest.Host
	}
}
