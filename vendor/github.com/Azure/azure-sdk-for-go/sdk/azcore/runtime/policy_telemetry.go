//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"bytes"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
)

type telemetryPolicy struct {
	telemetryValue string
}

// NewTelemetryPolicy creates a telemetry policy object that adds telemetry information to outgoing HTTP requests.
// The format is [<application_id> ]azsdk-go-<mod>/<ver> <platform_info>.
// Pass nil to accept the default values; this is the same as passing a zero-value options.
func NewTelemetryPolicy(mod, ver string, o *policy.TelemetryOptions) policy.Policy {
	if o == nil {
		o = &policy.TelemetryOptions{}
	}
	tp := telemetryPolicy{}
	if o.Disabled {
		return &tp
	}
	b := &bytes.Buffer{}
	// normalize ApplicationID
	if o.ApplicationID != "" {
		o.ApplicationID = strings.ReplaceAll(o.ApplicationID, " ", "/")
		if len(o.ApplicationID) > 24 {
			o.ApplicationID = o.ApplicationID[:24]
		}
		b.WriteString(o.ApplicationID)
		b.WriteRune(' ')
	}
	b.WriteString(formatTelemetry(mod, ver))
	b.WriteRune(' ')
	b.WriteString(platformInfo)
	tp.telemetryValue = b.String()
	return &tp
}

func formatTelemetry(comp, ver string) string {
	return fmt.Sprintf("azsdk-go-%s/%s", comp, ver)
}

func (p telemetryPolicy) Do(req *policy.Request) (*http.Response, error) {
	if p.telemetryValue == "" {
		return req.Next()
	}
	// preserve the existing User-Agent string
	if ua := req.Raw().Header.Get(shared.HeaderUserAgent); ua != "" {
		p.telemetryValue = fmt.Sprintf("%s %s", p.telemetryValue, ua)
	}
	req.Raw().Header.Set(shared.HeaderUserAgent, p.telemetryValue)
	return req.Next()
}

// NOTE: the ONLY function that should write to this variable is this func
var platformInfo = func() string {
	operatingSystem := runtime.GOOS // Default OS string
	switch operatingSystem {
	case "windows":
		operatingSystem = os.Getenv("OS") // Get more specific OS information
	case "linux": // accept default OS info
	case "freebsd": //  accept default OS info
	}
	return fmt.Sprintf("(%s; %s)", runtime.Version(), operatingSystem)
}()
