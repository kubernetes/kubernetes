package cdncontainers

import (
	"strings"
	"time"

	"github.com/rackspace/gophercloud"
)

// EnableHeader represents the headers returned in the response from an Enable request.
type EnableHeader struct {
	CDNIosURI       string    `mapstructure:"X-Cdn-Ios-Uri"`
	CDNSslURI       string    `mapstructure:"X-Cdn-Ssl-Uri"`
	CDNStreamingURI string    `mapstructure:"X-Cdn-Streaming-Uri"`
	CDNUri          string    `mapstructure:"X-Cdn-Uri"`
	ContentLength   int       `mapstructure:"Content-Length"`
	ContentType     string    `mapstructure:"Content-Type"`
	Date            time.Time `mapstructure:"-"`
	TransID         string    `mapstructure:"X-Trans-Id"`
}

// EnableResult represents the result of an Enable operation.
type EnableResult struct {
	gophercloud.HeaderResult
}

// Extract will return extract an EnableHeader from the response to an Enable
// request. To obtain a map of headers, call the ExtractHeader method on the EnableResult.
func (er EnableResult) Extract() (EnableHeader, error) {
	var eh EnableHeader
	if er.Err != nil {
		return eh, er.Err
	}

	if err := gophercloud.DecodeHeader(er.Header, &eh); err != nil {
		return eh, err
	}

	if date, ok := er.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, er.Header["Date"][0])
		if err != nil {
			return eh, err
		}
		eh.Date = t
	}

	return eh, nil
}

// GetHeader represents the headers returned in the response from a Get request.
type GetHeader struct {
	CDNEnabled      bool      `mapstructure:"X-Cdn-Enabled"`
	CDNIosURI       string    `mapstructure:"X-Cdn-Ios-Uri"`
	CDNSslURI       string    `mapstructure:"X-Cdn-Ssl-Uri"`
	CDNStreamingURI string    `mapstructure:"X-Cdn-Streaming-Uri"`
	CDNUri          string    `mapstructure:"X-Cdn-Uri"`
	ContentLength   int       `mapstructure:"Content-Length"`
	ContentType     string    `mapstructure:"Content-Type"`
	Date            time.Time `mapstructure:"-"`
	LogRetention    bool      `mapstructure:"X-Log-Retention"`
	TransID         string    `mapstructure:"X-Trans-Id"`
	TTL             int       `mapstructure:"X-Ttl"`
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Get. To obtain
// a map of headers, call the ExtractHeader method on the GetResult.
func (gr GetResult) Extract() (GetHeader, error) {
	var gh GetHeader
	if gr.Err != nil {
		return gh, gr.Err
	}

	if err := gophercloud.DecodeHeader(gr.Header, &gh); err != nil {
		return gh, err
	}

	if date, ok := gr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, gr.Header["Date"][0])
		if err != nil {
			return gh, err
		}
		gh.Date = t
	}

	return gh, nil
}

// ExtractMetadata is a function that takes a GetResult (of type *http.Response)
// and returns the custom metadata associated with the container.
func (gr GetResult) ExtractMetadata() (map[string]string, error) {
	if gr.Err != nil {
		return nil, gr.Err
	}
	metadata := make(map[string]string)
	for k, v := range gr.Header {
		if strings.HasPrefix(k, "X-Container-Meta-") {
			key := strings.TrimPrefix(k, "X-Container-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}

// UpdateHeader represents the headers returned in the response from a Update request.
type UpdateHeader struct {
	CDNIosURI       string    `mapstructure:"X-Cdn-Ios-Uri"`
	CDNSslURI       string    `mapstructure:"X-Cdn-Ssl-Uri"`
	CDNStreamingURI string    `mapstructure:"X-Cdn-Streaming-Uri"`
	CDNUri          string    `mapstructure:"X-Cdn-Uri"`
	ContentLength   int       `mapstructure:"Content-Length"`
	ContentType     string    `mapstructure:"Content-Type"`
	Date            time.Time `mapstructure:"-"`
	TransID         string    `mapstructure:"X-Trans-Id"`
}

// UpdateResult represents the result of an update operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Update. To obtain
// a map of headers, call the ExtractHeader method on the UpdateResult.
func (ur UpdateResult) Extract() (UpdateHeader, error) {
	var uh UpdateHeader
	if ur.Err != nil {
		return uh, ur.Err
	}

	if err := gophercloud.DecodeHeader(ur.Header, &uh); err != nil {
		return uh, err
	}

	if date, ok := ur.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, ur.Header["Date"][0])
		if err != nil {
			return uh, err
		}
		uh.Date = t
	}

	return uh, nil
}
