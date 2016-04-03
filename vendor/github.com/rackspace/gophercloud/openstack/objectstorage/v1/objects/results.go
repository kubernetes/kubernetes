package objects

import (
	"fmt"
	"io"
	"io/ioutil"
	"strconv"
	"strings"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Object is a structure that holds information related to a storage object.
type Object struct {
	// Bytes is the total number of bytes that comprise the object.
	Bytes int64 `json:"bytes" mapstructure:"bytes"`

	// ContentType is the content type of the object.
	ContentType string `json:"content_type" mapstructure:"content_type"`

	// Hash represents the MD5 checksum value of the object's content.
	Hash string `json:"hash" mapstructure:"hash"`

	// LastModified is the RFC3339Milli time the object was last modified, represented
	// as a string. For any given object (obj), this value may be parsed to a time.Time:
	// lastModified, err := time.Parse(gophercloud.RFC3339Milli, obj.LastModified)
	LastModified string `json:"last_modified" mapstructure:"last_modified"`

	// Name is the unique name for the object.
	Name string `json:"name" mapstructure:"name"`
}

// ObjectPage is a single page of objects that is returned from a call to the
// List function.
type ObjectPage struct {
	pagination.MarkerPageBase
}

// IsEmpty returns true if a ListResult contains no object names.
func (r ObjectPage) IsEmpty() (bool, error) {
	names, err := ExtractNames(r)
	if err != nil {
		return true, err
	}
	return len(names) == 0, nil
}

// LastMarker returns the last object name in a ListResult.
func (r ObjectPage) LastMarker() (string, error) {
	names, err := ExtractNames(r)
	if err != nil {
		return "", err
	}
	if len(names) == 0 {
		return "", nil
	}
	return names[len(names)-1], nil
}

// ExtractInfo is a function that takes a page of objects and returns their full information.
func ExtractInfo(page pagination.Page) ([]Object, error) {
	untyped := page.(ObjectPage).Body.([]interface{})
	results := make([]Object, len(untyped))
	for index, each := range untyped {
		object := each.(map[string]interface{})
		err := mapstructure.Decode(object, &results[index])
		if err != nil {
			return results, err
		}
	}
	return results, nil
}

// ExtractNames is a function that takes a page of objects and returns only their names.
func ExtractNames(page pagination.Page) ([]string, error) {
	casted := page.(ObjectPage)
	ct := casted.Header.Get("Content-Type")
	switch {
	case strings.HasPrefix(ct, "application/json"):
		parsed, err := ExtractInfo(page)
		if err != nil {
			return nil, err
		}

		names := make([]string, 0, len(parsed))
		for _, object := range parsed {
			names = append(names, object.Name)
		}

		return names, nil
	case strings.HasPrefix(ct, "text/plain"):
		names := make([]string, 0, 50)

		body := string(page.(ObjectPage).Body.([]uint8))
		for _, name := range strings.Split(body, "\n") {
			if len(name) > 0 {
				names = append(names, name)
			}
		}

		return names, nil
	case strings.HasPrefix(ct, "text/html"):
		return []string{}, nil
	default:
		return nil, fmt.Errorf("Cannot extract names from response with content-type: [%s]", ct)
	}
}

// DownloadHeader represents the headers returned in the response from a Download request.
type DownloadHeader struct {
	AcceptRanges       string    `mapstructure:"Accept-Ranges"`
	ContentDisposition string    `mapstructure:"Content-Disposition"`
	ContentEncoding    string    `mapstructure:"Content-Encoding"`
	ContentLength      int64     `mapstructure:"Content-Length"`
	ContentType        string    `mapstructure:"Content-Type"`
	Date               time.Time `mapstructure:"-"`
	DeleteAt           time.Time `mapstructure:"-"`
	ETag               string    `mapstructure:"Etag"`
	LastModified       time.Time `mapstructure:"-"`
	ObjectManifest     string    `mapstructure:"X-Object-Manifest"`
	StaticLargeObject  bool      `mapstructure:"X-Static-Large-Object"`
	TransID            string    `mapstructure:"X-Trans-Id"`
}

// DownloadResult is a *http.Response that is returned from a call to the Download function.
type DownloadResult struct {
	gophercloud.HeaderResult
	Body io.ReadCloser
}

// Extract will return a struct of headers returned from a call to Download. To obtain
// a map of headers, call the ExtractHeader method on the DownloadResult.
func (dr DownloadResult) Extract() (DownloadHeader, error) {
	var dh DownloadHeader
	if dr.Err != nil {
		return dh, dr.Err
	}

	if err := gophercloud.DecodeHeader(dr.Header, &dh); err != nil {
		return dh, err
	}

	if date, ok := dr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, date[0])
		if err != nil {
			return dh, err
		}
		dh.Date = t
	}

	if date, ok := dr.Header["Last-Modified"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, date[0])
		if err != nil {
			return dh, err
		}
		dh.LastModified = t
	}

	if date, ok := dr.Header["X-Delete-At"]; ok && len(date) > 0 {
		unix, err := strconv.ParseInt(date[0], 10, 64)
		if err != nil {
			return dh, err
		}
		dh.DeleteAt = time.Unix(unix, 0)
	}

	return dh, nil
}

// ExtractContent is a function that takes a DownloadResult's io.Reader body
// and reads all available data into a slice of bytes. Please be aware that due
// the nature of io.Reader is forward-only - meaning that it can only be read
// once and not rewound. You can recreate a reader from the output of this
// function by using bytes.NewReader(downloadBytes)
func (dr DownloadResult) ExtractContent() ([]byte, error) {
	if dr.Err != nil {
		return nil, dr.Err
	}
	body, err := ioutil.ReadAll(dr.Body)
	if err != nil {
		return nil, err
	}
	dr.Body.Close()
	return body, nil
}

// GetHeader represents the headers returned in the response from a Get request.
type GetHeader struct {
	ContentDisposition string    `mapstructure:"Content-Disposition"`
	ContentEncoding    string    `mapstructure:"Content-Encoding"`
	ContentLength      int64     `mapstructure:"Content-Length"`
	ContentType        string    `mapstructure:"Content-Type"`
	Date               time.Time `mapstructure:"-"`
	DeleteAt           time.Time `mapstructure:"-"`
	ETag               string    `mapstructure:"Etag"`
	LastModified       time.Time `mapstructure:"-"`
	ObjectManifest     string    `mapstructure:"X-Object-Manifest"`
	StaticLargeObject  bool      `mapstructure:"X-Static-Large-Object"`
	TransID            string    `mapstructure:"X-Trans-Id"`
}

// GetResult is a *http.Response that is returned from a call to the Get function.
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

	if date, ok := gr.Header["Last-Modified"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, gr.Header["Last-Modified"][0])
		if err != nil {
			return gh, err
		}
		gh.LastModified = t
	}

	if date, ok := gr.Header["X-Delete-At"]; ok && len(date) > 0 {
		unix, err := strconv.ParseInt(date[0], 10, 64)
		if err != nil {
			return gh, err
		}
		gh.DeleteAt = time.Unix(unix, 0)
	}

	return gh, nil
}

// ExtractMetadata is a function that takes a GetResult (of type *http.Response)
// and returns the custom metadata associated with the object.
func (gr GetResult) ExtractMetadata() (map[string]string, error) {
	if gr.Err != nil {
		return nil, gr.Err
	}
	metadata := make(map[string]string)
	for k, v := range gr.Header {
		if strings.HasPrefix(k, "X-Object-Meta-") {
			key := strings.TrimPrefix(k, "X-Object-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}

// CreateHeader represents the headers returned in the response from a Create request.
type CreateHeader struct {
	ContentLength int64     `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	ETag          string    `mapstructure:"Etag"`
	LastModified  time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Create. To obtain
// a map of headers, call the ExtractHeader method on the CreateResult.
func (cr CreateResult) Extract() (CreateHeader, error) {
	var ch CreateHeader
	if cr.Err != nil {
		return ch, cr.Err
	}

	if err := gophercloud.DecodeHeader(cr.Header, &ch); err != nil {
		return ch, err
	}

	if date, ok := cr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, cr.Header["Date"][0])
		if err != nil {
			return ch, err
		}
		ch.Date = t
	}

	if date, ok := cr.Header["Last-Modified"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, cr.Header["Last-Modified"][0])
		if err != nil {
			return ch, err
		}
		ch.LastModified = t
	}

	return ch, nil
}

// UpdateHeader represents the headers returned in the response from a Update request.
type UpdateHeader struct {
	ContentLength int64     `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// UpdateResult represents the result of an update operation.
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

// DeleteHeader represents the headers returned in the response from a Delete request.
type DeleteHeader struct {
	ContentLength int64     `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Delete. To obtain
// a map of headers, call the ExtractHeader method on the DeleteResult.
func (dr DeleteResult) Extract() (DeleteHeader, error) {
	var dh DeleteHeader
	if dr.Err != nil {
		return dh, dr.Err
	}

	if err := gophercloud.DecodeHeader(dr.Header, &dh); err != nil {
		return dh, err
	}

	if date, ok := dr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, dr.Header["Date"][0])
		if err != nil {
			return dh, err
		}
		dh.Date = t
	}

	return dh, nil
}

// CopyHeader represents the headers returned in the response from a Copy request.
type CopyHeader struct {
	ContentLength          int64     `mapstructure:"Content-Length"`
	ContentType            string    `mapstructure:"Content-Type"`
	CopiedFrom             string    `mapstructure:"X-Copied-From"`
	CopiedFromLastModified time.Time `mapstructure:"-"`
	Date                   time.Time `mapstructure:"-"`
	ETag                   string    `mapstructure:"Etag"`
	LastModified           time.Time `mapstructure:"-"`
	TransID                string    `mapstructure:"X-Trans-Id"`
}

// CopyResult represents the result of a copy operation.
type CopyResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Copy. To obtain
// a map of headers, call the ExtractHeader method on the CopyResult.
func (cr CopyResult) Extract() (CopyHeader, error) {
	var ch CopyHeader
	if cr.Err != nil {
		return ch, cr.Err
	}

	if err := gophercloud.DecodeHeader(cr.Header, &ch); err != nil {
		return ch, err
	}

	if date, ok := cr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, cr.Header["Date"][0])
		if err != nil {
			return ch, err
		}
		ch.Date = t
	}

	if date, ok := cr.Header["Last-Modified"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, cr.Header["Last-Modified"][0])
		if err != nil {
			return ch, err
		}
		ch.LastModified = t
	}

	if date, ok := cr.Header["X-Copied-From-Last-Modified"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, cr.Header["X-Copied-From-Last-Modified"][0])
		if err != nil {
			return ch, err
		}
		ch.CopiedFromLastModified = t
	}

	return ch, nil
}
