package storage

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

var (
	fixedTime = time.Date(2050, time.December, 20, 21, 55, 0, 0, time.FixedZone("GMT", -6))
)

func (c Client) computeHmac256(message string) string {
	h := hmac.New(sha256.New, c.accountKey)
	h.Write([]byte(message))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func currentTimeRfc1123Formatted() string {
	return timeRfc1123Formatted(time.Now().UTC())
}

func timeRfc1123Formatted(t time.Time) string {
	return t.Format(http.TimeFormat)
}

func mergeParams(v1, v2 url.Values) url.Values {
	out := url.Values{}
	for k, v := range v1 {
		out[k] = v
	}
	for k, v := range v2 {
		vals, ok := out[k]
		if ok {
			vals = append(vals, v...)
			out[k] = vals
		} else {
			out[k] = v
		}
	}
	return out
}

func prepareBlockListRequest(blocks []Block) string {
	s := `<?xml version="1.0" encoding="utf-8"?><BlockList>`
	for _, v := range blocks {
		s += fmt.Sprintf("<%s>%s</%s>", v.Status, v.ID, v.Status)
	}
	s += `</BlockList>`
	return s
}

func xmlUnmarshal(body io.Reader, v interface{}) error {
	data, err := ioutil.ReadAll(body)
	if err != nil {
		return err
	}
	return xml.Unmarshal(data, v)
}

func xmlMarshal(v interface{}) (io.Reader, int, error) {
	b, err := xml.Marshal(v)
	if err != nil {
		return nil, 0, err
	}
	return bytes.NewReader(b), len(b), nil
}

func headersFromStruct(v interface{}) map[string]string {
	headers := make(map[string]string)
	value := reflect.ValueOf(v)
	for i := 0; i < value.NumField(); i++ {
		key := value.Type().Field(i).Tag.Get("header")
		if key != "" {
			reflectedValue := reflect.Indirect(value.Field(i))
			var val string
			if reflectedValue.IsValid() {
				switch reflectedValue.Type() {
				case reflect.TypeOf(fixedTime):
					val = timeRfc1123Formatted(reflectedValue.Interface().(time.Time))
				case reflect.TypeOf(uint64(0)), reflect.TypeOf(uint(0)):
					val = strconv.FormatUint(reflectedValue.Uint(), 10)
				case reflect.TypeOf(int(0)):
					val = strconv.FormatInt(reflectedValue.Int(), 10)
				default:
					val = reflectedValue.String()
				}
			}
			if val != "" {
				headers[key] = val
			}
		}
	}
	return headers
}

// merges extraHeaders into headers and returns headers
func mergeHeaders(headers, extraHeaders map[string]string) map[string]string {
	for k, v := range extraHeaders {
		headers[k] = v
	}
	return headers
}

func addToHeaders(h map[string]string, key, value string) map[string]string {
	if value != "" {
		h[key] = value
	}
	return h
}

func addTimeToHeaders(h map[string]string, key string, value *time.Time) map[string]string {
	if value != nil {
		h = addToHeaders(h, key, timeRfc1123Formatted(*value))
	}
	return h
}

func addTimeout(params url.Values, timeout uint) url.Values {
	if timeout > 0 {
		params.Add("timeout", fmt.Sprintf("%v", timeout))
	}
	return params
}

func addSnapshot(params url.Values, snapshot *time.Time) url.Values {
	if snapshot != nil {
		params.Add("snapshot", snapshot.Format("2006-01-02T15:04:05.0000000Z"))
	}
	return params
}

func getTimeFromHeaders(h http.Header, key string) (*time.Time, error) {
	var out time.Time
	var err error
	outStr := h.Get(key)
	if outStr != "" {
		out, err = time.Parse(time.RFC1123, outStr)
		if err != nil {
			return nil, err
		}
	}
	return &out, nil
}

// TimeRFC1123 is an alias for time.Time needed for custom Unmarshalling
type TimeRFC1123 time.Time

// UnmarshalXML is a custom unmarshaller that overrides the default time unmarshal which uses a different time layout.
func (t *TimeRFC1123) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var value string
	d.DecodeElement(&value, &start)
	parse, err := time.Parse(time.RFC1123, value)
	if err != nil {
		return err
	}
	*t = TimeRFC1123(parse)
	return nil
}

// returns a map of custom metadata values from the specified HTTP header
func getMetadataFromHeaders(header http.Header) map[string]string {
	metadata := make(map[string]string)
	for k, v := range header {
		// Can't trust CanonicalHeaderKey() to munge case
		// reliably. "_" is allowed in identifiers:
		// https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
		// https://msdn.microsoft.com/library/aa664670(VS.71).aspx
		// http://tools.ietf.org/html/rfc7230#section-3.2
		// ...but "_" is considered invalid by
		// CanonicalMIMEHeaderKey in
		// https://golang.org/src/net/textproto/reader.go?s=14615:14659#L542
		// so k can be "X-Ms-Meta-Lol" or "x-ms-meta-lol_rofl".
		k = strings.ToLower(k)
		if len(v) == 0 || !strings.HasPrefix(k, strings.ToLower(userDefinedMetadataHeaderPrefix)) {
			continue
		}
		// metadata["lol"] = content of the last X-Ms-Meta-Lol header
		k = k[len(userDefinedMetadataHeaderPrefix):]
		metadata[k] = v[len(v)-1]
	}

	if len(metadata) == 0 {
		return nil
	}

	return metadata
}
