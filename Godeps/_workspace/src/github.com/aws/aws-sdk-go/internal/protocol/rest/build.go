// Package rest provides RESTful serialization of AWS requests and responses.
package rest

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net/url"
	"path"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// RFC822 returns an RFC822 formatted timestamp for AWS protocols
const RFC822 = "Mon, 2 Jan 2006 15:04:05 GMT"

// Whether the byte value can be sent without escaping in AWS URLs
var noEscape [256]bool

func init() {
	for i := 0; i < len(noEscape); i++ {
		// AWS expects every character except these to be escaped
		noEscape[i] = (i >= 'A' && i <= 'Z') ||
			(i >= 'a' && i <= 'z') ||
			(i >= '0' && i <= '9') ||
			i == '-' ||
			i == '.' ||
			i == '_' ||
			i == '~'
	}
}

// Build builds the REST component of a service request.
func Build(r *request.Request) {
	if r.ParamsFilled() {
		v := reflect.ValueOf(r.Params).Elem()
		buildLocationElements(r, v)
		buildBody(r, v)
	}
}

func buildLocationElements(r *request.Request, v reflect.Value) {
	query := r.HTTPRequest.URL.Query()

	for i := 0; i < v.NumField(); i++ {
		m := v.Field(i)
		if n := v.Type().Field(i).Name; n[0:1] == strings.ToLower(n[0:1]) {
			continue
		}

		if m.IsValid() {
			field := v.Type().Field(i)
			name := field.Tag.Get("locationName")
			if name == "" {
				name = field.Name
			}
			if m.Kind() == reflect.Ptr {
				m = m.Elem()
			}
			if !m.IsValid() {
				continue
			}

			switch field.Tag.Get("location") {
			case "headers": // header maps
				buildHeaderMap(r, m, field.Tag.Get("locationName"))
			case "header":
				buildHeader(r, m, name)
			case "uri":
				buildURI(r, m, name)
			case "querystring":
				buildQueryString(r, m, name, query)
			}
		}
		if r.Error != nil {
			return
		}
	}

	r.HTTPRequest.URL.RawQuery = query.Encode()
	updatePath(r.HTTPRequest.URL, r.HTTPRequest.URL.Path)
}

func buildBody(r *request.Request, v reflect.Value) {
	if field, ok := v.Type().FieldByName("SDKShapeTraits"); ok {
		if payloadName := field.Tag.Get("payload"); payloadName != "" {
			pfield, _ := v.Type().FieldByName(payloadName)
			if ptag := pfield.Tag.Get("type"); ptag != "" && ptag != "structure" {
				payload := reflect.Indirect(v.FieldByName(payloadName))
				if payload.IsValid() && payload.Interface() != nil {
					switch reader := payload.Interface().(type) {
					case io.ReadSeeker:
						r.SetReaderBody(reader)
					case []byte:
						r.SetBufferBody(reader)
					case string:
						r.SetStringBody(reader)
					default:
						r.Error = awserr.New("SerializationError",
							"failed to encode REST request",
							fmt.Errorf("unknown payload type %s", payload.Type()))
					}
				}
			}
		}
	}
}

func buildHeader(r *request.Request, v reflect.Value, name string) {
	str, err := convertType(v)
	if err != nil {
		r.Error = awserr.New("SerializationError", "failed to encode REST request", err)
	} else if str != nil {
		r.HTTPRequest.Header.Add(name, *str)
	}
}

func buildHeaderMap(r *request.Request, v reflect.Value, prefix string) {
	for _, key := range v.MapKeys() {
		str, err := convertType(v.MapIndex(key))
		if err != nil {
			r.Error = awserr.New("SerializationError", "failed to encode REST request", err)
		} else if str != nil {
			r.HTTPRequest.Header.Add(prefix+key.String(), *str)
		}
	}
}

func buildURI(r *request.Request, v reflect.Value, name string) {
	value, err := convertType(v)
	if err != nil {
		r.Error = awserr.New("SerializationError", "failed to encode REST request", err)
	} else if value != nil {
		uri := r.HTTPRequest.URL.Path
		uri = strings.Replace(uri, "{"+name+"}", EscapePath(*value, true), -1)
		uri = strings.Replace(uri, "{"+name+"+}", EscapePath(*value, false), -1)
		r.HTTPRequest.URL.Path = uri
	}
}

func buildQueryString(r *request.Request, v reflect.Value, name string, query url.Values) {
	str, err := convertType(v)
	if err != nil {
		r.Error = awserr.New("SerializationError", "failed to encode REST request", err)
	} else if str != nil {
		query.Set(name, *str)
	}
}

func updatePath(url *url.URL, urlPath string) {
	scheme, query := url.Scheme, url.RawQuery

	hasSlash := strings.HasSuffix(urlPath, "/")

	// clean up path
	urlPath = path.Clean(urlPath)
	if hasSlash && !strings.HasSuffix(urlPath, "/") {
		urlPath += "/"
	}

	// get formatted URL minus scheme so we can build this into Opaque
	url.Scheme, url.Path, url.RawQuery = "", "", ""
	s := url.String()
	url.Scheme = scheme
	url.RawQuery = query

	// build opaque URI
	url.Opaque = s + urlPath
}

// EscapePath escapes part of a URL path in Amazon style
func EscapePath(path string, encodeSep bool) string {
	var buf bytes.Buffer
	for i := 0; i < len(path); i++ {
		c := path[i]
		if noEscape[c] || (c == '/' && !encodeSep) {
			buf.WriteByte(c)
		} else {
			buf.WriteByte('%')
			buf.WriteString(strings.ToUpper(strconv.FormatUint(uint64(c), 16)))
		}
	}
	return buf.String()
}

func convertType(v reflect.Value) (*string, error) {
	v = reflect.Indirect(v)
	if !v.IsValid() {
		return nil, nil
	}

	var str string
	switch value := v.Interface().(type) {
	case string:
		str = value
	case []byte:
		str = base64.StdEncoding.EncodeToString(value)
	case bool:
		str = strconv.FormatBool(value)
	case int64:
		str = strconv.FormatInt(value, 10)
	case float64:
		str = strconv.FormatFloat(value, 'f', -1, 64)
	case time.Time:
		str = value.UTC().Format(RFC822)
	default:
		err := fmt.Errorf("Unsupported value for param %v (%s)", v.Interface(), v.Type())
		return nil, err
	}
	return &str, nil
}
