// Package rest provides RESTful serialization of AWS requests and responses.
package rest

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
)

// RFC822 returns an RFC822 formatted timestamp for AWS protocols
const RFC822 = "Mon, 2 Jan 2006 15:04:05 GMT"

// Whether the byte value can be sent without escaping in AWS URLs
var noEscape [256]bool

var errValueNotSet = fmt.Errorf("value not set")

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

// BuildHandler is a named request handler for building rest protocol requests
var BuildHandler = request.NamedHandler{Name: "awssdk.rest.Build", Fn: Build}

// Build builds the REST component of a service request.
func Build(r *request.Request) {
	if r.ParamsFilled() {
		v := reflect.ValueOf(r.Params).Elem()
		buildLocationElements(r, v, false)
		buildBody(r, v)
	}
}

// BuildAsGET builds the REST component of a service request with the ability to hoist
// data from the body.
func BuildAsGET(r *request.Request) {
	if r.ParamsFilled() {
		v := reflect.ValueOf(r.Params).Elem()
		buildLocationElements(r, v, true)
		buildBody(r, v)
	}
}

func buildLocationElements(r *request.Request, v reflect.Value, buildGETQuery bool) {
	query := r.HTTPRequest.URL.Query()

	// Setup the raw path to match the base path pattern. This is needed
	// so that when the path is mutated a custom escaped version can be
	// stored in RawPath that will be used by the Go client.
	r.HTTPRequest.URL.RawPath = r.HTTPRequest.URL.Path

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
			if kind := m.Kind(); kind == reflect.Ptr {
				m = m.Elem()
			} else if kind == reflect.Interface {
				if !m.Elem().IsValid() {
					continue
				}
			}
			if !m.IsValid() {
				continue
			}
			if field.Tag.Get("ignore") != "" {
				continue
			}

			var err error
			switch field.Tag.Get("location") {
			case "headers": // header maps
				err = buildHeaderMap(&r.HTTPRequest.Header, m, field.Tag)
			case "header":
				err = buildHeader(&r.HTTPRequest.Header, m, name, field.Tag)
			case "uri":
				err = buildURI(r.HTTPRequest.URL, m, name, field.Tag)
			case "querystring":
				err = buildQueryString(query, m, name, field.Tag)
			default:
				if buildGETQuery {
					err = buildQueryString(query, m, name, field.Tag)
				}
			}
			r.Error = err
		}
		if r.Error != nil {
			return
		}
	}

	r.HTTPRequest.URL.RawQuery = query.Encode()
	if !aws.BoolValue(r.Config.DisableRestProtocolURICleaning) {
		cleanPath(r.HTTPRequest.URL)
	}
}

func buildBody(r *request.Request, v reflect.Value) {
	if field, ok := v.Type().FieldByName("_"); ok {
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

func buildHeader(header *http.Header, v reflect.Value, name string, tag reflect.StructTag) error {
	str, err := convertType(v, tag)
	if err == errValueNotSet {
		return nil
	} else if err != nil {
		return awserr.New("SerializationError", "failed to encode REST request", err)
	}

	header.Add(name, str)

	return nil
}

func buildHeaderMap(header *http.Header, v reflect.Value, tag reflect.StructTag) error {
	prefix := tag.Get("locationName")
	for _, key := range v.MapKeys() {
		str, err := convertType(v.MapIndex(key), tag)
		if err == errValueNotSet {
			continue
		} else if err != nil {
			return awserr.New("SerializationError", "failed to encode REST request", err)

		}

		header.Add(prefix+key.String(), str)
	}
	return nil
}

func buildURI(u *url.URL, v reflect.Value, name string, tag reflect.StructTag) error {
	value, err := convertType(v, tag)
	if err == errValueNotSet {
		return nil
	} else if err != nil {
		return awserr.New("SerializationError", "failed to encode REST request", err)
	}

	u.Path = strings.Replace(u.Path, "{"+name+"}", value, -1)
	u.Path = strings.Replace(u.Path, "{"+name+"+}", value, -1)

	u.RawPath = strings.Replace(u.RawPath, "{"+name+"}", EscapePath(value, true), -1)
	u.RawPath = strings.Replace(u.RawPath, "{"+name+"+}", EscapePath(value, false), -1)

	return nil
}

func buildQueryString(query url.Values, v reflect.Value, name string, tag reflect.StructTag) error {
	switch value := v.Interface().(type) {
	case []*string:
		for _, item := range value {
			query.Add(name, *item)
		}
	case map[string]*string:
		for key, item := range value {
			query.Add(key, *item)
		}
	case map[string][]*string:
		for key, items := range value {
			for _, item := range items {
				query.Add(key, *item)
			}
		}
	default:
		str, err := convertType(v, tag)
		if err == errValueNotSet {
			return nil
		} else if err != nil {
			return awserr.New("SerializationError", "failed to encode REST request", err)
		}
		query.Set(name, str)
	}

	return nil
}

func cleanPath(u *url.URL) {
	hasSlash := strings.HasSuffix(u.Path, "/")

	// clean up path, removing duplicate `/`
	u.Path = path.Clean(u.Path)
	u.RawPath = path.Clean(u.RawPath)

	if hasSlash && !strings.HasSuffix(u.Path, "/") {
		u.Path += "/"
		u.RawPath += "/"
	}
}

// EscapePath escapes part of a URL path in Amazon style
func EscapePath(path string, encodeSep bool) string {
	var buf bytes.Buffer
	for i := 0; i < len(path); i++ {
		c := path[i]
		if noEscape[c] || (c == '/' && !encodeSep) {
			buf.WriteByte(c)
		} else {
			fmt.Fprintf(&buf, "%%%02X", c)
		}
	}
	return buf.String()
}

func convertType(v reflect.Value, tag reflect.StructTag) (str string, err error) {
	v = reflect.Indirect(v)
	if !v.IsValid() {
		return "", errValueNotSet
	}

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
	case aws.JSONValue:
		if len(value) == 0 {
			return "", errValueNotSet
		}
		escaping := protocol.NoEscape
		if tag.Get("location") == "header" {
			escaping = protocol.Base64Escape
		}
		str, err = protocol.EncodeJSONValue(value, escaping)
		if err != nil {
			return "", fmt.Errorf("unable to encode JSONValue, %v", err)
		}
	default:
		err := fmt.Errorf("unsupported value for param %v (%s)", v.Interface(), v.Type())
		return "", err
	}
	return str, nil
}
