package rest

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
)

// UnmarshalHandler is a named request handler for unmarshaling rest protocol requests
var UnmarshalHandler = request.NamedHandler{Name: "awssdk.rest.Unmarshal", Fn: Unmarshal}

// UnmarshalMetaHandler is a named request handler for unmarshaling rest protocol request metadata
var UnmarshalMetaHandler = request.NamedHandler{Name: "awssdk.rest.UnmarshalMeta", Fn: UnmarshalMeta}

// Unmarshal unmarshals the REST component of a response in a REST service.
func Unmarshal(r *request.Request) {
	if r.DataFilled() {
		v := reflect.Indirect(reflect.ValueOf(r.Data))
		unmarshalBody(r, v)
	}
}

// UnmarshalMeta unmarshals the REST metadata of a response in a REST service
func UnmarshalMeta(r *request.Request) {
	r.RequestID = r.HTTPResponse.Header.Get("X-Amzn-Requestid")
	if r.RequestID == "" {
		// Alternative version of request id in the header
		r.RequestID = r.HTTPResponse.Header.Get("X-Amz-Request-Id")
	}
	if r.DataFilled() {
		v := reflect.Indirect(reflect.ValueOf(r.Data))
		unmarshalLocationElements(r, v)
	}
}

func unmarshalBody(r *request.Request, v reflect.Value) {
	if field, ok := v.Type().FieldByName("_"); ok {
		if payloadName := field.Tag.Get("payload"); payloadName != "" {
			pfield, _ := v.Type().FieldByName(payloadName)
			if ptag := pfield.Tag.Get("type"); ptag != "" && ptag != "structure" {
				payload := v.FieldByName(payloadName)
				if payload.IsValid() {
					switch payload.Interface().(type) {
					case []byte:
						defer r.HTTPResponse.Body.Close()
						b, err := ioutil.ReadAll(r.HTTPResponse.Body)
						if err != nil {
							r.Error = awserr.New("SerializationError", "failed to decode REST response", err)
						} else {
							payload.Set(reflect.ValueOf(b))
						}
					case *string:
						defer r.HTTPResponse.Body.Close()
						b, err := ioutil.ReadAll(r.HTTPResponse.Body)
						if err != nil {
							r.Error = awserr.New("SerializationError", "failed to decode REST response", err)
						} else {
							str := string(b)
							payload.Set(reflect.ValueOf(&str))
						}
					default:
						switch payload.Type().String() {
						case "io.ReadCloser":
							payload.Set(reflect.ValueOf(r.HTTPResponse.Body))
						case "io.ReadSeeker":
							b, err := ioutil.ReadAll(r.HTTPResponse.Body)
							if err != nil {
								r.Error = awserr.New("SerializationError",
									"failed to read response body", err)
								return
							}
							payload.Set(reflect.ValueOf(ioutil.NopCloser(bytes.NewReader(b))))
						default:
							io.Copy(ioutil.Discard, r.HTTPResponse.Body)
							defer r.HTTPResponse.Body.Close()
							r.Error = awserr.New("SerializationError",
								"failed to decode REST response",
								fmt.Errorf("unknown payload type %s", payload.Type()))
						}
					}
				}
			}
		}
	}
}

func unmarshalLocationElements(r *request.Request, v reflect.Value) {
	for i := 0; i < v.NumField(); i++ {
		m, field := v.Field(i), v.Type().Field(i)
		if n := field.Name; n[0:1] == strings.ToLower(n[0:1]) {
			continue
		}

		if m.IsValid() {
			name := field.Tag.Get("locationName")
			if name == "" {
				name = field.Name
			}

			switch field.Tag.Get("location") {
			case "statusCode":
				unmarshalStatusCode(m, r.HTTPResponse.StatusCode)
			case "header":
				err := unmarshalHeader(m, r.HTTPResponse.Header.Get(name), field.Tag)
				if err != nil {
					r.Error = awserr.New("SerializationError", "failed to decode REST response", err)
					break
				}
			case "headers":
				prefix := field.Tag.Get("locationName")
				err := unmarshalHeaderMap(m, r.HTTPResponse.Header, prefix)
				if err != nil {
					r.Error = awserr.New("SerializationError", "failed to decode REST response", err)
					break
				}
			}
		}
		if r.Error != nil {
			return
		}
	}
}

func unmarshalStatusCode(v reflect.Value, statusCode int) {
	if !v.IsValid() {
		return
	}

	switch v.Interface().(type) {
	case *int64:
		s := int64(statusCode)
		v.Set(reflect.ValueOf(&s))
	}
}

func unmarshalHeaderMap(r reflect.Value, headers http.Header, prefix string) error {
	switch r.Interface().(type) {
	case map[string]*string: // we only support string map value types
		out := map[string]*string{}
		for k, v := range headers {
			k = http.CanonicalHeaderKey(k)
			if strings.HasPrefix(strings.ToLower(k), strings.ToLower(prefix)) {
				out[k[len(prefix):]] = &v[0]
			}
		}
		r.Set(reflect.ValueOf(out))
	}
	return nil
}

func unmarshalHeader(v reflect.Value, header string, tag reflect.StructTag) error {
	isJSONValue := tag.Get("type") == "jsonvalue"
	if isJSONValue {
		if len(header) == 0 {
			return nil
		}
	} else if !v.IsValid() || (header == "" && v.Elem().Kind() != reflect.String) {
		return nil
	}

	switch v.Interface().(type) {
	case *string:
		v.Set(reflect.ValueOf(&header))
	case []byte:
		b, err := base64.StdEncoding.DecodeString(header)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(&b))
	case *bool:
		b, err := strconv.ParseBool(header)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(&b))
	case *int64:
		i, err := strconv.ParseInt(header, 10, 64)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(&i))
	case *float64:
		f, err := strconv.ParseFloat(header, 64)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(&f))
	case *time.Time:
		format := tag.Get("timestampFormat")
		if len(format) == 0 {
			format = protocol.RFC822TimeFormatName
		}
		t, err := protocol.ParseTime(format, header)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(&t))
	case aws.JSONValue:
		escaping := protocol.NoEscape
		if tag.Get("location") == "header" {
			escaping = protocol.Base64Escape
		}
		m, err := protocol.DecodeJSONValue(header, escaping)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(m))
	default:
		err := fmt.Errorf("Unsupported value for param %v (%s)", v.Interface(), v.Type())
		return err
	}
	return nil
}
