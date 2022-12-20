package rest

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	awsStrings "github.com/aws/aws-sdk-go/internal/strings"
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
		if err := unmarshalBody(r, v); err != nil {
			r.Error = err
		}
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
		if err := UnmarshalResponse(r.HTTPResponse, r.Data, aws.BoolValue(r.Config.LowerCaseHeaderMaps)); err != nil {
			r.Error = err
		}
	}
}

// UnmarshalResponse attempts to unmarshal the REST response headers to
// the data type passed in. The type must be a pointer. An error is returned
// with any error unmarshaling the response into the target datatype.
func UnmarshalResponse(resp *http.Response, data interface{}, lowerCaseHeaderMaps bool) error {
	v := reflect.Indirect(reflect.ValueOf(data))
	return unmarshalLocationElements(resp, v, lowerCaseHeaderMaps)
}

func unmarshalBody(r *request.Request, v reflect.Value) error {
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
							return awserr.New(request.ErrCodeSerialization, "failed to decode REST response", err)
						}

						payload.Set(reflect.ValueOf(b))

					case *string:
						defer r.HTTPResponse.Body.Close()
						b, err := ioutil.ReadAll(r.HTTPResponse.Body)
						if err != nil {
							return awserr.New(request.ErrCodeSerialization, "failed to decode REST response", err)
						}

						str := string(b)
						payload.Set(reflect.ValueOf(&str))

					default:
						switch payload.Type().String() {
						case "io.ReadCloser":
							payload.Set(reflect.ValueOf(r.HTTPResponse.Body))

						case "io.ReadSeeker":
							b, err := ioutil.ReadAll(r.HTTPResponse.Body)
							if err != nil {
								return awserr.New(request.ErrCodeSerialization,
									"failed to read response body", err)
							}
							payload.Set(reflect.ValueOf(ioutil.NopCloser(bytes.NewReader(b))))

						default:
							io.Copy(ioutil.Discard, r.HTTPResponse.Body)
							r.HTTPResponse.Body.Close()
							return awserr.New(request.ErrCodeSerialization,
								"failed to decode REST response",
								fmt.Errorf("unknown payload type %s", payload.Type()))
						}
					}
				}
			}
		}
	}

	return nil
}

func unmarshalLocationElements(resp *http.Response, v reflect.Value, lowerCaseHeaderMaps bool) error {
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
				unmarshalStatusCode(m, resp.StatusCode)

			case "header":
				err := unmarshalHeader(m, resp.Header.Get(name), field.Tag)
				if err != nil {
					return awserr.New(request.ErrCodeSerialization, "failed to decode REST response", err)
				}

			case "headers":
				prefix := field.Tag.Get("locationName")
				err := unmarshalHeaderMap(m, resp.Header, prefix, lowerCaseHeaderMaps)
				if err != nil {
					return awserr.New(request.ErrCodeSerialization, "failed to decode REST response", err)
				}
			}
		}
	}

	return nil
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

func unmarshalHeaderMap(r reflect.Value, headers http.Header, prefix string, normalize bool) error {
	if len(headers) == 0 {
		return nil
	}
	switch r.Interface().(type) {
	case map[string]*string: // we only support string map value types
		out := map[string]*string{}
		for k, v := range headers {
			if awsStrings.HasPrefixFold(k, prefix) {
				if normalize == true {
					k = strings.ToLower(k)
				} else {
					k = http.CanonicalHeaderKey(k)
				}
				out[k[len(prefix):]] = &v[0]
			}
		}
		if len(out) != 0 {
			r.Set(reflect.ValueOf(out))
		}

	}
	return nil
}

func unmarshalHeader(v reflect.Value, header string, tag reflect.StructTag) error {
	switch tag.Get("type") {
	case "jsonvalue":
		if len(header) == 0 {
			return nil
		}
	case "blob":
		if len(header) == 0 {
			return nil
		}
	default:
		if !v.IsValid() || (header == "" && v.Elem().Kind() != reflect.String) {
			return nil
		}
	}

	switch v.Interface().(type) {
	case *string:
		if tag.Get("suppressedJSONValue") == "true" && tag.Get("location") == "header" {
			b, err := base64.StdEncoding.DecodeString(header)
			if err != nil {
				return fmt.Errorf("failed to decode JSONValue, %v", err)
			}
			header = string(b)
		}
		v.Set(reflect.ValueOf(&header))
	case []byte:
		b, err := base64.StdEncoding.DecodeString(header)
		if err != nil {
			return err
		}
		v.Set(reflect.ValueOf(b))
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
		var f float64
		switch {
		case strings.EqualFold(header, floatNaN):
			f = math.NaN()
		case strings.EqualFold(header, floatInf):
			f = math.Inf(1)
		case strings.EqualFold(header, floatNegInf):
			f = math.Inf(-1)
		default:
			var err error
			f, err = strconv.ParseFloat(header, 64)
			if err != nil {
				return err
			}
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
