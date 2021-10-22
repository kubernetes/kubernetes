package protocol

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
)

var testJSONValueCases = []struct {
	Value  aws.JSONValue
	Mode   EscapeMode
	String string
}{
	{
		Value: aws.JSONValue{
			"abc": 123.,
		},
		Mode:   NoEscape,
		String: `{"abc":123}`,
	},
	{
		Value: aws.JSONValue{
			"abc": 123.,
		},
		Mode:   Base64Escape,
		String: `eyJhYmMiOjEyM30=`,
	},
	{
		Value: aws.JSONValue{
			"abc": 123.,
		},
		Mode:   QuotedEscape,
		String: `"{\"abc\":123}"`,
	},
}

func TestEncodeJSONValue(t *testing.T) {
	for i, c := range testJSONValueCases {
		str, err := EncodeJSONValue(c.Value, c.Mode)
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}
		if e, a := c.String, str; e != a {
			t.Errorf("%d, expect %v encoded value, got %v", i, e, a)
		}
	}
}

func TestDecodeJSONValue(t *testing.T) {
	for i, c := range testJSONValueCases {
		val, err := DecodeJSONValue(c.String, c.Mode)
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}
		if e, a := c.Value, val; !reflect.DeepEqual(e, a) {
			t.Errorf("%d, expect %v encoded value, got %v", i, e, a)
		}
	}
}

func TestEncodeJSONValue_PanicUnkownMode(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expect panic, got none")
		} else {
			reason := fmt.Sprintf("%v", r)
			if e, a := "unknown EscapeMode", reason; !strings.Contains(a, e) {
				t.Errorf("expect %q to be in %v", e, a)
			}
		}
	}()

	val := aws.JSONValue{}

	EncodeJSONValue(val, 123456)
}
func TestDecodeJSONValue_PanicUnkownMode(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expect panic, got none")
		} else {
			reason := fmt.Sprintf("%v", r)
			if e, a := "unknown EscapeMode", reason; !strings.Contains(a, e) {
				t.Errorf("expect %q to be in %v", e, a)
			}
		}
	}()

	DecodeJSONValue(`{"abc":123}`, 123456)
}
