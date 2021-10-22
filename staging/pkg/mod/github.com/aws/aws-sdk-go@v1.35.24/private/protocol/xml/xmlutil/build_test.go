// +build go1.7

package xmlutil

import (
	"bytes"
	"encoding/xml"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
)

type implicitPayload struct {
	_ struct{} `type:"structure"`

	StrVal *string     `type:"string"`
	Second *nestedType `type:"structure"`
	Third  *nestedType `type:"structure"`
}

type namedImplicitPayload struct {
	_ struct{} `type:"structure" locationName:"namedPayload"`

	StrVal *string     `type:"string"`
	Second *nestedType `type:"structure"`
	Third  *nestedType `type:"structure"`
}

type explicitPayload struct {
	_ struct{} `type:"structure" payload:"Second"`

	Second *nestedType `type:"structure" locationName:"Second"`
}

type useEmptyNested struct {
	_ struct{} `type:"structure" locationName:"useEmptyNested"`

	StrVal *string    `type:"string"`
	Empty  *emptyType `type:"structure"`
}

type useIgnoreNested struct {
	_      struct{}      `type:"structure" locationName:"useIgnoreNested"`
	StrVal *string       `type:"string"`
	Ignore *ignoreNested `type:"structure"`
}

type skipNonPayload struct {
	_     struct{} `type:"structure" locationName:"skipNonPayload"`
	Field *string  `type:"string" location:"header"`
}
type namedEmptyPayload struct {
	_ struct{} `type:"structure" locationName:"namedEmptyPayload"`
}

type nestedType struct {
	_ struct{} `type:"structure"`

	IntVal *int64  `type:"integer"`
	StrVal *string `type:"string"`
}

type emptyType struct {
	_ struct{} `type:"structure"`
}

type ignoreNested struct {
	_ struct{} `type:"structure"`

	IgnoreMe *string `type:"string" ignore:"true"`
}

func TestBuildXML(t *testing.T) {
	cases := map[string]struct {
		Input  interface{}
		Expect string
	}{
		"explicit payload": {
			Input: &explicitPayload{
				Second: &nestedType{
					IntVal: aws.Int64(1234),
					StrVal: aws.String("string value"),
				},
			},
			Expect: `<Second><IntVal>1234</IntVal><StrVal>string value</StrVal></Second>`,
		},
		"implicit payload": {
			Input: &implicitPayload{
				StrVal: aws.String("string value"),
				Second: &nestedType{
					IntVal: aws.Int64(1111),
					StrVal: aws.String("second string"),
				},
				Third: &nestedType{
					IntVal: aws.Int64(2222),
					StrVal: aws.String("third string"),
				},
			},
			Expect: `<Second><IntVal>1111</IntVal><StrVal>second string</StrVal></Second><StrVal>string value</StrVal><Third><IntVal>2222</IntVal><StrVal>third string</StrVal></Third>`,
		},
		"named implicit payload": {
			Input: &namedImplicitPayload{
				StrVal: aws.String("string value"),
				Second: &nestedType{
					IntVal: aws.Int64(1111),
					StrVal: aws.String("second string"),
				},
				Third: &nestedType{
					IntVal: aws.Int64(2222),
					StrVal: aws.String("third string"),
				},
			},
			Expect: `<namedPayload><Second><IntVal>1111</IntVal><StrVal>second string</StrVal></Second><StrVal>string value</StrVal><Third><IntVal>2222</IntVal><StrVal>third string</StrVal></Third></namedPayload>`,
		},
		"empty with fields nested type": {
			Input: &namedImplicitPayload{
				StrVal: aws.String("string value"),
				Second: &nestedType{},
				Third: &nestedType{
					IntVal: aws.Int64(2222),
					StrVal: aws.String("third string"),
				},
			},
			Expect: `<namedPayload><Second></Second><StrVal>string value</StrVal><Third><IntVal>2222</IntVal><StrVal>third string</StrVal></Third></namedPayload>`,
		},
		"empty no fields nested type": {
			Input: &useEmptyNested{
				StrVal: aws.String("string value"),
				Empty:  &emptyType{},
			},
			Expect: `<useEmptyNested><Empty></Empty><StrVal>string value</StrVal></useEmptyNested>`,
		},
		"ignored nested field": {
			Input: &useIgnoreNested{
				StrVal: aws.String("string value"),
				Ignore: &ignoreNested{
					IgnoreMe: aws.String("abc123"),
				},
			},
			Expect: `<useIgnoreNested><Ignore></Ignore><StrVal>string value</StrVal></useIgnoreNested>`,
		},
		"skip non payload root": {
			Input: &skipNonPayload{
				Field: aws.String("value"),
			},
			Expect: "",
		},
		"skip empty root": {
			Input:  &emptyType{},
			Expect: "",
		},
		"named empty payload": {
			Input:  &namedEmptyPayload{},
			Expect: "<namedEmptyPayload></namedEmptyPayload>",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			var w bytes.Buffer
			if err := buildXML(c.Input, xml.NewEncoder(&w), true); err != nil {
				t.Fatalf("expect no error, %v", err)
			}

			if e, a := c.Expect, w.String(); e != a {
				t.Errorf("expect:\n%s\nactual:\n%s\n", e, a)
			}
		})
	}
}
