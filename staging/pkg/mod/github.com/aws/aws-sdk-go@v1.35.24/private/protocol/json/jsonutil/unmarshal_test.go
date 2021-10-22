// +build go1.7

package jsonutil_test

import (
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/private/protocol/json/jsonutil"
)

func TestUnmarshalJSON_JSONNumber(t *testing.T) {
	type input struct {
		TimeField  *time.Time `locationName:"timeField"`
		IntField   *int64     `locationName:"intField"`
		FloatField *float64   `locationName:"floatField"`
	}

	cases := map[string]struct {
		JSON     string
		Value    input
		Expected input
	}{
		"seconds precision": {
			JSON: `{"timeField":1597094942}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, 00, time.UTC)
					return &dt
				}(),
			},
		},
		"exact milliseconds precision": {
			JSON: `{"timeField":1597094942.123}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, int(123*time.Millisecond), time.UTC)
					return &dt
				}(),
			},
		},
		"microsecond precision truncated": {
			JSON: `{"timeField":1597094942.1235}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, int(123*time.Millisecond), time.UTC)
					return &dt
				}(),
			},
		},
		"nanosecond precision truncated": {
			JSON: `{"timeField":1597094942.123456789}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, int(123*time.Millisecond), time.UTC)
					return &dt
				}(),
			},
		},
		"milliseconds precision as small exponent": {
			JSON: `{"timeField":1.597094942123e9}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, int(123*time.Millisecond), time.UTC)
					return &dt
				}(),
			},
		},
		"milliseconds precision as large exponent": {
			JSON: `{"timeField":1.597094942123E9}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, int(123*time.Millisecond), time.UTC)
					return &dt
				}(),
			},
		},
		"milliseconds precision as exponent with sign": {
			JSON: `{"timeField":1.597094942123e+9}`,
			Expected: input{
				TimeField: func() *time.Time {
					dt := time.Date(2020, 8, 10, 21, 29, 02, int(123*time.Millisecond), time.UTC)
					return &dt
				}(),
			},
		},
		"integer field": {
			JSON: `{"intField":123456789}`,
			Expected: input{
				IntField: aws.Int64(123456789),
			},
		},
		"integer field truncated": {
			JSON: `{"intField":123456789.123}`,
			Expected: input{
				IntField: aws.Int64(123456789),
			},
		},
		"float64 field": {
			JSON: `{"floatField":123456789.123}`,
			Expected: input{
				FloatField: aws.Float64(123456789.123),
			},
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			err := jsonutil.UnmarshalJSON(&tt.Value, bytes.NewReader([]byte(tt.JSON)))
			if err != nil {
				t.Errorf("expect no error, got %v", err)
			}
			if e, a := tt.Expected, tt.Value; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}
