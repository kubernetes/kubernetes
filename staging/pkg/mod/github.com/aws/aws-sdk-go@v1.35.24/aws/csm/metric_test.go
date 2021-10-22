// +build go1.7

package csm

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
)

func TestTruncateString(t *testing.T) {
	cases := map[string]struct {
		Val    string
		Len    int
		Expect string
	}{
		"no change": {
			Val: "123456789", Len: 10,
			Expect: "123456789",
		},
		"max len": {
			Val: "1234567890", Len: 10,
			Expect: "1234567890",
		},
		"too long": {
			Val: "12345678901", Len: 10,
			Expect: "1234567890",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			v := c.Val
			actual := truncateString(&v, c.Len)
			if e, a := c.Val, v; e != a {
				t.Errorf("expect input value not to change, %v, %v", e, a)
			}
			if e, a := c.Expect, *actual; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}

}

func TestMetric_SetException(t *testing.T) {
	cases := map[string]struct {
		Exc    metricException
		Expect metric
		Final  bool
	}{
		"aws exc": {
			Exc: awsException{
				requestException{exception: "abc", message: "123"},
			},
			Expect: metric{
				AWSException:        aws.String("abc"),
				AWSExceptionMessage: aws.String("123"),
			},
		},
		"sdk exc": {
			Exc: sdkException{
				requestException{exception: "abc", message: "123"},
			},
			Expect: metric{
				SDKException:        aws.String("abc"),
				SDKExceptionMessage: aws.String("123"),
			},
		},
		"final aws exc": {
			Exc: awsException{
				requestException{exception: "abc", message: "123"},
			},
			Expect: metric{
				FinalAWSException:        aws.String("abc"),
				FinalAWSExceptionMessage: aws.String("123"),
			},
			Final: true,
		},
		"final sdk exc": {
			Exc: sdkException{
				requestException{exception: "abc", message: "123"},
			},
			Expect: metric{
				FinalSDKException:        aws.String("abc"),
				FinalSDKExceptionMessage: aws.String("123"),
			},
			Final: true,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			var m metric
			if c.Final {
				m.SetFinalException(c.Exc)
			} else {
				m.SetException(c.Exc)
			}
			if e, a := c.Expect, m; !reflect.DeepEqual(e, a) {
				t.Errorf("expect:\n%#v\nactual:\n%#v\n", e, a)
			}
		})
	}
}
