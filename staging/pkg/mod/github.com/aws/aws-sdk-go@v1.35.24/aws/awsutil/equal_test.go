package awsutil_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
)

func TestDeepEqual(t *testing.T) {
	cases := []struct {
		a, b  interface{}
		equal bool
	}{
		{"a", "a", true},
		{"a", "b", false},
		{"a", aws.String(""), false},
		{"a", nil, false},
		{"a", aws.String("a"), true},
		{(*bool)(nil), (*bool)(nil), true},
		{(*bool)(nil), (*string)(nil), false},
		{nil, nil, true},
	}

	for i, c := range cases {
		if awsutil.DeepEqual(c.a, c.b) != c.equal {
			t.Errorf("%d, a:%v b:%v, %t", i, c.a, c.b, c.equal)
		}
	}
}
