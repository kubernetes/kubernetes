package ec2_test

import (
	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"
	. "github.com/motain/gocheck"
)

// EC2 ReST authentication docs: http://goo.gl/fQmAN

var testAuth = aws.Auth{"user", "secret", ""}

func (s *S) TestBasicSignature(c *C) {
	params := map[string]string{}
	ec2.Sign(testAuth, "GET", "/path", params, "localhost")
	c.Assert(params["SignatureVersion"], Equals, "2")
	c.Assert(params["SignatureMethod"], Equals, "HmacSHA256")
	expected := "6lSe5QyXum0jMVc7cOUz32/52ZnL7N5RyKRk/09yiK4="
	c.Assert(params["Signature"], Equals, expected)
}

func (s *S) TestParamSignature(c *C) {
	params := map[string]string{
		"param1": "value1",
		"param2": "value2",
		"param3": "value3",
	}
	ec2.Sign(testAuth, "GET", "/path", params, "localhost")
	expected := "XWOR4+0lmK8bD8CGDGZ4kfuSPbb2JibLJiCl/OPu1oU="
	c.Assert(params["Signature"], Equals, expected)
}

func (s *S) TestManyParams(c *C) {
	params := map[string]string{
		"param1":  "value10",
		"param2":  "value2",
		"param3":  "value3",
		"param4":  "value4",
		"param5":  "value5",
		"param6":  "value6",
		"param7":  "value7",
		"param8":  "value8",
		"param9":  "value9",
		"param10": "value1",
	}
	ec2.Sign(testAuth, "GET", "/path", params, "localhost")
	expected := "di0sjxIvezUgQ1SIL6i+C/H8lL+U0CQ9frLIak8jkVg="
	c.Assert(params["Signature"], Equals, expected)
}

func (s *S) TestEscaping(c *C) {
	params := map[string]string{"Nonce": "+ +"}
	ec2.Sign(testAuth, "GET", "/path", params, "localhost")
	c.Assert(params["Nonce"], Equals, "+ +")
	expected := "bqffDELReIqwjg/W0DnsnVUmfLK4wXVLO4/LuG+1VFA="
	c.Assert(params["Signature"], Equals, expected)
}

func (s *S) TestSignatureExample1(c *C) {
	params := map[string]string{
		"Timestamp": "2009-02-01T12:53:20+00:00",
		"Version":   "2007-11-07",
		"Action":    "ListDomains",
	}
	ec2.Sign(aws.Auth{"access", "secret", ""}, "GET", "/", params, "sdb.amazonaws.com")
	expected := "okj96/5ucWBSc1uR2zXVfm6mDHtgfNv657rRtt/aunQ="
	c.Assert(params["Signature"], Equals, expected)
}
