package awstesting

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"net/url"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/private/model/api"
	"github.com/stretchr/testify/assert"
)

// findMember searches the shape for the member with the matching key name.
func findMember(shape *api.Shape, key string) string {
	for actualKey := range shape.MemberRefs {
		if strings.ToLower(key) == strings.ToLower(actualKey) {
			return actualKey
		}
	}
	return ""
}

// GenerateAssertions builds assertions for a shape based on its type.
//
// The shape's recursive values also will have assertions generated for them.
func GenerateAssertions(out interface{}, shape *api.Shape, prefix string) string {
	switch t := out.(type) {
	case map[string]interface{}:
		keys := SortedKeys(t)

		code := ""
		if shape.Type == "map" {
			for _, k := range keys {
				v := t[k]
				s := shape.ValueRef.Shape
				code += GenerateAssertions(v, s, prefix+"[\""+k+"\"]")
			}
		} else {
			for _, k := range keys {
				v := t[k]
				m := findMember(shape, k)
				s := shape.MemberRefs[m].Shape
				code += GenerateAssertions(v, s, prefix+"."+m+"")
			}
		}
		return code
	case []interface{}:
		code := ""
		for i, v := range t {
			s := shape.MemberRef.Shape
			code += GenerateAssertions(v, s, prefix+"["+strconv.Itoa(i)+"]")
		}
		return code
	default:
		switch shape.Type {
		case "timestamp":
			return fmt.Sprintf("assert.Equal(t, time.Unix(%#v, 0).UTC().String(), %s.String())\n", out, prefix)
		case "blob":
			return fmt.Sprintf("assert.Equal(t, %#v, string(%s))\n", out, prefix)
		case "integer", "long":
			return fmt.Sprintf("assert.Equal(t, int64(%#v), *%s)\n", out, prefix)
		default:
			if !reflect.ValueOf(out).IsValid() {
				return fmt.Sprintf("assert.Nil(t, %s)\n", prefix)
			}
			return fmt.Sprintf("assert.Equal(t, %#v, *%s)\n", out, prefix)
		}
	}
}

// Match is a testing helper to test for testing error by comparing expected
// with a regular expression.
func Match(t *testing.T, regex, expected string) {
	if !regexp.MustCompile(regex).Match([]byte(expected)) {
		t.Errorf("%q\n\tdoes not match /%s/", expected, regex)
	}
}

// AssertURL verifies the expected URL is matches the actual.
func AssertURL(t *testing.T, expect, actual string, msgAndArgs ...interface{}) bool {
	expectURL, err := url.Parse(expect)
	if err != nil {
		t.Errorf(errMsg("unable to parse expected URL", err, msgAndArgs))
		return false
	}
	actualURL, err := url.Parse(actual)
	if err != nil {
		t.Errorf(errMsg("unable to parse actual URL", err, msgAndArgs))
		return false
	}

	assert.Equal(t, expectURL.Host, actualURL.Host, msgAndArgs...)
	assert.Equal(t, expectURL.Scheme, actualURL.Scheme, msgAndArgs...)
	assert.Equal(t, expectURL.Path, actualURL.Path, msgAndArgs...)

	return AssertQuery(t, expectURL.Query().Encode(), actualURL.Query().Encode(), msgAndArgs...)
}

// AssertQuery verifies the expect HTTP query string matches the actual.
func AssertQuery(t *testing.T, expect, actual string, msgAndArgs ...interface{}) bool {
	expectQ, err := url.ParseQuery(expect)
	if err != nil {
		t.Errorf(errMsg("unable to parse expected Query", err, msgAndArgs))
		return false
	}
	actualQ, err := url.ParseQuery(expect)
	if err != nil {
		t.Errorf(errMsg("unable to parse actual Query", err, msgAndArgs))
		return false
	}

	// Make sure the keys are the same
	if !assert.Equal(t, queryValueKeys(expectQ), queryValueKeys(actualQ), msgAndArgs...) {
		return false
	}

	for k, expectQVals := range expectQ {
		sort.Strings(expectQVals)
		actualQVals := actualQ[k]
		sort.Strings(actualQVals)
		assert.Equal(t, expectQVals, actualQVals, msgAndArgs...)
	}

	return true
}

// AssertJSON verifies that the expect json string matches the actual.
func AssertJSON(t *testing.T, expect, actual string, msgAndArgs ...interface{}) bool {
	expectVal := map[string]interface{}{}
	if err := json.Unmarshal([]byte(expect), &expectVal); err != nil {
		t.Errorf(errMsg("unable to parse expected JSON", err, msgAndArgs...))
		return false
	}

	actualVal := map[string]interface{}{}
	if err := json.Unmarshal([]byte(actual), &actualVal); err != nil {
		t.Errorf(errMsg("unable to parse actual JSON", err, msgAndArgs...))
		return false
	}

	return assert.Equal(t, expectVal, actualVal, msgAndArgs...)
}

// AssertXML verifies that the expect xml string matches the actual.
func AssertXML(t *testing.T, expect, actual string, container interface{}, msgAndArgs ...interface{}) bool {
	expectVal := container
	if err := xml.Unmarshal([]byte(expect), &expectVal); err != nil {
		t.Errorf(errMsg("unable to parse expected XML", err, msgAndArgs...))
	}

	actualVal := container
	if err := xml.Unmarshal([]byte(actual), &actualVal); err != nil {
		t.Errorf(errMsg("unable to parse actual XML", err, msgAndArgs...))
	}
	return assert.Equal(t, expectVal, actualVal, msgAndArgs...)
}

func errMsg(baseMsg string, err error, msgAndArgs ...interface{}) string {
	message := messageFromMsgAndArgs(msgAndArgs)
	if message != "" {
		message += ", "
	}
	return fmt.Sprintf("%s%s, %v", message, baseMsg, err)
}

func messageFromMsgAndArgs(msgAndArgs []interface{}) string {
	if len(msgAndArgs) == 0 || msgAndArgs == nil {
		return ""
	}
	if len(msgAndArgs) == 1 {
		return msgAndArgs[0].(string)
	}
	if len(msgAndArgs) > 1 {
		return fmt.Sprintf(msgAndArgs[0].(string), msgAndArgs[1:]...)
	}
	return ""
}

func queryValueKeys(v url.Values) []string {
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
