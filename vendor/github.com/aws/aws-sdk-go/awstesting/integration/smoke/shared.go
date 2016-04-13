// Package smoke contains shared step definitions that are used across integration tests
package smoke

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	. "github.com/lsegal/gucumber"
	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/session"
)

// Session is a shared session for all integration smoke tests to use.
var Session = session.New()

func init() {
	logLevel := Session.Config.LogLevel
	if os.Getenv("DEBUG") != "" {
		logLevel = aws.LogLevel(aws.LogDebug)
	}
	if os.Getenv("DEBUG_SIGNING") != "" {
		logLevel = aws.LogLevel(aws.LogDebugWithSigning)
	}
	if os.Getenv("DEBUG_BODY") != "" {
		logLevel = aws.LogLevel(aws.LogDebugWithHTTPBody)
	}
	Session.Config.LogLevel = logLevel

	When(`^I call the "(.+?)" API$`, func(op string) {
		call(op, nil, false)
	})

	When(`^I call the "(.+?)" API with:$`, func(op string, args [][]string) {
		call(op, args, false)
	})

	Then(`^the value at "(.+?)" should be a list$`, func(member string) {
		vals, _ := awsutil.ValuesAtPath(World["response"], member)
		assert.NotNil(T, vals)
	})

	Then(`^the response should contain a "(.+?)"$`, func(member string) {
		vals, _ := awsutil.ValuesAtPath(World["response"], member)
		assert.NotEmpty(T, vals)
	})

	When(`^I attempt to call the "(.+?)" API with:$`, func(op string, args [][]string) {
		call(op, args, true)
	})

	Then(`^I expect the response error code to be "(.+?)"$`, func(code string) {
		err, ok := World["error"].(awserr.Error)
		assert.True(T, ok, "no error returned")
		if ok {
			assert.Equal(T, code, err.Code(), "Error: %v", err)
		}
	})

	And(`^I expect the response error message to include:$`, func(data string) {
		err, ok := World["error"].(awserr.Error)
		assert.True(T, ok, "no error returned")
		if ok {
			assert.Contains(T, err.Message(), data)
		}
	})

	And(`^I expect the response error message to include one of:$`, func(table [][]string) {
		err, ok := World["error"].(awserr.Error)
		assert.True(T, ok, "no error returned")
		if ok {
			found := false
			for _, row := range table {
				if strings.Contains(err.Message(), row[0]) {
					found = true
					break
				}
			}

			assert.True(T, found, fmt.Sprintf("no error messages matched: \"%s\"", err.Message()))
		}
	})

	When(`^I call the "(.+?)" API with JSON:$`, func(s1 string, data string) {
		callWithJSON(s1, data, false)
	})

	When(`^I attempt to call the "(.+?)" API with JSON:$`, func(s1 string, data string) {
		callWithJSON(s1, data, true)
	})

	Then(`^the error code should be "(.+?)"$`, func(s1 string) {
		err, ok := World["error"].(awserr.Error)
		assert.True(T, ok, "no error returned")
		assert.Equal(T, s1, err.Code())
	})

	And(`^the error message should contain:$`, func(data string) {
		err, ok := World["error"].(awserr.Error)
		assert.True(T, ok, "no error returned")
		assert.Contains(T, err.Error(), data)
	})

	Then(`^the request should fail$`, func() {
		err, ok := World["error"].(awserr.Error)
		assert.True(T, ok, "no error returned")
		assert.Error(T, err)
	})

	Then(`^the request should be successful$`, func() {
		err, ok := World["error"].(awserr.Error)
		assert.False(T, ok, "error returned")
		assert.NoError(T, err)
	})
}

// findMethod finds the op operation on the v structure using a case-insensitive
// lookup. Returns nil if no method is found.
func findMethod(v reflect.Value, op string) *reflect.Value {
	t := v.Type()
	op = strings.ToLower(op)
	for i := 0; i < t.NumMethod(); i++ {
		name := t.Method(i).Name
		if strings.ToLower(name) == op {
			m := v.MethodByName(name)
			return &m
		}
	}
	return nil
}

// call calls an operation on World["client"] by the name op using the args
// table of arguments to set.
func call(op string, args [][]string, allowError bool) {
	v := reflect.ValueOf(World["client"])
	if m := findMethod(v, op); m != nil {
		t := m.Type()
		in := reflect.New(t.In(0).Elem())
		fillArgs(in, args)

		resps := m.Call([]reflect.Value{in})
		World["response"] = resps[0].Interface()
		World["error"] = resps[1].Interface()

		if !allowError {
			err, _ := World["error"].(error)
			assert.NoError(T, err)
		}
	} else {
		assert.Fail(T, "failed to find operation "+op)
	}
}

// reIsNum is a regular expression matching a numeric input (integer)
var reIsNum = regexp.MustCompile(`^\d+$`)

// reIsArray is a regular expression matching a list
var reIsArray = regexp.MustCompile(`^\['.*?'\]$`)
var reArrayElem = regexp.MustCompile(`'(.+?)'`)

// fillArgs fills arguments on the input structure using the args table of
// arguments.
func fillArgs(in reflect.Value, args [][]string) {
	if args == nil {
		return
	}

	for _, row := range args {
		path := row[0]
		var val interface{} = row[1]
		if reIsArray.MatchString(row[1]) {
			quotedStrs := reArrayElem.FindAllString(row[1], -1)
			strs := make([]*string, len(quotedStrs))
			for i, e := range quotedStrs {
				str := e[1 : len(e)-1]
				strs[i] = &str
			}
			val = strs
		} else if reIsNum.MatchString(row[1]) { // handle integer values
			num, err := strconv.ParseInt(row[1], 10, 64)
			if err == nil {
				val = num
			}
		}
		awsutil.SetValueAtPath(in.Interface(), path, val)
	}
}

func callWithJSON(op, j string, allowError bool) {
	v := reflect.ValueOf(World["client"])
	if m := findMethod(v, op); m != nil {
		t := m.Type()
		in := reflect.New(t.In(0).Elem())
		fillJSON(in, j)

		resps := m.Call([]reflect.Value{in})
		World["response"] = resps[0].Interface()
		World["error"] = resps[1].Interface()

		if !allowError {
			err, _ := World["error"].(error)
			assert.NoError(T, err)
		}
	} else {
		assert.Fail(T, "failed to find operation "+op)
	}
}

func fillJSON(in reflect.Value, j string) {
	d := json.NewDecoder(strings.NewReader(j))
	if err := d.Decode(in.Interface()); err != nil {
		panic(err)
	}
}
