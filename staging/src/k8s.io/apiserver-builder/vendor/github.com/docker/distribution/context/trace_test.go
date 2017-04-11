package context

import (
	"runtime"
	"testing"
	"time"
)

// TestWithTrace ensures that tracing has the expected values in the context.
func TestWithTrace(t *testing.T) {
	pc, file, _, _ := runtime.Caller(0) // get current caller.
	f := runtime.FuncForPC(pc)

	base := []valueTestCase{
		{
			key:           "trace.id",
			notnilorempty: true,
		},

		{
			key:           "trace.file",
			expected:      file,
			notnilorempty: true,
		},
		{
			key:           "trace.line",
			notnilorempty: true,
		},
		{
			key:           "trace.start",
			notnilorempty: true,
		},
	}

	ctx, done := WithTrace(Background())
	defer done("this will be emitted at end of test")

	checkContextForValues(t, ctx, append(base, valueTestCase{
		key:      "trace.func",
		expected: f.Name(),
	}))

	traced := func() {
		parentID := ctx.Value("trace.id") // ensure the parent trace id is correct.

		pc, _, _, _ := runtime.Caller(0) // get current caller.
		f := runtime.FuncForPC(pc)
		ctx, done := WithTrace(ctx)
		defer done("this should be subordinate to the other trace")
		time.Sleep(time.Second)
		checkContextForValues(t, ctx, append(base, valueTestCase{
			key:      "trace.func",
			expected: f.Name(),
		}, valueTestCase{
			key:      "trace.parent.id",
			expected: parentID,
		}))
	}
	traced()

	time.Sleep(time.Second)
}

type valueTestCase struct {
	key           string
	expected      interface{}
	notnilorempty bool // just check not empty/not nil
}

func checkContextForValues(t *testing.T, ctx Context, values []valueTestCase) {

	for _, testcase := range values {
		v := ctx.Value(testcase.key)
		if testcase.notnilorempty {
			if v == nil || v == "" {
				t.Fatalf("value was nil or empty for %q: %#v", testcase.key, v)
			}
			continue
		}

		if v != testcase.expected {
			t.Fatalf("unexpected value for key %q: %v != %v", testcase.key, v, testcase.expected)
		}
	}
}
