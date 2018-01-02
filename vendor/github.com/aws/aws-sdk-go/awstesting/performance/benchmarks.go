// +build integration

package performance

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/mock"
	"github.com/gucumber/gucumber"
)

// mapCreateClients allows for the creation of clients
func mapCreateClients() {
	clientFns := []func(){}
	for _, c := range clients {
		clientFns = append(clientFns, func() { c.Call([]reflect.Value{reflect.ValueOf(mock.Session)}) })
	}

	gucumber.World["services"] = clientFns
}

func buildAnArrayOfClients() {
	methods := []reflect.Value{}
	params := [][]reflect.Value{}

	for _, c := range clients {
		method, param, err := findAndGetMethod(c.Call([]reflect.Value{reflect.ValueOf(mock.Session)}))
		if err == nil {
			methods = append(methods, method)
			params = append(params, param)
		}
	}

	fns := []func(){}
	for i := 0; i < len(methods); i++ {
		m := methods[i]
		p := params[i]
		f := func() {
			reqs := m.Call(p)
			resp := reqs[0].Interface().(*request.Request).Send()
			fmt.Println(resp)
		}
		fns = append(fns, f)
	}
	gucumber.World["clientFns"] = fns
}

// findAndGetMethod will grab the method, params to be passed to the method, and an error.
// The method that is found, is a method that doesn't have any required input
func findAndGetMethod(client interface{}) (reflect.Value, []reflect.Value, error) {
	v := reflect.ValueOf(client).Type()
	n := v.NumMethod()

outer:
	for i := 0; i < n; i++ {
		method := v.Method(i)
		if method.Type.NumIn() != 2 || strings.HasSuffix(method.Name, "Request") {
			continue
		}
		param := reflect.New(method.Type.In(1).Elem())
		for j := 0; j < param.Elem().NumField(); j++ {
			field := param.Elem().Type().Field(j)
			req := field.Tag.Get("required")

			if req == "true" {
				continue outer
			}
		}

		params := []reflect.Value{reflect.ValueOf(client), param}
		return method.Func, params, nil
	}

	return reflect.Value{}, nil, errors.New("No method found")
}

// benchmarkTask takes a unique key to write to the logger with the benchmark
// result's data
func benchmarkTask(key string, fns []func(), i1 int) error {
	gucumber.World["error"] = nil
	memStatStart := &runtime.MemStats{}
	runtime.ReadMemStats(memStatStart)

	results := testing.Benchmark(func(b *testing.B) {
		for _, f := range fns {
			for i := 0; i < i1; i++ {
				f()
			}
		}
	})

	results.N = i1
	memStatEnd := &runtime.MemStats{}
	runtime.ReadMemStats(memStatEnd)
	l, err := newBenchmarkLogger("stdout")
	if err != nil {
		return err
	}
	l.log(key, results)

	toDynamodb := os.Getenv("AWS_TESTING_LOG_RESULTS") == "true"
	if toDynamodb {
		l, err := newBenchmarkLogger("dynamodb")
		if err != nil {
			return err
		}
		l.log(key+"_start_benchmarks", memStatStart)
		l.log(key+"_end_benchmarks", memStatEnd)
	}

	if memStatStart.Alloc < memStatEnd.Alloc {
		return errors.New("Leaked memory")
	}
	return nil
}
