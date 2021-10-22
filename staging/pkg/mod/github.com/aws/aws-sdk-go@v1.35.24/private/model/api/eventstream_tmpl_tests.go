// +build codegen

package api

import (
	"bytes"
	"fmt"
	"strings"
)

// APIEventStreamTestGoCode generates Go code for EventStream operation tests.
func (a *API) APIEventStreamTestGoCode() string {
	var buf bytes.Buffer

	a.resetImports()
	a.AddImport("bytes")
	a.AddImport("io/ioutil")
	a.AddImport("net/http")
	a.AddImport("reflect")
	a.AddImport("testing")
	a.AddImport("time")
	a.AddImport("context")
	a.AddImport("strings")
	a.AddImport("sync")
	a.AddSDKImport("aws")
	a.AddSDKImport("aws/corehandlers")
	a.AddSDKImport("aws/request")
	a.AddSDKImport("aws/awserr")
	a.AddSDKImport("awstesting/unit")
	a.AddSDKImport("private/protocol")
	a.AddSDKImport("private/protocol/", a.ProtocolPackage())
	a.AddSDKImport("private/protocol/eventstream")
	a.AddSDKImport("private/protocol/eventstream/eventstreamapi")
	a.AddSDKImport("private/protocol/eventstream/eventstreamtest")

	unused := `
	var _ time.Time
	var _ awserr.Error
	var _ context.Context
	var _ sync.WaitGroup
	var _ strings.Reader
	`

	if err := eventStreamReaderTestTmpl.Execute(&buf, a); err != nil {
		panic(err)
	}

	if err := eventStreamWriterTestTmpl.Execute(&buf, a); err != nil {
		panic(err)
	}

	return a.importsGoCode() + unused + strings.TrimSpace(buf.String())
}

func templateMap(args ...interface{}) map[string]interface{} {
	if len(args)%2 != 0 {
		panic(fmt.Sprintf("invalid map call, non-even args %v", args))
	}

	m := map[string]interface{}{}
	for i := 0; i < len(args); i += 2 {
		k, ok := args[i].(string)
		if !ok {
			panic(fmt.Sprintf("invalid map call, arg is not string, %T, %v", args[i], args[i]))
		}
		m[k] = args[i+1]
	}

	return m
}

func valueForType(s *Shape, visited []string) string {
	for _, v := range visited {
		if v == s.ShapeName {
			return "nil"
		}
	}

	visited = append(visited, s.ShapeName)

	switch s.Type {
	case "blob":
		return `[]byte("blob value goes here")`
	case "string":
		return `aws.String("string value goes here")`
	case "boolean":
		return `aws.Bool(true)`
	case "byte":
		return `aws.Int64(1)`
	case "short":
		return `aws.Int64(12)`
	case "integer":
		return `aws.Int64(123)`
	case "long":
		return `aws.Int64(1234)`
	case "float":
		return `aws.Float64(123.4)`
	case "double":
		return `aws.Float64(123.45)`
	case "timestamp":
		return `aws.Time(time.Unix(1396594860, 0).UTC())`
	case "structure":
		w := bytes.NewBuffer(nil)
		fmt.Fprintf(w, "&%s{\n", s.ShapeName)
		if s.Exception {
			fmt.Fprintf(w, `RespMetadata: protocol.ResponseMetadata{
	StatusCode: 200,
},
`)
		}
		for _, refName := range s.MemberNames() {
			fmt.Fprintf(w, "%s: %s,\n", refName, valueForType(s.MemberRefs[refName].Shape, visited))
		}
		fmt.Fprintf(w, "}")
		return w.String()
	case "list":
		w := bytes.NewBuffer(nil)
		fmt.Fprintf(w, "%s{\n", s.GoType())
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "%s,\n", valueForType(s.MemberRef.Shape, visited))
		}
		fmt.Fprintf(w, "}")
		return w.String()

	case "map":
		w := bytes.NewBuffer(nil)
		fmt.Fprintf(w, "%s{\n", s.GoType())
		for _, k := range []string{"a", "b", "c"} {
			fmt.Fprintf(w, "%q: %s,\n", k, valueForType(s.ValueRef.Shape, visited))
		}
		fmt.Fprintf(w, "}")
		return w.String()

	default:
		panic(fmt.Sprintf("valueForType does not support %s, %s", s.ShapeName, s.Type))
	}
}
