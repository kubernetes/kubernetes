/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package benchmark

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"text/template"

	v1 "k8s.io/api/core/v1"
	runtimev1 "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
)

type logMessage struct {
	msg       string
	verbosity int
	err       error
	isError   bool
	kvs       []interface{}
}

const (
	stringArg          = "string"
	multiLineStringArg = "multiLineString"
	objectStringArg    = "objectString"
	numberArg          = "number"
	krefArg            = "kref"
	otherArg           = "other"
	totalArg           = "total"
)

type logStats struct {
	TotalLines, JsonLines, SplitLines, ErrorMessages int

	ArgCounts     map[string]int
	OtherLines    []string
	OtherArgs     []interface{}
	MultiLineArgs [][]string
	ObjectTypes   map[string]int
}

var (
	logStatsTemplate = template.Must(template.New("format").Funcs(template.FuncMap{
		"percent": func(x, y int) string {
			if y == 0 {
				return "NA"
			}
			return fmt.Sprintf("%d%%", x*100/y)
		},
		"sub": func(x, y int) int {
			return x - y
		},
	}).Parse(`Total number of lines: {{.TotalLines}}
JSON line continuation: {{.SplitLines}}
Valid JSON messages: {{.JsonLines}} ({{percent .JsonLines .TotalLines}} of total lines)
Error messages: {{.ErrorMessages}} ({{percent .ErrorMessages .JsonLines}} of valid JSON messages)
Unrecognized lines: {{sub (sub .TotalLines .JsonLines) .SplitLines}}
{{range .OtherLines}} {{if gt (len .) 80}}{{slice . 0 80}}{{else}}{{.}}{{end}}
{{end}}
Args:
 total: {{if .ArgCounts.total}}{{.ArgCounts.total}}{{else}}0{{end}}{{if .ArgCounts.string}}
 strings: {{.ArgCounts.string}} ({{percent .ArgCounts.string .ArgCounts.total}}){{end}} {{if .ArgCounts.multiLineString}}
   with line breaks: {{.ArgCounts.multiLineString}} ({{percent .ArgCounts.multiLineString .ArgCounts.total}} of all arguments)
   {{range .MultiLineArgs}}  ===== {{index . 0}} =====
{{index . 1}}

{{end}}{{end}}{{if .ArgCounts.objectString}}
   with API objects: {{.ArgCounts.objectString}} ({{percent .ArgCounts.objectString .ArgCounts.total}} of all arguments)
     types and their number of usage:{{range $key, $value := .ObjectTypes}} {{ $key }}:{{ $value }}{{end}}{{end}}{{if .ArgCounts.number}}
 numbers: {{.ArgCounts.number}} ({{percent .ArgCounts.number .ArgCounts.total}}){{end}}{{if .ArgCounts.kref}}
 ObjectRef: {{.ArgCounts.kref}} ({{percent .ArgCounts.kref .ArgCounts.total}}){{end}}{{if .ArgCounts.other}}
 others: {{.ArgCounts.other}} ({{percent .ArgCounts.other .ArgCounts.total}}){{end}}
`))
)

// This produces too much output:
// {{range .OtherArgs}} {{.}}
// {{end}}

// Doesn't work?
// Unrecognized lines: {{with $delta := sub .TotalLines .JsonLines}}{{$delta}} ({{percent $delta .TotalLines}} of total lines){{end}}

func (s logStats) String() string {
	var buffer bytes.Buffer
	err := logStatsTemplate.Execute(&buffer, &s)
	if err != nil {
		return err.Error()
	}
	return buffer.String()
}

func loadLog(path string) (messages []logMessage, stats logStats, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, logStats{}, err
	}
	defer file.Close()

	stats.ArgCounts = map[string]int{}
	scanner := bufio.NewScanner(file)
	var buffer bytes.Buffer
	for lineNo := 0; scanner.Scan(); lineNo++ {
		stats.TotalLines++
		line := scanner.Bytes()
		buffer.Write(line)
		msg, err := parseLine(buffer.Bytes(), &stats)
		if err != nil {
			// JSON might have been split across multiple lines.
			var jsonErr *json.SyntaxError
			if errors.As(err, &jsonErr) && jsonErr.Offset > 1 {
				// The start of the buffer was okay. Keep the
				// data and add the next line to it.
				stats.SplitLines++
				continue
			}
			stats.OtherLines = append(stats.OtherLines, fmt.Sprintf("%d: %s", lineNo, string(line)))
			buffer.Reset()
			continue
		}
		stats.JsonLines++
		messages = append(messages, msg)
		buffer.Reset()
	}

	if err := scanner.Err(); err != nil {
		return nil, logStats{}, fmt.Errorf("reading %s failed: %v", path, err)
	}

	return
}

// String format for API structs from generated.pb.go.
// &Container{...}
var objectRE = regexp.MustCompile(`^&([a-zA-Z]*)\{`)

func parseLine(line []byte, stats *logStats) (item logMessage, err error) {

	content := map[string]interface{}{}
	if err := json.Unmarshal(line, &content); err != nil {
		return logMessage{}, fmt.Errorf("JSON parsing failed: %w", err)
	}

	kvs := map[string]interface{}{}
	item.isError = true
	for key, value := range content {
		switch key {
		case "v":
			verbosity, ok := value.(float64)
			if !ok {
				return logMessage{}, fmt.Errorf("expected number for v, got: %T %v", value, value)
			}
			item.verbosity = int(verbosity)
			item.isError = false
		case "msg":
			msg, ok := value.(string)
			if !ok {
				return logMessage{}, fmt.Errorf("expected string for msg, got: %T %v", value, value)
			}
			item.msg = msg
		case "ts", "caller":
			// ignore
		case "err":
			errStr, ok := value.(string)
			if !ok {
				return logMessage{}, fmt.Errorf("expected string for err, got: %T %v", value, value)
			}
			item.err = errors.New(errStr)
			stats.ArgCounts[stringArg]++
			stats.ArgCounts[totalArg]++
		default:
			if obj := toObject(value); obj != nil {
				value = obj
			}
			switch value := value.(type) {
			case string:
				stats.ArgCounts[stringArg]++
				if strings.Contains(value, "\n") {
					stats.ArgCounts[multiLineStringArg]++
					stats.MultiLineArgs = append(stats.MultiLineArgs, []string{key, value})
				}
				match := objectRE.FindStringSubmatch(value)
				if match != nil {
					if stats.ObjectTypes == nil {
						stats.ObjectTypes = map[string]int{}
					}
					stats.ArgCounts[objectStringArg]++
					stats.ObjectTypes[match[1]]++
				}
			case float64:
				stats.ArgCounts[numberArg]++
			case klog.ObjectRef:
				stats.ArgCounts[krefArg]++
			default:
				stats.ArgCounts[otherArg]++
				stats.OtherArgs = append(stats.OtherArgs, value)
			}
			stats.ArgCounts[totalArg]++
			kvs[key] = value
		}
	}

	// Sort by key.
	var keys []string
	for key := range kvs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		item.kvs = append(item.kvs, key, kvs[key])
	}

	if !item.isError && item.err != nil {
		// Error is a normal key/value.
		item.kvs = append(item.kvs, "err", item.err)
		item.err = nil
	}
	if item.isError {
		stats.ErrorMessages++
	}
	return
}

// This is a list of objects that might have been dumped.  The simple ones must
// come first because unmarshaling will try one after the after and an
// ObjectRef would unmarshal fine into any of the others whereas any of the
// other types hopefully have enough extra fields that they won't fit (unknown
// fields are an error).
var objectTypes = []reflect.Type{
	reflect.TypeOf(klog.ObjectRef{}),
	reflect.TypeOf(&runtimev1.VersionResponse{}),
	reflect.TypeOf(&v1.Pod{}),
	reflect.TypeOf(&v1.Container{}),
}

func toObject(value interface{}) interface{} {
	data, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil
	}
	for _, t := range objectTypes {
		obj := reflect.New(t)
		decoder := json.NewDecoder(bytes.NewBuffer(jsonData))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(obj.Interface()); err == nil {
			return reflect.Indirect(obj).Interface()
		}
	}
	return nil
}
