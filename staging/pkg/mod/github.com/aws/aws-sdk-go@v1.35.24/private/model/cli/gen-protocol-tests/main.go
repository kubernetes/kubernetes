// +build codegen

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/template"

	"github.com/aws/aws-sdk-go/private/model/api"
	"github.com/aws/aws-sdk-go/private/util"
)

// TestSuiteTypeInput input test
// TestSuiteTypeInput output test
const (
	TestSuiteTypeInput = iota
	TestSuiteTypeOutput
)

type testSuite struct {
	*api.API
	Description    string
	ClientEndpoint string
	Cases          []testCase
	Type           uint
	title          string
}

func (s *testSuite) UnmarshalJSON(p []byte) error {
	type stub testSuite

	var v stub
	if err := json.Unmarshal(p, &v); err != nil {
		return err
	}

	if len(v.ClientEndpoint) == 0 {
		v.ClientEndpoint = "https://test"
	}
	for i := 0; i < len(v.Cases); i++ {
		if len(v.Cases[i].InputTest.Host) == 0 {
			v.Cases[i].InputTest.Host = "test"
		}
		if len(v.Cases[i].InputTest.URI) == 0 {
			v.Cases[i].InputTest.URI = "/"
		}
	}

	*s = testSuite(v)
	return nil
}

type testCase struct {
	TestSuite  *testSuite
	Given      *api.Operation
	Params     interface{}     `json:",omitempty"`
	Data       interface{}     `json:"result,omitempty"`
	InputTest  testExpectation `json:"serialized"`
	OutputTest testExpectation `json:"response"`
}

type testExpectation struct {
	Body       string
	Host       string
	URI        string
	Headers    map[string]string
	JSONValues map[string]string
	StatusCode uint `json:"status_code"`
}

const preamble = `
var _ bytes.Buffer // always import bytes
var _ http.Request
var _ json.Marshaler
var _ time.Time
var _ xmlutil.XMLNode
var _ xml.Attr
var _ = ioutil.Discard
var _ = util.Trim("")
var _ = url.Values{}
var _ = io.EOF
var _ = aws.String
var _ = fmt.Println
var _ = reflect.Value{}

func init() {
	protocol.RandReader = &awstesting.ZeroReader{}
}
`

var reStripSpace = regexp.MustCompile(`\s(\w)`)

var reImportRemoval = regexp.MustCompile(`(?s:import \((.+?)\))`)

func removeImports(code string) string {
	return reImportRemoval.ReplaceAllString(code, "")
}

var extraImports = []string{
	"bytes",
	"encoding/json",
	"encoding/xml",
	"fmt",
	"io",
	"io/ioutil",
	"net/http",
	"testing",
	"time",
	"reflect",
	"net/url",
	"",
	"github.com/aws/aws-sdk-go/awstesting",
	"github.com/aws/aws-sdk-go/awstesting/unit",
	"github.com/aws/aws-sdk-go/private/protocol",
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
	"github.com/aws/aws-sdk-go/private/util",
}

func addImports(code string) string {
	importNames := make([]string, len(extraImports))
	for i, n := range extraImports {
		if n != "" {
			importNames[i] = fmt.Sprintf("%q", n)
		}
	}

	str := reImportRemoval.ReplaceAllString(code, "import (\n"+strings.Join(importNames, "\n")+"$1\n)")
	return str
}

func (t *testSuite) TestSuite() string {
	var buf bytes.Buffer

	t.title = reStripSpace.ReplaceAllStringFunc(t.Description, func(x string) string {
		return strings.ToUpper(x[1:])
	})
	t.title = regexp.MustCompile(`\W`).ReplaceAllString(t.title, "")

	for idx, c := range t.Cases {
		c.TestSuite = t
		buf.WriteString(c.TestCase(idx) + "\n")
	}
	return buf.String()
}

var tplInputTestCase = template.Must(template.New("inputcase").Parse(`
func Test{{ .OpName }}(t *testing.T) {
	svc := New{{ .TestCase.TestSuite.API.StructName }}(unit.Session, &aws.Config{Endpoint: aws.String("{{ .TestCase.TestSuite.ClientEndpoint  }}")})
	{{ if ne .ParamsString "" }}input := {{ .ParamsString }}
	{{ range $k, $v := .JSONValues -}}
	input.{{ $k }} = {{ $v }} 
	{{ end -}}
	req, _ := svc.{{ .TestCase.Given.ExportedName }}Request(input){{ else }}req, _ := svc.{{ .TestCase.Given.ExportedName }}Request(nil){{ end }}
	r := req.HTTPRequest

	// build request
	req.Build()
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	{{ if ne .TestCase.InputTest.Body "" }}// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	{{ .BodyAssertions }}{{ end }}

	// assert URL
	awstesting.AssertURL(t, "https://{{ .TestCase.InputTest.Host }}{{ .TestCase.InputTest.URI }}", r.URL.String())

	// assert headers
	{{ range $k, $v := .TestCase.InputTest.Headers -}}
		if e, a := "{{ $v }}", r.Header.Get("{{ $k }}"); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
	{{ end }}
}
`))

type tplInputTestCaseData struct {
	TestCase             *testCase
	JSONValues           map[string]string
	OpName, ParamsString string
}

func (t tplInputTestCaseData) BodyAssertions() string {
	code := &bytes.Buffer{}
	protocol := t.TestCase.TestSuite.API.Metadata.Protocol

	// Extract the body bytes
	switch protocol {
	case "rest-xml":
		fmt.Fprintln(code, "body := util.SortXML(r.Body)")
	default:
		fmt.Fprintln(code, "body, _ := ioutil.ReadAll(r.Body)")
	}

	// Generate the body verification code
	expectedBody := util.Trim(t.TestCase.InputTest.Body)
	switch protocol {
	case "ec2", "query":
		fmt.Fprintf(code, "awstesting.AssertQuery(t, `%s`, util.Trim(string(body)))",
			expectedBody)
	case "rest-xml":
		if strings.HasPrefix(expectedBody, "<") {
			fmt.Fprintf(code, "awstesting.AssertXML(t, `%s`, util.Trim(body))",
				expectedBody)
		} else {
			code.WriteString(fmtAssertEqual(fmt.Sprintf("%q", expectedBody), "util.Trim(string(body))"))
		}
	case "json", "jsonrpc", "rest-json":
		if strings.HasPrefix(expectedBody, "{") {
			fmt.Fprintf(code, "awstesting.AssertJSON(t, `%s`, util.Trim(string(body)))",
				expectedBody)
		} else {
			code.WriteString(fmtAssertEqual(fmt.Sprintf("%q", expectedBody), "util.Trim(string(body))"))
		}
	default:
		code.WriteString(fmtAssertEqual(expectedBody, "util.Trim(string(body))"))
	}

	return code.String()
}

func fmtAssertEqual(e, a string) string {
	const format = `if e, a := %s, %s; e != a {
		t.Errorf("expect %%v, got %%v", e, a)
	}
	`

	return fmt.Sprintf(format, e, a)
}

func fmtAssertNil(v string) string {
	const format = `if e := %s; e != nil {
		t.Errorf("expect nil, got %%v", e)
	}
	`

	return fmt.Sprintf(format, v)
}

var tplOutputTestCase = template.Must(template.New("outputcase").Parse(`
func Test{{ .OpName }}(t *testing.T) {
	svc := New{{ .TestCase.TestSuite.API.StructName }}(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte({{ .Body }}))
	req, out := svc.{{ .TestCase.Given.ExportedName }}Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers
	{{ range $k, $v := .TestCase.OutputTest.Headers }}req.HTTPResponse.Header.Set("{{ $k }}", "{{ $v }}")
	{{ end }}

	// unmarshal response
	req.Handlers.UnmarshalMeta.Run(req)
	req.Handlers.Unmarshal.Run(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	{{ .Assertions }}
}
`))

type tplOutputTestCaseData struct {
	TestCase                 *testCase
	Body, OpName, Assertions string
}

func (i *testCase) TestCase(idx int) string {
	var buf bytes.Buffer

	opName := i.TestSuite.API.StructName() + i.TestSuite.title + "Case" + strconv.Itoa(idx+1)

	if i.TestSuite.Type == TestSuiteTypeInput { // input test
		// query test should sort body as form encoded values
		switch i.TestSuite.API.Metadata.Protocol {
		case "query", "ec2":
			m, _ := url.ParseQuery(i.InputTest.Body)
			i.InputTest.Body = m.Encode()
		case "rest-xml":
			i.InputTest.Body = util.SortXML(bytes.NewReader([]byte(i.InputTest.Body)))
		case "json", "rest-json":
			// Nothing to do
		}

		jsonValues := buildJSONValues(i.Given.InputRef.Shape)
		var params interface{}
		if m, ok := i.Params.(map[string]interface{}); ok {
			paramsMap := map[string]interface{}{}
			for k, v := range m {
				if _, ok := jsonValues[k]; !ok {
					paramsMap[k] = v
				} else {
					if i.InputTest.JSONValues == nil {
						i.InputTest.JSONValues = map[string]string{}
					}
					i.InputTest.JSONValues[k] = serializeJSONValue(v.(map[string]interface{}))
				}
			}
			params = paramsMap
		} else {
			params = i.Params
		}
		input := tplInputTestCaseData{
			TestCase:     i,
			OpName:       strings.ToUpper(opName[0:1]) + opName[1:],
			ParamsString: api.ParamsStructFromJSON(params, i.Given.InputRef.Shape, false),
			JSONValues:   i.InputTest.JSONValues,
		}

		if err := tplInputTestCase.Execute(&buf, input); err != nil {
			panic(err)
		}
	} else if i.TestSuite.Type == TestSuiteTypeOutput {
		output := tplOutputTestCaseData{
			TestCase:   i,
			Body:       fmt.Sprintf("%q", i.OutputTest.Body),
			OpName:     strings.ToUpper(opName[0:1]) + opName[1:],
			Assertions: GenerateAssertions(i.Data, i.Given.OutputRef.Shape, "out"),
		}

		if err := tplOutputTestCase.Execute(&buf, output); err != nil {
			panic(err)
		}
	}

	return buf.String()
}

func serializeJSONValue(m map[string]interface{}) string {
	str := "aws.JSONValue"
	str += walkMap(m)
	return str
}

func walkMap(m map[string]interface{}) string {
	str := "{"
	for k, v := range m {
		str += fmt.Sprintf("%q:", k)
		switch v.(type) {
		case bool:
			str += fmt.Sprintf("%t,\n", v.(bool))
		case string:
			str += fmt.Sprintf("%q,\n", v.(string))
		case int:
			str += fmt.Sprintf("%d,\n", v.(int))
		case float64:
			str += fmt.Sprintf("%f,\n", v.(float64))
		case map[string]interface{}:
			str += walkMap(v.(map[string]interface{}))
		}
	}
	str += "}"
	return str
}

func buildJSONValues(shape *api.Shape) map[string]struct{} {
	keys := map[string]struct{}{}
	for key, field := range shape.MemberRefs {
		if field.JSONValue {
			keys[key] = struct{}{}
		}
	}
	return keys
}

// generateTestSuite generates a protocol test suite for a given configuration
// JSON protocol test file.
func generateTestSuite(filename string) string {
	inout := "Input"
	if strings.Contains(filename, "output/") {
		inout = "Output"
	}

	var suites []testSuite
	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	err = json.NewDecoder(f).Decode(&suites)
	if err != nil {
		panic(err)
	}

	var buf bytes.Buffer
	buf.WriteString("// Code generated by models/protocol_tests/generate.go. DO NOT EDIT.\n\n")
	buf.WriteString("package " + suites[0].ProtocolPackage() + "_test\n\n")

	var innerBuf bytes.Buffer
	innerBuf.WriteString("//\n// Tests begin here\n//\n\n\n")

	for i, suite := range suites {
		svcPrefix := inout + "Service" + strconv.Itoa(i+1)
		suite.API.Metadata.ServiceAbbreviation = svcPrefix + "ProtocolTest"
		suite.API.Operations = map[string]*api.Operation{}
		for idx, c := range suite.Cases {
			c.Given.ExportedName = svcPrefix + "TestCaseOperation" + strconv.Itoa(idx+1)
			suite.API.Operations[c.Given.ExportedName] = c.Given
		}

		suite.Type = getType(inout)
		suite.API.NoInitMethods = true       // don't generate init methods
		suite.API.NoStringerMethods = true   // don't generate stringer methods
		suite.API.NoConstServiceNames = true // don't generate service names
		suite.API.Setup()
		suite.API.Metadata.EndpointPrefix = suite.API.PackageName()
		suite.API.Metadata.EndpointsID = suite.API.Metadata.EndpointPrefix

		// Sort in order for deterministic test generation
		names := make([]string, 0, len(suite.API.Shapes))
		for n := range suite.API.Shapes {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, name := range names {
			s := suite.API.Shapes[name]
			s.Rename(svcPrefix + "TestShape" + name)
		}

		svcCode := addImports(suite.API.ServiceGoCode())
		if i == 0 {
			importMatch := reImportRemoval.FindStringSubmatch(svcCode)
			buf.WriteString(importMatch[0] + "\n\n")
			buf.WriteString(preamble + "\n\n")
		}
		svcCode = removeImports(svcCode)
		svcCode = strings.Replace(svcCode, "func New(", "func New"+suite.API.StructName()+"(", -1)
		svcCode = strings.Replace(svcCode, "func newClient(", "func new"+suite.API.StructName()+"Client(", -1)
		svcCode = strings.Replace(svcCode, "return newClient(", "return new"+suite.API.StructName()+"Client(", -1)
		buf.WriteString(svcCode + "\n\n")

		apiCode := removeImports(suite.API.APIGoCode())
		apiCode = strings.Replace(apiCode, "var oprw sync.Mutex", "", -1)
		apiCode = strings.Replace(apiCode, "oprw.Lock()", "", -1)
		apiCode = strings.Replace(apiCode, "defer oprw.Unlock()", "", -1)
		buf.WriteString(apiCode + "\n\n")

		innerBuf.WriteString(suite.TestSuite() + "\n")
	}

	return buf.String() + innerBuf.String()
}

// findMember searches the shape for the member with the matching key name.
func findMember(shape *api.Shape, key string) string {
	for actualKey := range shape.MemberRefs {
		if strings.EqualFold(key, actualKey) {
			return actualKey
		}
	}
	return ""
}

// GenerateAssertions builds assertions for a shape based on its type.
//
// The shape's recursive values also will have assertions generated for them.
func GenerateAssertions(out interface{}, shape *api.Shape, prefix string) string {
	if shape == nil {
		return ""
	}
	switch t := out.(type) {
	case map[string]interface{}:
		keys := util.SortedKeys(t)

		code := ""
		if shape.Type == "map" {
			for _, k := range keys {
				v := t[k]
				s := shape.ValueRef.Shape
				code += GenerateAssertions(v, s, prefix+"[\""+k+"\"]")
			}
		} else if shape.Type == "jsonvalue" {
			code += fmt.Sprintf("reflect.DeepEqual(%s, map[string]interface{}%s)\n", prefix, walkMap(out.(map[string]interface{})))
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
			return fmtAssertEqual(
				fmt.Sprintf("time.Unix(%#v, 0).UTC().String()", out),
				fmt.Sprintf("%s.UTC().String()", prefix),
			)
		case "blob":
			return fmtAssertEqual(
				fmt.Sprintf("%#v", out),
				fmt.Sprintf("string(%s)", prefix),
			)
		case "integer", "long":
			return fmtAssertEqual(
				fmt.Sprintf("int64(%#v)", out),
				fmt.Sprintf("*%s", prefix),
			)
		default:
			if !reflect.ValueOf(out).IsValid() {
				return fmtAssertNil(prefix)
			}
			return fmtAssertEqual(
				fmt.Sprintf("%#v", out),
				fmt.Sprintf("*%s", prefix),
			)
		}
	}
}

func getType(t string) uint {
	switch t {
	case "Input":
		return TestSuiteTypeInput
	case "Output":
		return TestSuiteTypeOutput
	default:
		panic("Invalid type for test suite")
	}
}

func main() {
	if len(os.Getenv("AWS_SDK_CODEGEN_DEBUG")) != 0 {
		api.LogDebug(os.Stdout)
	}

	fmt.Println("Generating test suite", os.Args[1:])
	out := generateTestSuite(os.Args[1])
	if len(os.Args) == 3 {
		f, err := os.Create(os.Args[2])
		defer f.Close()
		if err != nil {
			panic(err)
		}
		f.WriteString(util.GoFmt(out))
		f.Close()

		c := exec.Command("gofmt", "-s", "-w", os.Args[2])
		if err := c.Run(); err != nil {
			panic(err)
		}
	} else {
		fmt.Println(out)
	}
}
