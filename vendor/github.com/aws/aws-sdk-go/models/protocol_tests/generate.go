package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/template"

	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/private/model/api"
	"github.com/aws/aws-sdk-go/private/util"
)

type testSuite struct {
	*api.API
	Description string
	Cases       []testCase
	title       string
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
	URI        string
	Headers    map[string]string
	StatusCode uint `json:"status_code"`
}

const preamble = `
var _ bytes.Buffer // always import bytes
var _ http.Request
var _ json.Marshaler
var _ time.Time
var _ xmlutil.XMLNode
var _ xml.Attr
var _ = awstesting.GenerateAssertions
var _ = ioutil.Discard
var _ = util.Trim("")
var _ = url.Values{}
var _ = io.EOF
var _ = aws.String
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
	"io",
	"io/ioutil",
	"net/http",
	"testing",
	"time",
	"net/url",
	"",
	"github.com/aws/aws-sdk-go/awstesting",
	"github.com/aws/aws-sdk-go/aws/session",
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
	"github.com/aws/aws-sdk-go/private/util",
	"github.com/stretchr/testify/assert",
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
	sess := session.New()
	svc := New{{ .TestCase.TestSuite.API.StructName }}(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := {{ .ParamsString }}
	req, _ := svc.{{ .TestCase.Given.ExportedName }}Request(input)
	r := req.HTTPRequest

	// build request
	{{ .TestCase.TestSuite.API.ProtocolPackage }}.Build(req)
	assert.NoError(t, req.Error)

	{{ if ne .TestCase.InputTest.Body "" }}// assert body
	assert.NotNil(t, r.Body)
	{{ .BodyAssertions }}{{ end }}

	{{ if ne .TestCase.InputTest.URI "" }}// assert URL
	awstesting.AssertURL(t, "https://test{{ .TestCase.InputTest.URI }}", r.URL.String()){{ end }}

	// assert headers
{{ range $k, $v := .TestCase.InputTest.Headers }}assert.Equal(t, "{{ $v }}", r.Header.Get("{{ $k }}"))
{{ end }}
}
`))

type tplInputTestCaseData struct {
	TestCase             *testCase
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
			fmt.Fprintf(code, "awstesting.AssertXML(t, `%s`, util.Trim(string(body)), %s{})",
				expectedBody, t.TestCase.Given.InputRef.ShapeName)
		} else {
			fmt.Fprintf(code, "assert.Equal(t, `%s`, util.Trim(string(body)))",
				expectedBody)
		}
	case "json", "jsonrpc", "rest-json":
		if strings.HasPrefix(expectedBody, "{") {
			fmt.Fprintf(code, "awstesting.AssertJSON(t, `%s`, util.Trim(string(body)))",
				expectedBody)
		} else {
			fmt.Fprintf(code, "assert.Equal(t, `%s`, util.Trim(string(body)))",
				expectedBody)
		}
	default:
		fmt.Fprintf(code, "assert.Equal(t, `%s`, util.Trim(string(body)))",
			expectedBody)
	}

	return code.String()
}

var tplOutputTestCase = template.Must(template.New("outputcase").Parse(`
func Test{{ .OpName }}(t *testing.T) {
	sess := session.New()
	svc := New{{ .TestCase.TestSuite.API.StructName }}(sess, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte({{ .Body }}))
	req, out := svc.{{ .TestCase.Given.ExportedName }}Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers
	{{ range $k, $v := .TestCase.OutputTest.Headers }}req.HTTPResponse.Header.Set("{{ $k }}", "{{ $v }}")
	{{ end }}

	// unmarshal response
	{{ .TestCase.TestSuite.API.ProtocolPackage }}.UnmarshalMeta(req)
	{{ .TestCase.TestSuite.API.ProtocolPackage }}.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
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

	if i.Params != nil { // input test
		// query test should sort body as form encoded values
		switch i.TestSuite.API.Metadata.Protocol {
		case "query", "ec2":
			m, _ := url.ParseQuery(i.InputTest.Body)
			i.InputTest.Body = m.Encode()
		case "rest-xml":
			i.InputTest.Body = util.SortXML(bytes.NewReader([]byte(i.InputTest.Body)))
		case "json", "rest-json":
			i.InputTest.Body = strings.Replace(i.InputTest.Body, " ", "", -1)
		}

		input := tplInputTestCaseData{
			TestCase:     i,
			OpName:       strings.ToUpper(opName[0:1]) + opName[1:],
			ParamsString: awstesting.ParamsStructFromJSON(i.Params, i.Given.InputRef.Shape, false),
		}

		if err := tplInputTestCase.Execute(&buf, input); err != nil {
			panic(err)
		}
	} else {
		output := tplOutputTestCaseData{
			TestCase:   i,
			Body:       fmt.Sprintf("%q", i.OutputTest.Body),
			OpName:     strings.ToUpper(opName[0:1]) + opName[1:],
			Assertions: awstesting.GenerateAssertions(i.Data, i.Given.OutputRef.Shape, "out"),
		}

		if err := tplOutputTestCase.Execute(&buf, output); err != nil {
			panic(err)
		}
	}

	return buf.String()
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

		suite.API.NoInitMethods = true       // don't generate init methods
		suite.API.NoStringerMethods = true   // don't generate stringer methods
		suite.API.NoConstServiceNames = true // don't generate service names
		suite.API.Setup()
		suite.API.Metadata.EndpointPrefix = suite.API.PackageName()

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

func main() {
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
