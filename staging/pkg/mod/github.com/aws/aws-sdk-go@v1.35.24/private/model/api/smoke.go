// +build codegen

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"text/template"
)

// SmokeTestSuite defines the test suite for smoke tests.
type SmokeTestSuite struct {
	Version       int             `json:"version"`
	DefaultRegion string          `json:"defaultRegion"`
	TestCases     []SmokeTestCase `json:"testCases"`
}

// SmokeTestCase provides the definition for a integration smoke test case.
type SmokeTestCase struct {
	OpName    string                 `json:"operationName"`
	Input     map[string]interface{} `json:"input"`
	ExpectErr bool                   `json:"errorExpectedFromService"`
}

// BuildInputShape returns the Go code as a string for initializing the test
// case's input shape.
func (c SmokeTestCase) BuildInputShape(ref *ShapeRef) string {
	b := NewShapeValueBuilder()
	return fmt.Sprintf("&%s{\n%s\n}",
		b.GoType(ref, true),
		b.BuildShape(ref, c.Input, false),
	)
}

// AttachSmokeTests attaches the smoke test cases to the API model.
func (a *API) AttachSmokeTests(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open smoke tests %s, err: %v", filename, err)
	}
	defer f.Close()

	if err := json.NewDecoder(f).Decode(&a.SmokeTests); err != nil {
		return fmt.Errorf("failed to decode smoke tests %s, err: %v", filename, err)
	}

	if v := a.SmokeTests.Version; v != 1 {
		return fmt.Errorf("invalid smoke test version, %d", v)
	}

	return nil
}

// APISmokeTestsGoCode returns the Go Code string for the smoke tests.
func (a *API) APISmokeTestsGoCode() string {
	w := bytes.NewBuffer(nil)

	a.resetImports()
	a.AddImport("context")
	a.AddImport("testing")
	a.AddImport("time")
	a.AddSDKImport("aws")
	a.AddSDKImport("aws/request")
	a.AddSDKImport("aws/awserr")
	a.AddSDKImport("aws/request")
	a.AddSDKImport("awstesting/integration")
	a.AddImport(a.ImportPath())

	smokeTests := struct {
		API *API
		SmokeTestSuite
	}{
		API:            a,
		SmokeTestSuite: a.SmokeTests,
	}

	if err := smokeTestTmpl.Execute(w, smokeTests); err != nil {
		panic(fmt.Sprintf("failed to create smoke tests, %v", err))
	}

	ignoreImports := `
	var _ aws.Config
	var _ awserr.Error
	var _ request.Request
	`

	return a.importsGoCode() + ignoreImports + w.String()
}

var smokeTestTmpl = template.Must(template.New(`smokeTestTmpl`).Parse(`
{{- range $i, $testCase := $.TestCases }}
	{{- $op := index $.API.Operations $testCase.OpName }}
	func TestInteg_{{ printf "%02d" $i }}_{{ $op.ExportedName }}(t *testing.T) {
		ctx, cancelFn := context.WithTimeout(context.Background(), 5 *time.Second)
		defer cancelFn()
	
		sess := integration.SessionWithDefaultRegion("{{ $.DefaultRegion }}")
		svc := {{ $.API.PackageName }}.New(sess)
		params := {{ $testCase.BuildInputShape $op.InputRef }}
		_, err := svc.{{ $op.ExportedName }}WithContext(ctx, params, func(r *request.Request) {
			r.Handlers.Validate.RemoveByName("core.ValidateParametersHandler")
		})
		{{- if $testCase.ExpectErr }}
			if err == nil {
				t.Fatalf("expect request to fail")
			}
			aerr, ok := err.(awserr.RequestFailure)
			if !ok {
				t.Fatalf("expect awserr, was %T", err)
			}
			if len(aerr.Code()) == 0 {
				t.Errorf("expect non-empty error code")
			}
			if len(aerr.Message()) == 0 {
				t.Errorf("expect non-empty error message")
			}
			if v := aerr.Code(); v == request.ErrCodeSerialization {
				t.Errorf("expect API error code got serialization failure")
			}
		{{- else }}
			if err != nil {
				t.Errorf("expect no error, got %v", err)
			}
		{{- end }}
	}
{{- end }}
`))
