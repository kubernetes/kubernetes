// +build codegen

package api

type wafregionalExamplesBuilder struct {
	defaultExamplesBuilder
}

func NewWAFregionalExamplesBuilder() wafregionalExamplesBuilder {
	return wafregionalExamplesBuilder{defaultExamplesBuilder: NewExamplesBuilder()}
}
func (builder wafregionalExamplesBuilder) Imports(a *API) string {
	return `"fmt"
	"strings"
	"time"

	"` + SDKImportRoot + `/aws"
	"` + SDKImportRoot + `/aws/awserr"
	"` + SDKImportRoot + `/aws/session"
	"` + SDKImportRoot + `/service/waf"
	"` + SDKImportRoot + `/service/` + a.PackageName() + `"
	`
}
