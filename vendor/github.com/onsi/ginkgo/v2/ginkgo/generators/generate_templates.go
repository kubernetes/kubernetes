package generators

var specText = `{{.BuildTags}}
package {{.Package}}

import (
	{{.GinkgoImport}}
	{{.GomegaImport}}

	{{if .ImportPackage}}"{{.PackageImportPath}}"{{end}}
)

var _ = {{.GinkgoPackage}}Describe("{{.Subject}}", func() {

})
`

var agoutiSpecText = `{{.BuildTags}}
package {{.Package}}

import (
	{{.GinkgoImport}}
	{{.GomegaImport}}
	"github.com/sclevine/agouti"
	. "github.com/sclevine/agouti/matchers"

	{{if .ImportPackage}}"{{.PackageImportPath}}"{{end}}
)

var _ = {{.GinkgoPackage}}Describe("{{.Subject}}", func() {
	var page *agouti.Page

	{{.GinkgoPackage}}BeforeEach(func() {
		var err error
		page, err = agoutiDriver.NewPage()
		{{.GomegaPackage}}Expect(err).NotTo({{.GomegaPackage}}HaveOccurred())
	})

	{{.GinkgoPackage}}AfterEach(func() {
		{{.GomegaPackage}}Expect(page.Destroy()).To({{.GomegaPackage}}Succeed())
	})
})
`
