package generators

var bootstrapText = `package {{.Package}}

import (
	"testing"

	{{.GinkgoImport}}
	{{.GomegaImport}}
)

func Test{{.FormattedName}}(t *testing.T) {
	{{.GomegaPackage}}RegisterFailHandler({{.GinkgoPackage}}Fail)
	{{.GinkgoPackage}}RunSpecs(t, "{{.FormattedName}} Suite")
}
`

var agoutiBootstrapText = `package {{.Package}}

import (
	"testing"

	{{.GinkgoImport}}
	{{.GomegaImport}}
	"github.com/sclevine/agouti"
)

func Test{{.FormattedName}}(t *testing.T) {
	{{.GomegaPackage}}RegisterFailHandler({{.GinkgoPackage}}Fail)
	{{.GinkgoPackage}}RunSpecs(t, "{{.FormattedName}} Suite")
}

var agoutiDriver *agouti.WebDriver

var _ = {{.GinkgoPackage}}BeforeSuite(func() {
	// Choose a WebDriver:

	agoutiDriver = agouti.PhantomJS()
	// agoutiDriver = agouti.Selenium()
	// agoutiDriver = agouti.ChromeDriver()

	{{.GomegaPackage}}Expect(agoutiDriver.Start()).To({{.GomegaPackage}}Succeed())
})

var _ = {{.GinkgoPackage}}AfterSuite(func() {
	{{.GomegaPackage}}Expect(agoutiDriver.Stop()).To({{.GomegaPackage}}Succeed())
})
`
