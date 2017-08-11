package apps_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestApps(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Apps Suite")
}
