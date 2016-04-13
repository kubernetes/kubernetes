package flags_test

import (
	"flag"
	"fmt"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/integration/_fixtures/flags_tests"
	. "github.com/onsi/gomega"
	"time"
)

var customFlag string

func init() {
	flag.StringVar(&customFlag, "customFlag", "default", "custom flag!")
}

var _ = Describe("Testing various flags", func() {
	FDescribe("the focused set", func() {
		Measure("a measurement", func(b Benchmarker) {
			b.RecordValue("a value", 3)
		}, 3)

		It("should honor -cover", func() {
			Ω(Tested()).Should(Equal("tested"))
		})

		PIt("should honor -failOnPending and -noisyPendings")

		Describe("smores", func() {
			It("should honor -skip: marshmallow", func() {
				println("marshmallow")
			})

			It("should honor -focus: chocolate", func() {
				println("chocolate")
			})
		})

		It("should detect races", func(done Done) {
			var a string
			go func() {
				a = "now you don't"
				close(done)
			}()
			a = "now you see me"
			println(a)
		})

		It("should randomize A", func() {
			println("RANDOM_A")
		})

		It("should randomize B", func() {
			println("RANDOM_B")
		})

		It("should randomize C", func() {
			println("RANDOM_C")
		})

		It("should honor -slowSpecThreshold", func() {
			time.Sleep(100 * time.Millisecond)
		})

		It("should pass in additional arguments after '--' directly to the test process", func() {
			fmt.Printf("CUSTOM_FLAG: %s", customFlag)
		})
	})

	Describe("more smores", func() {
		It("should not run these unless -focus is set", func() {
			println("smores")
		})
	})

	Describe("a failing test", func() {
		It("should fail", func() {
			Ω(true).Should(Equal(false))
		})
	})
})
