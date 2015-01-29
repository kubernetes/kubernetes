package writer_test

import (
	"github.com/onsi/gomega/gbytes"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/writer"
	. "github.com/onsi/gomega"
)

var _ = Describe("Writer", func() {
	var writer *Writer
	var out *gbytes.Buffer

	BeforeEach(func() {
		out = gbytes.NewBuffer()
		writer = New(out)
	})

	It("should stream directly to the outbuffer by default", func() {
		writer.Write([]byte("foo"))
		Ω(out).Should(gbytes.Say("foo"))
	})

	It("should not emit the header when asked to DumpOutWitHeader", func() {
		writer.Write([]byte("foo"))
		writer.DumpOutWithHeader("my header")
		Ω(out).ShouldNot(gbytes.Say("my header"))
		Ω(out).Should(gbytes.Say("foo"))
	})

	Context("when told not to stream", func() {
		BeforeEach(func() {
			writer.SetStream(false)
		})

		It("should only write to the buffer when told to DumpOut", func() {
			writer.Write([]byte("foo"))
			Ω(out).ShouldNot(gbytes.Say("foo"))
			writer.DumpOut()
			Ω(out).Should(gbytes.Say("foo"))
		})

		It("should truncate the internal buffer when told to truncate", func() {
			writer.Write([]byte("foo"))
			writer.Truncate()
			writer.DumpOut()
			Ω(out).ShouldNot(gbytes.Say("foo"))

			writer.Write([]byte("bar"))
			writer.DumpOut()
			Ω(out).Should(gbytes.Say("bar"))
		})

		Describe("emitting a header", func() {
			Context("when the buffer has content", func() {
				It("should emit the header followed by the content", func() {
					writer.Write([]byte("foo"))
					writer.DumpOutWithHeader("my header")

					Ω(out).Should(gbytes.Say("my header"))
					Ω(out).Should(gbytes.Say("foo"))
				})
			})

			Context("when the buffer has no content", func() {
				It("should not emit the header", func() {
					writer.DumpOutWithHeader("my header")

					Ω(out).ShouldNot(gbytes.Say("my header"))
				})
			})
		})
	})
})
