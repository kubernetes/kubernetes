package gbytes_test

import (
	. "github.com/onsi/gomega/gbytes"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type speaker struct {
	buffer *Buffer
}

func (s *speaker) Buffer() *Buffer {
	return s.buffer
}

var _ = Describe("SayMatcher", func() {
	var buffer *Buffer

	BeforeEach(func() {
		buffer = NewBuffer()
		buffer.Write([]byte("abc"))
	})

	Context("when actual is not a gexec Buffer, or a BufferProvider", func() {
		It("should error", func() {
			failures := InterceptGomegaFailures(func() {
				Ω("foo").Should(Say("foo"))
			})
			Ω(failures[0]).Should(ContainSubstring("*gbytes.Buffer"))
		})
	})

	Context("when a match is found", func() {
		It("should succeed", func() {
			Ω(buffer).Should(Say("abc"))
		})

		It("should support printf-like formatting", func() {
			Ω(buffer).Should(Say("a%sc", "b"))
		})

		It("should use a regular expression", func() {
			Ω(buffer).Should(Say("a.c"))
		})

		It("should fastforward the buffer", func() {
			buffer.Write([]byte("def"))
			Ω(buffer).Should(Say("abcd"))
			Ω(buffer).Should(Say("ef"))
			Ω(buffer).ShouldNot(Say("[a-z]"))
		})
	})

	Context("when no match is found", func() {
		It("should not error", func() {
			Ω(buffer).ShouldNot(Say("def"))
		})

		Context("when the buffer is closed", func() {
			BeforeEach(func() {
				buffer.Close()
			})

			It("should abort an eventually", func() {
				t := time.Now()
				failures := InterceptGomegaFailures(func() {
					Eventually(buffer).Should(Say("def"))
				})
				Eventually(buffer).ShouldNot(Say("def"))
				Ω(time.Since(t)).Should(BeNumerically("<", 500*time.Millisecond))
				Ω(failures).Should(HaveLen(1))

				t = time.Now()
				Eventually(buffer).Should(Say("abc"))
				Ω(time.Since(t)).Should(BeNumerically("<", 500*time.Millisecond))
			})

			It("should abort a consistently", func() {
				t := time.Now()
				Consistently(buffer, 2.0).ShouldNot(Say("def"))
				Ω(time.Since(t)).Should(BeNumerically("<", 500*time.Millisecond))
			})

			It("should not error with a synchronous matcher", func() {
				Ω(buffer).ShouldNot(Say("def"))
				Ω(buffer).Should(Say("abc"))
			})
		})
	})

	Context("when a positive match fails", func() {
		It("should report where it got stuck", func() {
			Ω(buffer).Should(Say("abc"))
			buffer.Write([]byte("def"))
			failures := InterceptGomegaFailures(func() {
				Ω(buffer).Should(Say("abc"))
			})
			Ω(failures[0]).Should(ContainSubstring("Got stuck at:"))
			Ω(failures[0]).Should(ContainSubstring("def"))
		})
	})

	Context("when a negative match fails", func() {
		It("should report where it got stuck", func() {
			failures := InterceptGomegaFailures(func() {
				Ω(buffer).ShouldNot(Say("abc"))
			})
			Ω(failures[0]).Should(ContainSubstring("Saw:"))
			Ω(failures[0]).Should(ContainSubstring("Which matches the unexpected:"))
			Ω(failures[0]).Should(ContainSubstring("abc"))
		})
	})

	Context("when a match is not found", func() {
		It("should not fastforward the buffer", func() {
			Ω(buffer).ShouldNot(Say("def"))
			Ω(buffer).Should(Say("abc"))
		})
	})

	Context("a nice real-life example", func() {
		It("should behave well", func() {
			Ω(buffer).Should(Say("abc"))
			go func() {
				time.Sleep(10 * time.Millisecond)
				buffer.Write([]byte("def"))
			}()
			Ω(buffer).ShouldNot(Say("def"))
			Eventually(buffer).Should(Say("def"))
		})
	})

	Context("when actual is a BufferProvider", func() {
		It("should use actual's buffer", func() {
			s := &speaker{
				buffer: NewBuffer(),
			}

			Ω(s).ShouldNot(Say("abc"))

			s.Buffer().Write([]byte("abc"))
			Ω(s).Should(Say("abc"))
		})

		It("should abort an eventually", func() {
			s := &speaker{
				buffer: NewBuffer(),
			}

			s.buffer.Close()

			t := time.Now()
			failures := InterceptGomegaFailures(func() {
				Eventually(s).Should(Say("def"))
			})
			Ω(failures).Should(HaveLen(1))
			Ω(time.Since(t)).Should(BeNumerically("<", 500*time.Millisecond))
		})
	})
})
