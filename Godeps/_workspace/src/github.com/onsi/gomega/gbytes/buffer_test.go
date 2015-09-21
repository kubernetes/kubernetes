package gbytes_test

import (
	"io"
	"time"

	. "github.com/onsi/gomega/gbytes"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Buffer", func() {
	var buffer *Buffer

	BeforeEach(func() {
		buffer = NewBuffer()
	})

	Describe("dumping the entire contents of the buffer", func() {
		It("should return everything that's been written", func() {
			buffer.Write([]byte("abc"))
			buffer.Write([]byte("def"))
			Ω(buffer.Contents()).Should(Equal([]byte("abcdef")))

			Ω(buffer).Should(Say("bcd"))
			Ω(buffer.Contents()).Should(Equal([]byte("abcdef")))
		})
	})

	Describe("creating a buffer with bytes", func() {
		It("should create the buffer with the cursor set to the beginning", func() {
			buffer := BufferWithBytes([]byte("abcdef"))
			Ω(buffer.Contents()).Should(Equal([]byte("abcdef")))
			Ω(buffer).Should(Say("abc"))
			Ω(buffer).ShouldNot(Say("abc"))
			Ω(buffer).Should(Say("def"))
		})
	})

	Describe("reading from a buffer", func() {
		It("should read the current contents of the buffer", func() {
			buffer := BufferWithBytes([]byte("abcde"))

			dest := make([]byte, 3)
			n, err := buffer.Read(dest)
			Ω(err).ShouldNot(HaveOccurred())
			Ω(n).Should(Equal(3))
			Ω(string(dest)).Should(Equal("abc"))

			dest = make([]byte, 3)
			n, err = buffer.Read(dest)
			Ω(err).ShouldNot(HaveOccurred())
			Ω(n).Should(Equal(2))
			Ω(string(dest[:n])).Should(Equal("de"))

			n, err = buffer.Read(dest)
			Ω(err).Should(Equal(io.EOF))
			Ω(n).Should(Equal(0))
		})

		Context("after the buffer has been closed", func() {
			It("returns an error", func() {
				buffer := BufferWithBytes([]byte("abcde"))

				buffer.Close()

				dest := make([]byte, 3)
				n, err := buffer.Read(dest)
				Ω(err).Should(HaveOccurred())
				Ω(n).Should(Equal(0))
			})
		})
	})

	Describe("detecting regular expressions", func() {
		It("should fire the appropriate channel when the passed in pattern matches, then close it", func(done Done) {
			go func() {
				time.Sleep(10 * time.Millisecond)
				buffer.Write([]byte("abcde"))
			}()

			A := buffer.Detect("%s", "a.c")
			B := buffer.Detect("def")

			var gotIt bool
			select {
			case gotIt = <-A:
			case <-B:
				Fail("should not have gotten here")
			}

			Ω(gotIt).Should(BeTrue())
			Eventually(A).Should(BeClosed())

			buffer.Write([]byte("f"))
			Eventually(B).Should(Receive())
			Eventually(B).Should(BeClosed())

			close(done)
		})

		It("should fast-forward the buffer upon detection", func(done Done) {
			buffer.Write([]byte("abcde"))
			<-buffer.Detect("abc")
			Ω(buffer).ShouldNot(Say("abc"))
			Ω(buffer).Should(Say("de"))
			close(done)
		})

		It("should only fast-forward the buffer when the channel is read, and only if doing so would not rewind it", func(done Done) {
			buffer.Write([]byte("abcde"))
			A := buffer.Detect("abc")
			time.Sleep(20 * time.Millisecond) //give the goroutine a chance to detect and write to the channel
			Ω(buffer).Should(Say("abcd"))
			<-A
			Ω(buffer).ShouldNot(Say("d"))
			Ω(buffer).Should(Say("e"))
			Eventually(A).Should(BeClosed())
			close(done)
		})

		It("should be possible to cancel a detection", func(done Done) {
			A := buffer.Detect("abc")
			B := buffer.Detect("def")
			buffer.CancelDetects()
			buffer.Write([]byte("abcdef"))
			Eventually(A).Should(BeClosed())
			Eventually(B).Should(BeClosed())

			Ω(buffer).Should(Say("bcde"))
			<-buffer.Detect("f")
			close(done)
		})
	})

	Describe("closing the buffer", func() {
		It("should error when further write attempts are made", func() {
			_, err := buffer.Write([]byte("abc"))
			Ω(err).ShouldNot(HaveOccurred())

			buffer.Close()

			_, err = buffer.Write([]byte("def"))
			Ω(err).Should(HaveOccurred())

			Ω(buffer.Contents()).Should(Equal([]byte("abc")))
		})

		It("should be closed", func() {
			Ω(buffer.Closed()).Should(BeFalse())

			buffer.Close()

			Ω(buffer.Closed()).Should(BeTrue())
		})
	})
})
