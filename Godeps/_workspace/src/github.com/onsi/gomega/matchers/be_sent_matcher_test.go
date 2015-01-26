package matchers_test

import (
	"time"
	. "github.com/onsi/gomega/matchers"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("BeSent", func() {
	Context("when passed a channel and a matching type", func() {
		Context("when the channel is ready to receive", func() {
			It("should succeed and send the value down the channel", func() {
				c := make(chan string)
				d := make(chan string)
				go func() {
					val := <-c
					d <- val
				}()

				time.Sleep(10 * time.Millisecond)

				Ω(c).Should(BeSent("foo"))
				Eventually(d).Should(Receive(Equal("foo")))
			})

			It("should succeed (with a buffered channel)", func() {
				c := make(chan string, 1)
				Ω(c).Should(BeSent("foo"))
				Ω(<-c).Should(Equal("foo"))
			})
		})

		Context("when the channel is not ready to receive", func() {
			It("should fail and not send down the channel", func() {
				c := make(chan string)
				Ω(c).ShouldNot(BeSent("foo"))
				Consistently(c).ShouldNot(Receive())
			})
		})

		Context("when the channel is eventually ready to receive", func() {
			It("should succeed", func() {
				c := make(chan string)
				d := make(chan string)
				go func() {
					time.Sleep(30 * time.Millisecond)
					val := <-c
					d <- val
				}()

				Eventually(c).Should(BeSent("foo"))
				Eventually(d).Should(Receive(Equal("foo")))
			})
		})

		Context("when the channel is closed", func() {
			It("should error", func() {
				c := make(chan string)
				close(c)
				success, err := (&BeSentMatcher{Arg: "foo"}).Match(c)
				Ω(success).Should(BeFalse())
				Ω(err).Should(HaveOccurred())
			})

			It("should short-circuit Eventually", func() {
				c := make(chan string)
				close(c)

				t := time.Now()
				failures := InterceptGomegaFailures(func() {
					Eventually(c, 10.0).Should(BeSent("foo"))
				})
				Ω(failures).Should(HaveLen(1))
				Ω(time.Since(t)).Should(BeNumerically("<", time.Second))
			})
		})
	})

	Context("when passed a channel and a non-matching type", func() {
		It("should error", func() {
			success, err := (&BeSentMatcher{Arg: "foo"}).Match(make(chan int, 1))
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("when passed a receive-only channel", func() {
		It("should error", func() {
			var c <-chan string
			c = make(chan string, 1)
			success, err := (&BeSentMatcher{Arg: "foo"}).Match(c)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("when passed a nonchannel", func() {
		It("should error", func() {
			success, err := (&BeSentMatcher{Arg: "foo"}).Match("bar")
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
