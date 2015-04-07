package matchers_test

import (
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

type kungFuActor interface {
	DrunkenMaster() bool
}

type jackie struct {
	name string
}

func (j *jackie) DrunkenMaster() bool {
	return true
}

var _ = Describe("ReceiveMatcher", func() {
	Context("with no argument", func() {
		Context("for a buffered channel", func() {
			It("should succeed", func() {
				channel := make(chan bool, 1)

				Ω(channel).ShouldNot(Receive())

				channel <- true

				Ω(channel).Should(Receive())
			})
		})

		Context("for an unbuffered channel", func() {
			It("should succeed (eventually)", func() {
				channel := make(chan bool)

				Ω(channel).ShouldNot(Receive())

				go func() {
					time.Sleep(10 * time.Millisecond)
					channel <- true
				}()

				Eventually(channel).Should(Receive())
			})
		})
	})

	Context("with a pointer argument", func() {
		Context("of the correct type", func() {
			It("should write the value received on the channel to the pointer", func() {
				channel := make(chan int, 1)

				var value int

				Ω(channel).ShouldNot(Receive(&value))
				Ω(value).Should(BeZero())

				channel <- 17

				Ω(channel).Should(Receive(&value))
				Ω(value).Should(Equal(17))
			})
		})

		Context("to various types of objects", func() {
			It("should work", func() {
				//channels of strings
				stringChan := make(chan string, 1)
				stringChan <- "foo"

				var s string
				Ω(stringChan).Should(Receive(&s))
				Ω(s).Should(Equal("foo"))

				//channels of slices
				sliceChan := make(chan []bool, 1)
				sliceChan <- []bool{true, true, false}

				var sl []bool
				Ω(sliceChan).Should(Receive(&sl))
				Ω(sl).Should(Equal([]bool{true, true, false}))

				//channels of channels
				chanChan := make(chan chan bool, 1)
				c := make(chan bool)
				chanChan <- c

				var receivedC chan bool
				Ω(chanChan).Should(Receive(&receivedC))
				Ω(receivedC).Should(Equal(c))

				//channels of interfaces
				jackieChan := make(chan kungFuActor, 1)
				aJackie := &jackie{name: "Jackie Chan"}
				jackieChan <- aJackie

				var theJackie kungFuActor
				Ω(jackieChan).Should(Receive(&theJackie))
				Ω(theJackie).Should(Equal(aJackie))
			})
		})

		Context("of the wrong type", func() {
			It("should error", func() {
				channel := make(chan int)
				var incorrectType bool

				success, err := (&ReceiveMatcher{Arg: &incorrectType}).Match(channel)
				Ω(success).Should(BeFalse())
				Ω(err).Should(HaveOccurred())

				var notAPointer int
				success, err = (&ReceiveMatcher{Arg: notAPointer}).Match(channel)
				Ω(success).Should(BeFalse())
				Ω(err).Should(HaveOccurred())
			})
		})
	})

	Context("with a matcher", func() {
		It("should defer to the underlying matcher", func() {
			intChannel := make(chan int, 1)
			intChannel <- 3
			Ω(intChannel).Should(Receive(Equal(3)))

			intChannel <- 2
			Ω(intChannel).ShouldNot(Receive(Equal(3)))

			stringChannel := make(chan []string, 1)
			stringChannel <- []string{"foo", "bar", "baz"}
			Ω(stringChannel).Should(Receive(ContainElement(ContainSubstring("fo"))))

			stringChannel <- []string{"foo", "bar", "baz"}
			Ω(stringChannel).ShouldNot(Receive(ContainElement(ContainSubstring("archipelago"))))
		})

		It("should defer to the underlying matcher for the message", func() {
			matcher := Receive(Equal(3))
			channel := make(chan int, 1)
			channel <- 2
			matcher.Match(channel)
			Ω(matcher.FailureMessage(channel)).Should(MatchRegexp(`Expected\s+<int>: 2\s+to equal\s+<int>: 3`))

			channel <- 3
			matcher.Match(channel)
			Ω(matcher.NegatedFailureMessage(channel)).Should(MatchRegexp(`Expected\s+<int>: 3\s+not to equal\s+<int>: 3`))
		})

		It("should work just fine with Eventually", func() {
			stringChannel := make(chan string)

			go func() {
				time.Sleep(5 * time.Millisecond)
				stringChannel <- "A"
				time.Sleep(5 * time.Millisecond)
				stringChannel <- "B"
			}()

			Eventually(stringChannel).Should(Receive(Equal("B")))
		})

		Context("if the matcher errors", func() {
			It("should error", func() {
				channel := make(chan int, 1)
				channel <- 3
				success, err := (&ReceiveMatcher{Arg: ContainSubstring("three")}).Match(channel)
				Ω(success).Should(BeFalse())
				Ω(err).Should(HaveOccurred())
			})
		})

		Context("if nothing is received", func() {
			It("should fail", func() {
				channel := make(chan int, 1)
				success, err := (&ReceiveMatcher{Arg: Equal(1)}).Match(channel)
				Ω(success).Should(BeFalse())
				Ω(err).ShouldNot(HaveOccurred())
			})
		})
	})

	Context("When actual is a *closed* channel", func() {
		Context("for a buffered channel", func() {
			It("should work until it hits the end of the buffer", func() {
				channel := make(chan bool, 1)
				channel <- true

				close(channel)

				Ω(channel).Should(Receive())
				Ω(channel).ShouldNot(Receive())
			})
		})

		Context("for an unbuffered channel", func() {
			It("should always fail", func() {
				channel := make(chan bool)
				close(channel)

				Ω(channel).ShouldNot(Receive())
			})
		})
	})

	Context("When actual is a send-only channel", func() {
		It("should error", func() {
			channel := make(chan bool)

			var writerChannel chan<- bool
			writerChannel = channel

			success, err := (&ReceiveMatcher{}).Match(writerChannel)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("when acutal is a non-channel", func() {
		It("should error", func() {
			var nilChannel chan bool

			success, err := (&ReceiveMatcher{}).Match(nilChannel)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&ReceiveMatcher{}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&ReceiveMatcher{}).Match(3)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Describe("when used with eventually and a custom matcher", func() {
		It("should return the matcher's error when a failing value is received on the channel, instead of the must receive something failure", func() {
			failures := InterceptGomegaFailures(func() {
				c := make(chan string, 0)
				Eventually(c, 0.01).Should(Receive(Equal("hello")))
			})
			Ω(failures[0]).Should(ContainSubstring("When passed a matcher, ReceiveMatcher's channel *must* receive something."))

			failures = InterceptGomegaFailures(func() {
				c := make(chan string, 1)
				c <- "hi"
				Eventually(c, 0.01).Should(Receive(Equal("hello")))
			})
			Ω(failures[0]).Should(ContainSubstring("<string>: hello"))
		})
	})

	Describe("Bailing early", func() {
		It("should bail early when passed a closed channel", func() {
			c := make(chan bool)
			close(c)

			t := time.Now()
			failures := InterceptGomegaFailures(func() {
				Eventually(c).Should(Receive())
			})
			Ω(time.Since(t)).Should(BeNumerically("<", 500*time.Millisecond))
			Ω(failures).Should(HaveLen(1))
		})

		It("should bail early when passed a non-channel", func() {
			t := time.Now()
			failures := InterceptGomegaFailures(func() {
				Eventually(3).Should(Receive())
			})
			Ω(time.Since(t)).Should(BeNumerically("<", 500*time.Millisecond))
			Ω(failures).Should(HaveLen(1))
		})
	})
})
