package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("BeClosedMatcher", func() {
	Context("when passed a channel", func() {
		It("should do the right thing", func() {
			openChannel := make(chan bool)
			Ω(openChannel).ShouldNot(BeClosed())

			var openReaderChannel <-chan bool
			openReaderChannel = openChannel
			Ω(openReaderChannel).ShouldNot(BeClosed())

			closedChannel := make(chan bool)
			close(closedChannel)

			Ω(closedChannel).Should(BeClosed())

			var closedReaderChannel <-chan bool
			closedReaderChannel = closedChannel
			Ω(closedReaderChannel).Should(BeClosed())
		})
	})

	Context("when passed a send-only channel", func() {
		It("should error", func() {
			openChannel := make(chan bool)
			var openWriterChannel chan<- bool
			openWriterChannel = openChannel

			success, err := (&BeClosedMatcher{}).Match(openWriterChannel)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			closedChannel := make(chan bool)
			close(closedChannel)

			var closedWriterChannel chan<- bool
			closedWriterChannel = closedChannel

			success, err = (&BeClosedMatcher{}).Match(closedWriterChannel)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

		})
	})

	Context("when passed something else", func() {
		It("should error", func() {
			var nilChannel chan bool

			success, err := (&BeClosedMatcher{}).Match(nilChannel)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeClosedMatcher{}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeClosedMatcher{}).Match(7)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
