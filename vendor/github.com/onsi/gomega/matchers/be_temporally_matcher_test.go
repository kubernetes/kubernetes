package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
	"time"
)

var _ = Describe("BeTemporally", func() {

	var t0, t1, t2 time.Time
	BeforeEach(func() {
		t0 = time.Now()
		t1 = t0.Add(time.Second)
		t2 = t0.Add(-time.Second)
	})

	Context("When comparing times", func() {

		It("should support ==", func() {
			Ω(t0).Should(BeTemporally("==", t0))
			Ω(t1).ShouldNot(BeTemporally("==", t0))
			Ω(t0).ShouldNot(BeTemporally("==", t1))
			Ω(t0).ShouldNot(BeTemporally("==", time.Time{}))
		})

		It("should support >", func() {
			Ω(t0).Should(BeTemporally(">", t2))
			Ω(t0).ShouldNot(BeTemporally(">", t0))
			Ω(t2).ShouldNot(BeTemporally(">", t0))
		})

		It("should support <", func() {
			Ω(t0).Should(BeTemporally("<", t1))
			Ω(t0).ShouldNot(BeTemporally("<", t0))
			Ω(t1).ShouldNot(BeTemporally("<", t0))
		})

		It("should support >=", func() {
			Ω(t0).Should(BeTemporally(">=", t2))
			Ω(t0).Should(BeTemporally(">=", t0))
			Ω(t0).ShouldNot(BeTemporally(">=", t1))
		})

		It("should support <=", func() {
			Ω(t0).Should(BeTemporally("<=", t1))
			Ω(t0).Should(BeTemporally("<=", t0))
			Ω(t0).ShouldNot(BeTemporally("<=", t2))
		})

		Context("when passed ~", func() {
			Context("and there is no precision parameter", func() {
				BeforeEach(func() {
					t1 = t0.Add(time.Millisecond / 2)
					t2 = t0.Add(-2 * time.Millisecond)
				})
				It("should approximate", func() {
					Ω(t0).Should(BeTemporally("~", t0))
					Ω(t0).Should(BeTemporally("~", t1))
					Ω(t0).ShouldNot(BeTemporally("~", t2))
				})
			})

			Context("and there is a precision parameter", func() {
				BeforeEach(func() {
					t2 = t0.Add(3 * time.Second)
				})
				It("should use precision paramter", func() {
					d := 2 * time.Second
					Ω(t0).Should(BeTemporally("~", t0, d))
					Ω(t0).Should(BeTemporally("~", t1, d))
					Ω(t0).ShouldNot(BeTemporally("~", t2, d))
				})
			})
		})
	})

	Context("when passed a non-time", func() {
		It("should error", func() {
			success, err := (&BeTemporallyMatcher{Comparator: "==", CompareTo: t0}).Match("foo")
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeTemporallyMatcher{Comparator: "=="}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("when passed an unsupported comparator", func() {
		It("should error", func() {
			success, err := (&BeTemporallyMatcher{Comparator: "!=", CompareTo: t0}).Match(t2)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
