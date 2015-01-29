package remote_test

import (
	"encoding/json"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	. "github.com/onsi/ginkgo/internal/remote"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
)

var _ = Describe("ForwardingReporter", func() {
	var (
		reporter     *ForwardingReporter
		interceptor  *fakeOutputInterceptor
		poster       *fakePoster
		suiteSummary *types.SuiteSummary
		specSummary  *types.SpecSummary
		setupSummary *types.SetupSummary
		serverHost   string
	)

	BeforeEach(func() {
		serverHost = "http://127.0.0.1:7788"

		poster = newFakePoster()

		interceptor = &fakeOutputInterceptor{
			InterceptedOutput: "The intercepted output!",
		}

		reporter = NewForwardingReporter(serverHost, poster, interceptor)

		suiteSummary = &types.SuiteSummary{
			SuiteDescription: "My Test Suite",
		}

		setupSummary = &types.SetupSummary{
			State: types.SpecStatePassed,
		}

		specSummary = &types.SpecSummary{
			ComponentTexts: []string{"My", "Spec"},
			State:          types.SpecStatePassed,
		}
	})

	Context("When a suite begins", func() {
		BeforeEach(func() {
			reporter.SpecSuiteWillBegin(config.GinkgoConfig, suiteSummary)
		})

		It("should start intercepting output", func() {
			Ω(interceptor.DidStartInterceptingOutput).Should(BeTrue())
		})

		It("should POST the SuiteSummary and Ginkgo Config to the Ginkgo server", func() {
			Ω(poster.posts).Should(HaveLen(1))
			Ω(poster.posts[0].url).Should(Equal("http://127.0.0.1:7788/SpecSuiteWillBegin"))
			Ω(poster.posts[0].bodyType).Should(Equal("application/json"))

			var sentData struct {
				SentConfig       config.GinkgoConfigType `json:"config"`
				SentSuiteSummary *types.SuiteSummary     `json:"suite-summary"`
			}

			err := json.Unmarshal(poster.posts[0].bodyContent, &sentData)
			Ω(err).ShouldNot(HaveOccurred())

			Ω(sentData.SentConfig).Should(Equal(config.GinkgoConfig))
			Ω(sentData.SentSuiteSummary).Should(Equal(suiteSummary))
		})
	})

	Context("when a BeforeSuite completes", func() {
		BeforeEach(func() {
			reporter.BeforeSuiteDidRun(setupSummary)
		})

		It("should stop, then start intercepting output", func() {
			Ω(interceptor.DidStopInterceptingOutput).Should(BeTrue())
			Ω(interceptor.DidStartInterceptingOutput).Should(BeTrue())
		})

		It("should POST the SetupSummary to the Ginkgo server", func() {
			Ω(poster.posts).Should(HaveLen(1))
			Ω(poster.posts[0].url).Should(Equal("http://127.0.0.1:7788/BeforeSuiteDidRun"))
			Ω(poster.posts[0].bodyType).Should(Equal("application/json"))

			var summary *types.SetupSummary
			err := json.Unmarshal(poster.posts[0].bodyContent, &summary)
			Ω(err).ShouldNot(HaveOccurred())
			setupSummary.CapturedOutput = interceptor.InterceptedOutput
			Ω(summary).Should(Equal(setupSummary))
		})
	})

	Context("when an AfterSuite completes", func() {
		BeforeEach(func() {
			reporter.AfterSuiteDidRun(setupSummary)
		})

		It("should stop, then start intercepting output", func() {
			Ω(interceptor.DidStopInterceptingOutput).Should(BeTrue())
			Ω(interceptor.DidStartInterceptingOutput).Should(BeTrue())
		})

		It("should POST the SetupSummary to the Ginkgo server", func() {
			Ω(poster.posts).Should(HaveLen(1))
			Ω(poster.posts[0].url).Should(Equal("http://127.0.0.1:7788/AfterSuiteDidRun"))
			Ω(poster.posts[0].bodyType).Should(Equal("application/json"))

			var summary *types.SetupSummary
			err := json.Unmarshal(poster.posts[0].bodyContent, &summary)
			Ω(err).ShouldNot(HaveOccurred())
			setupSummary.CapturedOutput = interceptor.InterceptedOutput
			Ω(summary).Should(Equal(setupSummary))
		})
	})

	Context("When a spec will run", func() {
		BeforeEach(func() {
			reporter.SpecWillRun(specSummary)
		})

		It("should POST the SpecSummary to the Ginkgo server", func() {
			Ω(poster.posts).Should(HaveLen(1))
			Ω(poster.posts[0].url).Should(Equal("http://127.0.0.1:7788/SpecWillRun"))
			Ω(poster.posts[0].bodyType).Should(Equal("application/json"))

			var summary *types.SpecSummary
			err := json.Unmarshal(poster.posts[0].bodyContent, &summary)
			Ω(err).ShouldNot(HaveOccurred())
			Ω(summary).Should(Equal(specSummary))
		})

		Context("When a spec completes", func() {
			BeforeEach(func() {
				specSummary.State = types.SpecStatePanicked
				reporter.SpecDidComplete(specSummary)
			})

			It("should POST the SpecSummary to the Ginkgo server and include any intercepted output", func() {
				Ω(poster.posts).Should(HaveLen(2))
				Ω(poster.posts[1].url).Should(Equal("http://127.0.0.1:7788/SpecDidComplete"))
				Ω(poster.posts[1].bodyType).Should(Equal("application/json"))

				var summary *types.SpecSummary
				err := json.Unmarshal(poster.posts[1].bodyContent, &summary)
				Ω(err).ShouldNot(HaveOccurred())
				specSummary.CapturedOutput = interceptor.InterceptedOutput
				Ω(summary).Should(Equal(specSummary))
			})

			It("should stop, then start intercepting output", func() {
				Ω(interceptor.DidStopInterceptingOutput).Should(BeTrue())
				Ω(interceptor.DidStartInterceptingOutput).Should(BeTrue())
			})
		})
	})

	Context("When a suite ends", func() {
		BeforeEach(func() {
			reporter.SpecSuiteDidEnd(suiteSummary)
		})

		It("should POST the SuiteSummary to the Ginkgo server", func() {
			Ω(poster.posts).Should(HaveLen(1))
			Ω(poster.posts[0].url).Should(Equal("http://127.0.0.1:7788/SpecSuiteDidEnd"))
			Ω(poster.posts[0].bodyType).Should(Equal("application/json"))

			var summary *types.SuiteSummary

			err := json.Unmarshal(poster.posts[0].bodyContent, &summary)
			Ω(err).ShouldNot(HaveOccurred())

			Ω(summary).Should(Equal(suiteSummary))
		})
	})
})
