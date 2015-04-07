package remote_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/remote"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/types"

	"bytes"
	"encoding/json"
	"net/http"
)

var _ = Describe("Server", func() {
	var (
		server *Server
	)

	BeforeEach(func() {
		var err error
		server, err = NewServer(3)
		Ω(err).ShouldNot(HaveOccurred())

		server.Start()
	})

	AfterEach(func() {
		server.Close()
	})

	Describe("Streaming endpoints", func() {
		var (
			reporterA, reporterB *reporters.FakeReporter
			forwardingReporter   *ForwardingReporter

			suiteSummary *types.SuiteSummary
			setupSummary *types.SetupSummary
			specSummary  *types.SpecSummary
		)

		BeforeEach(func() {
			reporterA = reporters.NewFakeReporter()
			reporterB = reporters.NewFakeReporter()

			server.RegisterReporters(reporterA, reporterB)

			forwardingReporter = NewForwardingReporter(server.Address(), &http.Client{}, &fakeOutputInterceptor{})

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

		It("should make its address available", func() {
			Ω(server.Address()).Should(MatchRegexp(`http://127.0.0.1:\d{2,}`))
		})

		Describe("/SpecSuiteWillBegin", func() {
			It("should decode and forward the Ginkgo config and suite summary", func(done Done) {
				forwardingReporter.SpecSuiteWillBegin(config.GinkgoConfig, suiteSummary)
				Ω(reporterA.Config).Should(Equal(config.GinkgoConfig))
				Ω(reporterB.Config).Should(Equal(config.GinkgoConfig))
				Ω(reporterA.BeginSummary).Should(Equal(suiteSummary))
				Ω(reporterB.BeginSummary).Should(Equal(suiteSummary))
				close(done)
			})
		})

		Describe("/BeforeSuiteDidRun", func() {
			It("should decode and forward the setup summary", func() {
				forwardingReporter.BeforeSuiteDidRun(setupSummary)
				Ω(reporterA.BeforeSuiteSummary).Should(Equal(setupSummary))
				Ω(reporterB.BeforeSuiteSummary).Should(Equal(setupSummary))
			})
		})

		Describe("/AfterSuiteDidRun", func() {
			It("should decode and forward the setup summary", func() {
				forwardingReporter.AfterSuiteDidRun(setupSummary)
				Ω(reporterA.AfterSuiteSummary).Should(Equal(setupSummary))
				Ω(reporterB.AfterSuiteSummary).Should(Equal(setupSummary))
			})
		})

		Describe("/SpecWillRun", func() {
			It("should decode and forward the spec summary", func(done Done) {
				forwardingReporter.SpecWillRun(specSummary)
				Ω(reporterA.SpecWillRunSummaries[0]).Should(Equal(specSummary))
				Ω(reporterB.SpecWillRunSummaries[0]).Should(Equal(specSummary))
				close(done)
			})
		})

		Describe("/SpecDidComplete", func() {
			It("should decode and forward the spec summary", func(done Done) {
				forwardingReporter.SpecDidComplete(specSummary)
				Ω(reporterA.SpecSummaries[0]).Should(Equal(specSummary))
				Ω(reporterB.SpecSummaries[0]).Should(Equal(specSummary))
				close(done)
			})
		})

		Describe("/SpecSuiteDidEnd", func() {
			It("should decode and forward the suite summary", func(done Done) {
				forwardingReporter.SpecSuiteDidEnd(suiteSummary)
				Ω(reporterA.EndSummary).Should(Equal(suiteSummary))
				Ω(reporterB.EndSummary).Should(Equal(suiteSummary))
				close(done)
			})
		})
	})

	Describe("Synchronization endpoints", func() {
		Describe("GETting and POSTing BeforeSuiteState", func() {
			getBeforeSuite := func() types.RemoteBeforeSuiteData {
				resp, err := http.Get(server.Address() + "/BeforeSuiteState")
				Ω(err).ShouldNot(HaveOccurred())
				Ω(resp.StatusCode).Should(Equal(http.StatusOK))

				r := types.RemoteBeforeSuiteData{}
				decoder := json.NewDecoder(resp.Body)
				err = decoder.Decode(&r)
				Ω(err).ShouldNot(HaveOccurred())

				return r
			}

			postBeforeSuite := func(r types.RemoteBeforeSuiteData) {
				resp, err := http.Post(server.Address()+"/BeforeSuiteState", "application/json", bytes.NewReader(r.ToJSON()))
				Ω(err).ShouldNot(HaveOccurred())
				Ω(resp.StatusCode).Should(Equal(http.StatusOK))
			}

			Context("when the first node's Alive has not been registered yet", func() {
				It("should return pending", func() {
					state := getBeforeSuite()
					Ω(state).Should(Equal(types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStatePending}))

					state = getBeforeSuite()
					Ω(state).Should(Equal(types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStatePending}))
				})
			})

			Context("when the first node is Alive but has not responded yet", func() {
				BeforeEach(func() {
					server.RegisterAlive(1, func() bool {
						return true
					})
				})

				It("should return pending", func() {
					state := getBeforeSuite()
					Ω(state).Should(Equal(types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStatePending}))

					state = getBeforeSuite()
					Ω(state).Should(Equal(types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStatePending}))
				})
			})

			Context("when the first node has responded", func() {
				var state types.RemoteBeforeSuiteData
				BeforeEach(func() {
					server.RegisterAlive(1, func() bool {
						return false
					})

					state = types.RemoteBeforeSuiteData{
						Data:  []byte("my data"),
						State: types.RemoteBeforeSuiteStatePassed,
					}
					postBeforeSuite(state)
				})

				It("should return the passed in state", func() {
					returnedState := getBeforeSuite()
					Ω(returnedState).Should(Equal(state))
				})
			})

			Context("when the first node is no longer Alive and has not responded yet", func() {
				BeforeEach(func() {
					server.RegisterAlive(1, func() bool {
						return false
					})
				})

				It("should return disappeared", func() {
					state := getBeforeSuite()
					Ω(state).Should(Equal(types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStateDisappeared}))

					state = getBeforeSuite()
					Ω(state).Should(Equal(types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStateDisappeared}))
				})
			})
		})

		Describe("GETting RemoteAfterSuiteData", func() {
			getRemoteAfterSuiteData := func() bool {
				resp, err := http.Get(server.Address() + "/RemoteAfterSuiteData")
				Ω(err).ShouldNot(HaveOccurred())
				Ω(resp.StatusCode).Should(Equal(http.StatusOK))

				a := types.RemoteAfterSuiteData{}
				decoder := json.NewDecoder(resp.Body)
				err = decoder.Decode(&a)
				Ω(err).ShouldNot(HaveOccurred())

				return a.CanRun
			}

			Context("when there are unregistered nodes", func() {
				BeforeEach(func() {
					server.RegisterAlive(2, func() bool {
						return false
					})
				})

				It("should return false", func() {
					Ω(getRemoteAfterSuiteData()).Should(BeFalse())
				})
			})

			Context("when all none-node-1 nodes are still running", func() {
				BeforeEach(func() {
					server.RegisterAlive(2, func() bool {
						return true
					})

					server.RegisterAlive(3, func() bool {
						return false
					})
				})

				It("should return false", func() {
					Ω(getRemoteAfterSuiteData()).Should(BeFalse())
				})
			})

			Context("when all none-1 nodes are done", func() {
				BeforeEach(func() {
					server.RegisterAlive(2, func() bool {
						return false
					})

					server.RegisterAlive(3, func() bool {
						return false
					})
				})

				It("should return true", func() {
					Ω(getRemoteAfterSuiteData()).Should(BeTrue())
				})

			})
		})
	})
})
