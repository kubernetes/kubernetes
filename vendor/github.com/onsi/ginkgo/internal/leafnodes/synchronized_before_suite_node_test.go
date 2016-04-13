package leafnodes_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/leafnodes"
	. "github.com/onsi/gomega"

	"github.com/onsi/gomega/ghttp"
	"net/http"

	"github.com/onsi/ginkgo/internal/codelocation"
	Failer "github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
	"time"
)

var _ = Describe("SynchronizedBeforeSuiteNode", func() {
	var failer *Failer.Failer
	var node SuiteNode
	var codeLocation types.CodeLocation
	var innerCodeLocation types.CodeLocation
	var outcome bool
	var server *ghttp.Server

	BeforeEach(func() {
		server = ghttp.NewServer()
		codeLocation = codelocation.New(0)
		innerCodeLocation = codelocation.New(0)
		failer = Failer.New()
	})

	AfterEach(func() {
		server.Close()
	})

	newNode := func(bodyA interface{}, bodyB interface{}) SuiteNode {
		return NewSynchronizedBeforeSuiteNode(bodyA, bodyB, codeLocation, time.Millisecond, failer)
	}

	Describe("when not running in parallel", func() {
		Context("when all is well", func() {
			var data []byte
			BeforeEach(func() {
				data = nil

				node = newNode(func() []byte {
					return []byte("my data")
				}, func(d []byte) {
					data = d
				})

				outcome = node.Run(1, 1, server.URL())
			})

			It("should run A, then B passing the output from A to B", func() {
				Ω(data).Should(Equal([]byte("my data")))
			})

			It("should report success", func() {
				Ω(outcome).Should(BeTrue())
				Ω(node.Passed()).Should(BeTrue())
				Ω(node.Summary().State).Should(Equal(types.SpecStatePassed))
			})
		})

		Context("when A fails", func() {
			var ranB bool
			BeforeEach(func() {
				ranB = false
				node = newNode(func() []byte {
					failer.Fail("boom", innerCodeLocation)
					return nil
				}, func([]byte) {
					ranB = true
				})

				outcome = node.Run(1, 1, server.URL())
			})

			It("should not run B", func() {
				Ω(ranB).Should(BeFalse())
			})

			It("should report failure", func() {
				Ω(outcome).Should(BeFalse())
				Ω(node.Passed()).Should(BeFalse())
				Ω(node.Summary().State).Should(Equal(types.SpecStateFailed))
			})
		})

		Context("when B fails", func() {
			BeforeEach(func() {
				node = newNode(func() []byte {
					return nil
				}, func([]byte) {
					failer.Fail("boom", innerCodeLocation)
				})

				outcome = node.Run(1, 1, server.URL())
			})

			It("should report failure", func() {
				Ω(outcome).Should(BeFalse())
				Ω(node.Passed()).Should(BeFalse())
				Ω(node.Summary().State).Should(Equal(types.SpecStateFailed))
			})
		})

		Context("when A times out", func() {
			var ranB bool
			BeforeEach(func() {
				ranB = false
				node = newNode(func(Done) []byte {
					time.Sleep(time.Second)
					return nil
				}, func([]byte) {
					ranB = true
				})

				outcome = node.Run(1, 1, server.URL())
			})

			It("should not run B", func() {
				Ω(ranB).Should(BeFalse())
			})

			It("should report failure", func() {
				Ω(outcome).Should(BeFalse())
				Ω(node.Passed()).Should(BeFalse())
				Ω(node.Summary().State).Should(Equal(types.SpecStateTimedOut))
			})
		})

		Context("when B times out", func() {
			BeforeEach(func() {
				node = newNode(func() []byte {
					return nil
				}, func([]byte, Done) {
					time.Sleep(time.Second)
				})

				outcome = node.Run(1, 1, server.URL())
			})

			It("should report failure", func() {
				Ω(outcome).Should(BeFalse())
				Ω(node.Passed()).Should(BeFalse())
				Ω(node.Summary().State).Should(Equal(types.SpecStateTimedOut))
			})
		})
	})

	Describe("when running in parallel", func() {
		var ranB bool
		var parallelNode, parallelTotal int
		BeforeEach(func() {
			ranB = false
			parallelNode, parallelTotal = 1, 3
		})

		Context("as the first node, it runs A", func() {
			var expectedState types.RemoteBeforeSuiteData

			BeforeEach(func() {
				parallelNode, parallelTotal = 1, 3
			})

			JustBeforeEach(func() {
				server.AppendHandlers(ghttp.CombineHandlers(
					ghttp.VerifyRequest("POST", "/BeforeSuiteState"),
					ghttp.VerifyJSONRepresenting(expectedState),
				))

				outcome = node.Run(parallelNode, parallelTotal, server.URL())
			})

			Context("when A succeeds", func() {
				BeforeEach(func() {
					expectedState = types.RemoteBeforeSuiteData{[]byte("my data"), types.RemoteBeforeSuiteStatePassed}

					node = newNode(func() []byte {
						return []byte("my data")
					}, func([]byte) {
						ranB = true
					})
				})

				It("should post about A succeeding", func() {
					Ω(server.ReceivedRequests()).Should(HaveLen(1))
				})

				It("should run B", func() {
					Ω(ranB).Should(BeTrue())
				})

				It("should report success", func() {
					Ω(outcome).Should(BeTrue())
				})
			})

			Context("when A fails", func() {
				BeforeEach(func() {
					expectedState = types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStateFailed}

					node = newNode(func() []byte {
						panic("BAM")
						return []byte("my data")
					}, func([]byte) {
						ranB = true
					})
				})

				It("should post about A failing", func() {
					Ω(server.ReceivedRequests()).Should(HaveLen(1))
				})

				It("should not run B", func() {
					Ω(ranB).Should(BeFalse())
				})

				It("should report failure", func() {
					Ω(outcome).Should(BeFalse())
				})
			})
		})

		Context("as the Nth node", func() {
			var statusCode int
			var response interface{}
			var ranA bool
			var bData []byte

			BeforeEach(func() {
				ranA = false
				bData = nil

				statusCode = http.StatusOK

				server.AppendHandlers(ghttp.CombineHandlers(
					ghttp.VerifyRequest("GET", "/BeforeSuiteState"),
					ghttp.RespondWith(http.StatusOK, string((types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStatePending}).ToJSON())),
				), ghttp.CombineHandlers(
					ghttp.VerifyRequest("GET", "/BeforeSuiteState"),
					ghttp.RespondWith(http.StatusOK, string((types.RemoteBeforeSuiteData{nil, types.RemoteBeforeSuiteStatePending}).ToJSON())),
				), ghttp.CombineHandlers(
					ghttp.VerifyRequest("GET", "/BeforeSuiteState"),
					ghttp.RespondWithJSONEncodedPtr(&statusCode, &response),
				))

				node = newNode(func() []byte {
					ranA = true
					return nil
				}, func(data []byte) {
					bData = data
				})

				parallelNode, parallelTotal = 2, 3
			})

			Context("when A on node1 succeeds", func() {
				BeforeEach(func() {
					response = types.RemoteBeforeSuiteData{[]byte("my data"), types.RemoteBeforeSuiteStatePassed}
					outcome = node.Run(parallelNode, parallelTotal, server.URL())
				})

				It("should not run A", func() {
					Ω(ranA).Should(BeFalse())
				})

				It("should poll for A", func() {
					Ω(server.ReceivedRequests()).Should(HaveLen(3))
				})

				It("should run B when the polling succeeds", func() {
					Ω(bData).Should(Equal([]byte("my data")))
				})

				It("should succeed", func() {
					Ω(outcome).Should(BeTrue())
					Ω(node.Passed()).Should(BeTrue())
				})
			})

			Context("when A on node1 fails", func() {
				BeforeEach(func() {
					response = types.RemoteBeforeSuiteData{[]byte("my data"), types.RemoteBeforeSuiteStateFailed}
					outcome = node.Run(parallelNode, parallelTotal, server.URL())
				})

				It("should not run A", func() {
					Ω(ranA).Should(BeFalse())
				})

				It("should poll for A", func() {
					Ω(server.ReceivedRequests()).Should(HaveLen(3))
				})

				It("should not run B", func() {
					Ω(bData).Should(BeNil())
				})

				It("should fail", func() {
					Ω(outcome).Should(BeFalse())
					Ω(node.Passed()).Should(BeFalse())

					summary := node.Summary()
					Ω(summary.State).Should(Equal(types.SpecStateFailed))
					Ω(summary.Failure.Message).Should(Equal("BeforeSuite on Node 1 failed"))
					Ω(summary.Failure.Location).Should(Equal(codeLocation))
					Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeBeforeSuite))
					Ω(summary.Failure.ComponentIndex).Should(Equal(0))
					Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
				})
			})

			Context("when node1 disappears", func() {
				BeforeEach(func() {
					response = types.RemoteBeforeSuiteData{[]byte("my data"), types.RemoteBeforeSuiteStateDisappeared}
					outcome = node.Run(parallelNode, parallelTotal, server.URL())
				})

				It("should not run A", func() {
					Ω(ranA).Should(BeFalse())
				})

				It("should poll for A", func() {
					Ω(server.ReceivedRequests()).Should(HaveLen(3))
				})

				It("should not run B", func() {
					Ω(bData).Should(BeNil())
				})

				It("should fail", func() {
					Ω(outcome).Should(BeFalse())
					Ω(node.Passed()).Should(BeFalse())

					summary := node.Summary()
					Ω(summary.State).Should(Equal(types.SpecStateFailed))
					Ω(summary.Failure.Message).Should(Equal("Node 1 disappeared before completing BeforeSuite"))
					Ω(summary.Failure.Location).Should(Equal(codeLocation))
					Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeBeforeSuite))
					Ω(summary.Failure.ComponentIndex).Should(Equal(0))
					Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
				})
			})
		})
	})

	Describe("construction", func() {
		Describe("the first function", func() {
			Context("when the first function returns a byte array", func() {
				Context("and takes nothing", func() {
					It("should be fine", func() {
						Ω(func() {
							newNode(func() []byte { return nil }, func([]byte) {})
						}).ShouldNot(Panic())
					})
				})

				Context("and takes a done function", func() {
					It("should be fine", func() {
						Ω(func() {
							newNode(func(Done) []byte { return nil }, func([]byte) {})
						}).ShouldNot(Panic())
					})
				})

				Context("and takes more than one thing", func() {
					It("should panic", func() {
						Ω(func() {
							newNode(func(Done, Done) []byte { return nil }, func([]byte) {})
						}).Should(Panic())
					})
				})

				Context("and takes something else", func() {
					It("should panic", func() {
						Ω(func() {
							newNode(func(bool) []byte { return nil }, func([]byte) {})
						}).Should(Panic())
					})
				})
			})

			Context("when the first function does not return a byte array", func() {
				It("should panic", func() {
					Ω(func() {
						newNode(func() {}, func([]byte) {})
					}).Should(Panic())

					Ω(func() {
						newNode(func() []int { return nil }, func([]byte) {})
					}).Should(Panic())
				})
			})
		})

		Describe("the second function", func() {
			Context("when the second function takes a byte array", func() {
				It("should be fine", func() {
					Ω(func() {
						newNode(func() []byte { return nil }, func([]byte) {})
					}).ShouldNot(Panic())
				})
			})

			Context("when it also takes a done channel", func() {
				It("should be fine", func() {
					Ω(func() {
						newNode(func() []byte { return nil }, func([]byte, Done) {})
					}).ShouldNot(Panic())
				})
			})

			Context("if it takes anything else", func() {
				It("should panic", func() {
					Ω(func() {
						newNode(func() []byte { return nil }, func([]byte, chan bool) {})
					}).Should(Panic())

					Ω(func() {
						newNode(func() []byte { return nil }, func(string) {})
					}).Should(Panic())
				})
			})

			Context("if it takes nothing at all", func() {
				It("should panic", func() {
					Ω(func() {
						newNode(func() []byte { return nil }, func() {})
					}).Should(Panic())
				})
			})

			Context("if it returns something", func() {
				It("should panic", func() {
					Ω(func() {
						newNode(func() []byte { return nil }, func([]byte) []byte { return nil })
					}).Should(Panic())
				})
			})
		})
	})
})
