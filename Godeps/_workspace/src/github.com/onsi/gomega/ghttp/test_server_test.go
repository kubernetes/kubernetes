package ghttp_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"regexp"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/ghttp"
)

var _ = Describe("TestServer", func() {
	var (
		resp *http.Response
		err  error
		s    *Server
	)

	BeforeEach(func() {
		s = NewServer()
	})

	AfterEach(func() {
		s.Close()
	})

	Describe("closing client connections", func() {
		It("closes", func() {
			s.AppendHandlers(
				func(w http.ResponseWriter, req *http.Request) {
					w.Write([]byte("hello"))
				},
				func(w http.ResponseWriter, req *http.Request) {
					s.CloseClientConnections()
				},
			)

			resp, err := http.Get(s.URL())
			Ω(err).ShouldNot(HaveOccurred())
			Ω(resp.StatusCode).Should(Equal(200))

			body, err := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			Ω(err).ShouldNot(HaveOccurred())
			Ω(body).Should(Equal([]byte("hello")))

			resp, err = http.Get(s.URL())
			Ω(err).Should(HaveOccurred())
			Ω(resp).Should(BeNil())
		})
	})

	Describe("allowing unhandled requests", func() {
		Context("when true", func() {
			BeforeEach(func() {
				s.AllowUnhandledRequests = true
				s.UnhandledRequestStatusCode = http.StatusForbidden
				resp, err = http.Get(s.URL() + "/foo")
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should allow unhandled requests and respond with the passed in status code", func() {
				Ω(err).ShouldNot(HaveOccurred())
				Ω(resp.StatusCode).Should(Equal(http.StatusForbidden))

				data, err := ioutil.ReadAll(resp.Body)
				Ω(err).ShouldNot(HaveOccurred())
				Ω(data).Should(BeEmpty())
			})

			It("should record the requests", func() {
				Ω(s.ReceivedRequests()).Should(HaveLen(1))
				Ω(s.ReceivedRequests()[0].URL.Path).Should(Equal("/foo"))
			})
		})

		Context("when false", func() {
			It("should fail when attempting a request", func() {
				failures := InterceptGomegaFailures(func() {
					http.Get(s.URL() + "/foo")
				})

				Ω(failures[0]).Should(ContainSubstring("Received Unhandled Request"))
			})
		})
	})

	Describe("Managing Handlers", func() {
		var called []string
		BeforeEach(func() {
			called = []string{}
			s.RouteToHandler("GET", "/routed", func(w http.ResponseWriter, req *http.Request) {
				called = append(called, "r1")
			})
			s.RouteToHandler("POST", regexp.MustCompile(`/routed\d`), func(w http.ResponseWriter, req *http.Request) {
				called = append(called, "r2")
			})
			s.AppendHandlers(func(w http.ResponseWriter, req *http.Request) {
				called = append(called, "A")
			}, func(w http.ResponseWriter, req *http.Request) {
				called = append(called, "B")
			})
		})

		It("should prefer routed handlers if there is a match", func() {
			http.Get(s.URL() + "/routed")
			http.Post(s.URL()+"/routed7", "application/json", nil)
			http.Get(s.URL() + "/foo")
			http.Get(s.URL() + "/routed")
			http.Post(s.URL()+"/routed9", "application/json", nil)
			http.Get(s.URL() + "/bar")

			failures := InterceptGomegaFailures(func() {
				http.Get(s.URL() + "/foo")
				http.Get(s.URL() + "/routed/not/a/match")
				http.Get(s.URL() + "/routed7")
				http.Post(s.URL()+"/routed", "application/json", nil)
			})

			Ω(failures[0]).Should(ContainSubstring("Received Unhandled Request"))
			Ω(failures).Should(HaveLen(4))

			http.Post(s.URL()+"/routed3", "application/json", nil)

			Ω(called).Should(Equal([]string{"r1", "r2", "A", "r1", "r2", "B", "r2"}))
		})

		It("should override routed handlers when reregistered", func() {
			s.RouteToHandler("GET", "/routed", func(w http.ResponseWriter, req *http.Request) {
				called = append(called, "r3")
			})
			s.RouteToHandler("POST", regexp.MustCompile(`/routed\d`), func(w http.ResponseWriter, req *http.Request) {
				called = append(called, "r4")
			})

			http.Get(s.URL() + "/routed")
			http.Post(s.URL()+"/routed7", "application/json", nil)

			Ω(called).Should(Equal([]string{"r3", "r4"}))
		})

		It("should call the appended handlers, in order, as requests come in", func() {
			http.Get(s.URL() + "/foo")
			Ω(called).Should(Equal([]string{"A"}))

			http.Get(s.URL() + "/foo")
			Ω(called).Should(Equal([]string{"A", "B"}))

			failures := InterceptGomegaFailures(func() {
				http.Get(s.URL() + "/foo")
			})

			Ω(failures[0]).Should(ContainSubstring("Received Unhandled Request"))
		})

		Describe("Overwriting an existing handler", func() {
			BeforeEach(func() {
				s.SetHandler(0, func(w http.ResponseWriter, req *http.Request) {
					called = append(called, "C")
				})
			})

			It("should override the specified handler", func() {
				http.Get(s.URL() + "/foo")
				http.Get(s.URL() + "/foo")
				Ω(called).Should(Equal([]string{"C", "B"}))
			})
		})

		Describe("Getting an existing handler", func() {
			It("should return the handler func", func() {
				s.GetHandler(1)(nil, nil)
				Ω(called).Should(Equal([]string{"B"}))
			})
		})

		Describe("Wrapping an existing handler", func() {
			BeforeEach(func() {
				s.WrapHandler(0, func(w http.ResponseWriter, req *http.Request) {
					called = append(called, "C")
				})
			})

			It("should wrap the existing handler in a new handler", func() {
				http.Get(s.URL() + "/foo")
				http.Get(s.URL() + "/foo")
				Ω(called).Should(Equal([]string{"A", "C", "B"}))
			})
		})
	})

	Describe("Request Handlers", func() {
		Describe("VerifyRequest", func() {
			BeforeEach(func() {
				s.AppendHandlers(VerifyRequest("GET", "/foo"))
			})

			It("should verify the method, path", func() {
				resp, err = http.Get(s.URL() + "/foo?baz=bar")
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify the method, path", func() {
				failures := InterceptGomegaFailures(func() {
					http.Get(s.URL() + "/foo2")
				})
				Ω(failures).Should(HaveLen(1))
			})

			It("should verify the method, path", func() {
				failures := InterceptGomegaFailures(func() {
					http.Post(s.URL()+"/foo", "application/json", nil)
				})
				Ω(failures).Should(HaveLen(1))
			})

			Context("when passed a rawQuery", func() {
				It("should also be possible to verify the rawQuery", func() {
					s.SetHandler(0, VerifyRequest("GET", "/foo", "baz=bar"))
					resp, err = http.Get(s.URL() + "/foo?baz=bar")
					Ω(err).ShouldNot(HaveOccurred())
				})
			})

			Context("when passed a matcher for path", func() {
				It("should apply the matcher", func() {
					s.SetHandler(0, VerifyRequest("GET", MatchRegexp(`/foo/[a-f]*/3`)))
					resp, err = http.Get(s.URL() + "/foo/abcdefa/3")
					Ω(err).ShouldNot(HaveOccurred())
				})
			})
		})

		Describe("VerifyContentType", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("GET", "/foo"),
					VerifyContentType("application/octet-stream"),
				))
			})

			It("should verify the content type", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/octet-stream")

				resp, err = http.DefaultClient.Do(req)
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify the content type", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				failures := InterceptGomegaFailures(func() {
					http.DefaultClient.Do(req)
				})
				Ω(failures).Should(HaveLen(1))
			})
		})

		Describe("Verify BasicAuth", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("GET", "/foo"),
					VerifyBasicAuth("bob", "password"),
				))
			})

			It("should verify basic auth", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.SetBasicAuth("bob", "password")

				resp, err = http.DefaultClient.Do(req)
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify basic auth", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.SetBasicAuth("bob", "bassword")

				failures := InterceptGomegaFailures(func() {
					http.DefaultClient.Do(req)
				})
				Ω(failures).Should(HaveLen(1))
			})

			It("should require basic auth header", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())

				failures := InterceptGomegaFailures(func() {
					http.DefaultClient.Do(req)
				})
				Ω(failures).Should(HaveLen(1))
			})
		})

		Describe("VerifyHeader", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("GET", "/foo"),
					VerifyHeader(http.Header{
						"accept":        []string{"jpeg", "png"},
						"cache-control": []string{"omicron"},
						"Return-Path":   []string{"hobbiton"},
					}),
				))
			})

			It("should verify the headers", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.Header.Add("Accept", "jpeg")
				req.Header.Add("Accept", "png")
				req.Header.Add("Cache-Control", "omicron")
				req.Header.Add("return-path", "hobbiton")

				resp, err = http.DefaultClient.Do(req)
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify the headers", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.Header.Add("Schmaccept", "jpeg")
				req.Header.Add("Schmaccept", "png")
				req.Header.Add("Cache-Control", "omicron")
				req.Header.Add("return-path", "hobbiton")

				failures := InterceptGomegaFailures(func() {
					http.DefaultClient.Do(req)
				})
				Ω(failures).Should(HaveLen(1))
			})
		})

		Describe("VerifyHeaderKV", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("GET", "/foo"),
					VerifyHeaderKV("accept", "jpeg", "png"),
					VerifyHeaderKV("cache-control", "omicron"),
					VerifyHeaderKV("Return-Path", "hobbiton"),
				))
			})

			It("should verify the headers", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.Header.Add("Accept", "jpeg")
				req.Header.Add("Accept", "png")
				req.Header.Add("Cache-Control", "omicron")
				req.Header.Add("return-path", "hobbiton")

				resp, err = http.DefaultClient.Do(req)
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify the headers", func() {
				req, err := http.NewRequest("GET", s.URL()+"/foo", nil)
				Ω(err).ShouldNot(HaveOccurred())
				req.Header.Add("Accept", "jpeg")
				req.Header.Add("Cache-Control", "omicron")
				req.Header.Add("return-path", "hobbiton")

				failures := InterceptGomegaFailures(func() {
					http.DefaultClient.Do(req)
				})
				Ω(failures).Should(HaveLen(1))
			})
		})

		Describe("VerifyJSON", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("POST", "/foo"),
					VerifyJSON(`{"a":3, "b":2}`),
				))
			})

			It("should verify the json body and the content type", func() {
				resp, err = http.Post(s.URL()+"/foo", "application/json", bytes.NewReader([]byte(`{"b":2, "a":3}`)))
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify the json body and the content type", func() {
				failures := InterceptGomegaFailures(func() {
					http.Post(s.URL()+"/foo", "application/json", bytes.NewReader([]byte(`{"b":2, "a":4}`)))
				})
				Ω(failures).Should(HaveLen(1))
			})

			It("should verify the json body and the content type", func() {
				failures := InterceptGomegaFailures(func() {
					http.Post(s.URL()+"/foo", "application/not-json", bytes.NewReader([]byte(`{"b":2, "a":3}`)))
				})
				Ω(failures).Should(HaveLen(1))
			})
		})

		Describe("VerifyJSONRepresenting", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("POST", "/foo"),
					VerifyJSONRepresenting([]int{1, 3, 5}),
				))
			})

			It("should verify the json body and the content type", func() {
				resp, err = http.Post(s.URL()+"/foo", "application/json", bytes.NewReader([]byte(`[1,3,5]`)))
				Ω(err).ShouldNot(HaveOccurred())
			})

			It("should verify the json body and the content type", func() {
				failures := InterceptGomegaFailures(func() {
					http.Post(s.URL()+"/foo", "application/json", bytes.NewReader([]byte(`[1,3]`)))
				})
				Ω(failures).Should(HaveLen(1))
			})
		})

		Describe("RespondWith", func() {
			Context("without headers", func() {
				BeforeEach(func() {
					s.AppendHandlers(CombineHandlers(
						VerifyRequest("POST", "/foo"),
						RespondWith(http.StatusCreated, "sweet"),
					), CombineHandlers(
						VerifyRequest("POST", "/foo"),
						RespondWith(http.StatusOK, []byte("sour")),
					))
				})

				It("should return the response", func() {
					resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
					Ω(err).ShouldNot(HaveOccurred())

					Ω(resp.StatusCode).Should(Equal(http.StatusCreated))

					body, err := ioutil.ReadAll(resp.Body)
					Ω(err).ShouldNot(HaveOccurred())
					Ω(body).Should(Equal([]byte("sweet")))

					resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
					Ω(err).ShouldNot(HaveOccurred())

					Ω(resp.StatusCode).Should(Equal(http.StatusOK))

					body, err = ioutil.ReadAll(resp.Body)
					Ω(err).ShouldNot(HaveOccurred())
					Ω(body).Should(Equal([]byte("sour")))
				})
			})

			Context("with headers", func() {
				BeforeEach(func() {
					s.AppendHandlers(CombineHandlers(
						VerifyRequest("POST", "/foo"),
						RespondWith(http.StatusCreated, "sweet", http.Header{"X-Custom-Header": []string{"my header"}}),
					))
				})

				It("should return the headers too", func() {
					resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
					Ω(err).ShouldNot(HaveOccurred())

					Ω(resp.StatusCode).Should(Equal(http.StatusCreated))
					Ω(ioutil.ReadAll(resp.Body)).Should(Equal([]byte("sweet")))
					Ω(resp.Header.Get("X-Custom-Header")).Should(Equal("my header"))
				})
			})
		})

		Describe("RespondWithPtr", func() {
			var code int
			var byteBody []byte
			var stringBody string
			BeforeEach(func() {
				code = http.StatusOK
				byteBody = []byte("sweet")
				stringBody = "sour"

				s.AppendHandlers(CombineHandlers(
					VerifyRequest("POST", "/foo"),
					RespondWithPtr(&code, &byteBody),
				), CombineHandlers(
					VerifyRequest("POST", "/foo"),
					RespondWithPtr(&code, &stringBody),
				))
			})

			It("should return the response", func() {
				code = http.StatusCreated
				byteBody = []byte("tasty")
				stringBody = "treat"

				resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
				Ω(err).ShouldNot(HaveOccurred())

				Ω(resp.StatusCode).Should(Equal(http.StatusCreated))

				body, err := ioutil.ReadAll(resp.Body)
				Ω(err).ShouldNot(HaveOccurred())
				Ω(body).Should(Equal([]byte("tasty")))

				resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
				Ω(err).ShouldNot(HaveOccurred())

				Ω(resp.StatusCode).Should(Equal(http.StatusCreated))

				body, err = ioutil.ReadAll(resp.Body)
				Ω(err).ShouldNot(HaveOccurred())
				Ω(body).Should(Equal([]byte("treat")))
			})

			Context("when passed a nil body", func() {
				BeforeEach(func() {
					s.SetHandler(0, CombineHandlers(
						VerifyRequest("POST", "/foo"),
						RespondWithPtr(&code, nil),
					))
				})

				It("should return an empty body and not explode", func() {
					resp, err = http.Post(s.URL()+"/foo", "application/json", nil)

					Ω(err).ShouldNot(HaveOccurred())
					Ω(resp.StatusCode).Should(Equal(http.StatusOK))
					body, err := ioutil.ReadAll(resp.Body)
					Ω(err).ShouldNot(HaveOccurred())
					Ω(body).Should(BeEmpty())

					Ω(s.ReceivedRequests()).Should(HaveLen(1))
				})
			})
		})

		Describe("RespondWithJSON", func() {
			BeforeEach(func() {
				s.AppendHandlers(CombineHandlers(
					VerifyRequest("POST", "/foo"),
					RespondWithJSONEncoded(http.StatusCreated, []int{1, 2, 3}),
				))
			})

			It("should return the response", func() {
				resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
				Ω(err).ShouldNot(HaveOccurred())

				Ω(resp.StatusCode).Should(Equal(http.StatusCreated))

				body, err := ioutil.ReadAll(resp.Body)
				Ω(err).ShouldNot(HaveOccurred())
				Ω(body).Should(MatchJSON("[1,2,3]"))
			})
		})

		Describe("RespondWithJSONPtr", func() {
			var code int
			var object interface{}
			BeforeEach(func() {
				code = http.StatusOK
				object = []int{1, 2, 3}

				s.AppendHandlers(CombineHandlers(
					VerifyRequest("POST", "/foo"),
					RespondWithJSONEncodedPtr(&code, &object),
				))
			})

			It("should return the response", func() {
				code = http.StatusCreated
				object = []int{4, 5, 6}
				resp, err = http.Post(s.URL()+"/foo", "application/json", nil)
				Ω(err).ShouldNot(HaveOccurred())

				Ω(resp.StatusCode).Should(Equal(http.StatusCreated))

				body, err := ioutil.ReadAll(resp.Body)
				Ω(err).ShouldNot(HaveOccurred())
				Ω(body).Should(MatchJSON("[4,5,6]"))
			})
		})
	})
})
