package spec_iterator_test

import (
	"net/http"

	. "github.com/onsi/ginkgo/internal/spec_iterator"
	"github.com/onsi/gomega/ghttp"

	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/internal/spec"
	"github.com/onsi/ginkgo/types"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ParallelSpecIterator", func() {
	var specs []*spec.Spec
	var iterator *ParallelIterator
	var server *ghttp.Server

	newSpec := func(text string, flag types.FlagType) *spec.Spec {
		subject := leafnodes.NewItNode(text, func() {}, flag, codelocation.New(0), 0, nil, 0)
		return spec.New(subject, []*containernode.ContainerNode{}, false)
	}

	BeforeEach(func() {
		specs = []*spec.Spec{
			newSpec("A", types.FlagTypePending),
			newSpec("B", types.FlagTypeNone),
			newSpec("C", types.FlagTypeNone),
			newSpec("D", types.FlagTypeNone),
		}
		specs[3].Skip()

		server = ghttp.NewServer()

		iterator = NewParallelIterator(specs, "http://"+server.Addr())
	})

	AfterEach(func() {
		server.Close()
	})

	It("should report the total number of specs", func() {
		Ω(iterator.NumberOfSpecsPriorToIteration()).Should(Equal(4))
	})

	It("should not report the number to be processed", func() {
		n, known := iterator.NumberOfSpecsToProcessIfKnown()
		Ω(n).Should(Equal(-1))
		Ω(known).Should(BeFalse())
	})

	It("should not report the number that will be run", func() {
		n, known := iterator.NumberOfSpecsThatWillBeRunIfKnown()
		Ω(n).Should(Equal(-1))
		Ω(known).Should(BeFalse())
	})

	Describe("iterating", func() {
		Describe("when the server returns well-formed responses", func() {
			BeforeEach(func() {
				server.AppendHandlers(
					ghttp.RespondWithJSONEncoded(http.StatusOK, Counter{0}),
					ghttp.RespondWithJSONEncoded(http.StatusOK, Counter{1}),
					ghttp.RespondWithJSONEncoded(http.StatusOK, Counter{3}),
					ghttp.RespondWithJSONEncoded(http.StatusOK, Counter{4}),
				)
			})

			It("should return the specs in question", func() {
				Ω(iterator.Next()).Should(Equal(specs[0]))
				Ω(iterator.Next()).Should(Equal(specs[1]))
				Ω(iterator.Next()).Should(Equal(specs[3]))
				spec, err := iterator.Next()
				Ω(spec).Should(BeNil())
				Ω(err).Should(MatchError(ErrClosed))
			})
		})

		Describe("when the server 404s", func() {
			BeforeEach(func() {
				server.AppendHandlers(
					ghttp.RespondWith(http.StatusNotFound, ""),
				)
			})

			It("should return an error", func() {
				spec, err := iterator.Next()
				Ω(spec).Should(BeNil())
				Ω(err).Should(MatchError("unexpected status code 404"))
			})
		})

		Describe("when the server returns gibberish", func() {
			BeforeEach(func() {
				server.AppendHandlers(
					ghttp.RespondWith(http.StatusOK, "ß"),
				)
			})

			It("should error", func() {
				spec, err := iterator.Next()
				Ω(spec).Should(BeNil())
				Ω(err).ShouldNot(BeNil())
			})
		})
	})
})
