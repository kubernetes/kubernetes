package spec_iterator_test

import (
	. "github.com/onsi/ginkgo/internal/spec_iterator"

	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/internal/spec"
	"github.com/onsi/ginkgo/types"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("SerialSpecIterator", func() {
	var specs []*spec.Spec
	var iterator *SerialIterator

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

		iterator = NewSerialIterator(specs)
	})

	It("should report the total number of specs", func() {
		Ω(iterator.NumberOfSpecsPriorToIteration()).Should(Equal(4))
	})

	It("should report the number to be processed", func() {
		n, known := iterator.NumberOfSpecsToProcessIfKnown()
		Ω(n).Should(Equal(4))
		Ω(known).Should(BeTrue())
	})

	It("should report the number that will be run", func() {
		n, known := iterator.NumberOfSpecsThatWillBeRunIfKnown()
		Ω(n).Should(Equal(2))
		Ω(known).Should(BeTrue())
	})

	Describe("iterating", func() {
		It("should return the specs in order", func() {
			Ω(iterator.Next()).Should(Equal(specs[0]))
			Ω(iterator.Next()).Should(Equal(specs[1]))
			Ω(iterator.Next()).Should(Equal(specs[2]))
			Ω(iterator.Next()).Should(Equal(specs[3]))
			spec, err := iterator.Next()
			Ω(spec).Should(BeNil())
			Ω(err).Should(MatchError(ErrClosed))
		})
	})
})
