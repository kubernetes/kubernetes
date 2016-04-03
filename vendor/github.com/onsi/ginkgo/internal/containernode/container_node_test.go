package containernode_test

import (
	"github.com/onsi/ginkgo/internal/leafnodes"
	"math/rand"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/internal/codelocation"
	. "github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/types"
)

var _ = Describe("Container Node", func() {
	var (
		codeLocation types.CodeLocation
		container    *ContainerNode
	)

	BeforeEach(func() {
		codeLocation = codelocation.New(0)
		container = New("description text", types.FlagTypeFocused, codeLocation)
	})

	Describe("creating a container node", func() {
		It("can answer questions about itself", func() {
			Ω(container.Text()).Should(Equal("description text"))
			Ω(container.Flag()).Should(Equal(types.FlagTypeFocused))
			Ω(container.CodeLocation()).Should(Equal(codeLocation))
		})
	})

	Describe("pushing setup nodes", func() {
		It("can append setup nodes of various types and fetch them by type", func() {
			befA := leafnodes.NewBeforeEachNode(func() {}, codelocation.New(0), 0, nil, 0)
			befB := leafnodes.NewBeforeEachNode(func() {}, codelocation.New(0), 0, nil, 0)
			aftA := leafnodes.NewAfterEachNode(func() {}, codelocation.New(0), 0, nil, 0)
			aftB := leafnodes.NewAfterEachNode(func() {}, codelocation.New(0), 0, nil, 0)
			jusBefA := leafnodes.NewJustBeforeEachNode(func() {}, codelocation.New(0), 0, nil, 0)
			jusBefB := leafnodes.NewJustBeforeEachNode(func() {}, codelocation.New(0), 0, nil, 0)

			container.PushSetupNode(befA)
			container.PushSetupNode(befB)
			container.PushSetupNode(aftA)
			container.PushSetupNode(aftB)
			container.PushSetupNode(jusBefA)
			container.PushSetupNode(jusBefB)

			subject := leafnodes.NewItNode("subject", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
			container.PushSubjectNode(subject)

			Ω(container.SetupNodesOfType(types.SpecComponentTypeBeforeEach)).Should(Equal([]leafnodes.BasicNode{befA, befB}))
			Ω(container.SetupNodesOfType(types.SpecComponentTypeAfterEach)).Should(Equal([]leafnodes.BasicNode{aftA, aftB}))
			Ω(container.SetupNodesOfType(types.SpecComponentTypeJustBeforeEach)).Should(Equal([]leafnodes.BasicNode{jusBefA, jusBefB}))
			Ω(container.SetupNodesOfType(types.SpecComponentTypeIt)).Should(BeEmpty()) //subjects are not setup nodes
		})
	})

	Context("With appended containers and subject nodes", func() {
		var (
			itA, itB, innerItA, innerItB leafnodes.SubjectNode
			innerContainer               *ContainerNode
		)

		BeforeEach(func() {
			itA = leafnodes.NewItNode("Banana", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
			itB = leafnodes.NewItNode("Apple", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)

			innerItA = leafnodes.NewItNode("inner A", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
			innerItB = leafnodes.NewItNode("inner B", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)

			innerContainer = New("Orange", types.FlagTypeNone, codelocation.New(0))

			container.PushSubjectNode(itA)
			container.PushContainerNode(innerContainer)
			innerContainer.PushSubjectNode(innerItA)
			innerContainer.PushSubjectNode(innerItB)
			container.PushSubjectNode(itB)
		})

		Describe("Collating", func() {
			It("should return a collated set of containers and subject nodes in the correct order", func() {
				collated := container.Collate()
				Ω(collated).Should(HaveLen(4))

				Ω(collated[0]).Should(Equal(CollatedNodes{
					Containers: []*ContainerNode{container},
					Subject:    itA,
				}))

				Ω(collated[1]).Should(Equal(CollatedNodes{
					Containers: []*ContainerNode{container, innerContainer},
					Subject:    innerItA,
				}))

				Ω(collated[2]).Should(Equal(CollatedNodes{
					Containers: []*ContainerNode{container, innerContainer},
					Subject:    innerItB,
				}))

				Ω(collated[3]).Should(Equal(CollatedNodes{
					Containers: []*ContainerNode{container},
					Subject:    itB,
				}))
			})
		})

		Describe("Backpropagating Programmatic Focus", func() {
			//This allows inner focused specs to override the focus of outer focussed
			//specs and more closely maps to what a developer wants to happen
			//when debugging a test suite

			Context("when a parent is focused *and* an inner subject is focused", func() {
				BeforeEach(func() {
					container = New("description text", types.FlagTypeFocused, codeLocation)
					itA = leafnodes.NewItNode("A", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
					container.PushSubjectNode(itA)

					innerContainer = New("Orange", types.FlagTypeNone, codelocation.New(0))
					container.PushContainerNode(innerContainer)
					innerItA = leafnodes.NewItNode("inner A", func() {}, types.FlagTypeFocused, codelocation.New(0), 0, nil, 0)
					innerContainer.PushSubjectNode(innerItA)
				})

				It("should unfocus the parent", func() {
					container.BackPropagateProgrammaticFocus()

					Ω(container.Flag()).Should(Equal(types.FlagTypeNone))
					Ω(itA.Flag()).Should(Equal(types.FlagTypeNone))
					Ω(innerContainer.Flag()).Should(Equal(types.FlagTypeNone))
					Ω(innerItA.Flag()).Should(Equal(types.FlagTypeFocused))
				})
			})

			Context("when a parent is focused *and* an inner container is focused", func() {
				BeforeEach(func() {
					container = New("description text", types.FlagTypeFocused, codeLocation)
					itA = leafnodes.NewItNode("A", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
					container.PushSubjectNode(itA)

					innerContainer = New("Orange", types.FlagTypeFocused, codelocation.New(0))
					container.PushContainerNode(innerContainer)
					innerItA = leafnodes.NewItNode("inner A", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
					innerContainer.PushSubjectNode(innerItA)
				})

				It("should unfocus the parent", func() {
					container.BackPropagateProgrammaticFocus()

					Ω(container.Flag()).Should(Equal(types.FlagTypeNone))
					Ω(itA.Flag()).Should(Equal(types.FlagTypeNone))
					Ω(innerContainer.Flag()).Should(Equal(types.FlagTypeFocused))
					Ω(innerItA.Flag()).Should(Equal(types.FlagTypeNone))
				})
			})

			Context("when a parent is pending and a child is focused", func() {
				BeforeEach(func() {
					container = New("description text", types.FlagTypeFocused, codeLocation)
					itA = leafnodes.NewItNode("A", func() {}, types.FlagTypeNone, codelocation.New(0), 0, nil, 0)
					container.PushSubjectNode(itA)

					innerContainer = New("Orange", types.FlagTypePending, codelocation.New(0))
					container.PushContainerNode(innerContainer)
					innerItA = leafnodes.NewItNode("inner A", func() {}, types.FlagTypeFocused, codelocation.New(0), 0, nil, 0)
					innerContainer.PushSubjectNode(innerItA)
				})

				It("should not do anything", func() {
					container.BackPropagateProgrammaticFocus()

					Ω(container.Flag()).Should(Equal(types.FlagTypeFocused))
					Ω(itA.Flag()).Should(Equal(types.FlagTypeNone))
					Ω(innerContainer.Flag()).Should(Equal(types.FlagTypePending))
					Ω(innerItA.Flag()).Should(Equal(types.FlagTypeFocused))
				})
			})
		})

		Describe("Shuffling", func() {
			var unshuffledCollation []CollatedNodes
			BeforeEach(func() {
				unshuffledCollation = container.Collate()

				r := rand.New(rand.NewSource(17))
				container.Shuffle(r)
			})

			It("should sort, and then shuffle, the top level contents of the container", func() {
				shuffledCollation := container.Collate()
				Ω(shuffledCollation).Should(HaveLen(len(unshuffledCollation)))
				Ω(shuffledCollation).ShouldNot(Equal(unshuffledCollation))

				for _, entry := range unshuffledCollation {
					Ω(shuffledCollation).Should(ContainElement(entry))
				}

				innerAIndex, innerBIndex := 0, 0
				for i, entry := range shuffledCollation {
					if entry.Subject == innerItA {
						innerAIndex = i
					} else if entry.Subject == innerItB {
						innerBIndex = i
					}
				}

				Ω(innerAIndex).Should(Equal(innerBIndex - 1))
			})
		})
	})
})
