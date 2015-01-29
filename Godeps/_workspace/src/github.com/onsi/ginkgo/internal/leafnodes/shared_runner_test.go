package leafnodes_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/leafnodes"
	. "github.com/onsi/gomega"

	"reflect"
	"runtime"
	"time"

	"github.com/onsi/ginkgo/internal/codelocation"
	Failer "github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
)

type runnable interface {
	Run() (outcome types.SpecState, failure types.SpecFailure)
	CodeLocation() types.CodeLocation
}

func SynchronousSharedRunnerBehaviors(build func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable, componentType types.SpecComponentType, componentIndex int) {
	var (
		outcome types.SpecState
		failure types.SpecFailure

		failer *Failer.Failer

		componentCodeLocation types.CodeLocation
		innerCodeLocation     types.CodeLocation

		didRun bool
	)

	BeforeEach(func() {
		failer = Failer.New()
		componentCodeLocation = codelocation.New(0)
		innerCodeLocation = codelocation.New(0)

		didRun = false
	})

	Describe("synchronous functions", func() {
		Context("when the function passes", func() {
			BeforeEach(func() {
				outcome, failure = build(func() {
					didRun = true
				}, 0, failer, componentCodeLocation).Run()
			})

			It("should have a succesful outcome", func() {
				Ω(didRun).Should(BeTrue())

				Ω(outcome).Should(Equal(types.SpecStatePassed))
				Ω(failure).Should(BeZero())
			})
		})

		Context("when a failure occurs", func() {
			BeforeEach(func() {
				outcome, failure = build(func() {
					didRun = true
					failer.Fail("bam", innerCodeLocation)
					panic("should not matter")
				}, 0, failer, componentCodeLocation).Run()
			})

			It("should return the failure", func() {
				Ω(didRun).Should(BeTrue())

				Ω(outcome).Should(Equal(types.SpecStateFailed))
				Ω(failure).Should(Equal(types.SpecFailure{
					Message:               "bam",
					Location:              innerCodeLocation,
					ForwardedPanic:        "",
					ComponentIndex:        componentIndex,
					ComponentType:         componentType,
					ComponentCodeLocation: componentCodeLocation,
				}))
			})
		})

		Context("when a panic occurs", func() {
			BeforeEach(func() {
				outcome, failure = build(func() {
					didRun = true
					innerCodeLocation = codelocation.New(0)
					panic("ack!")
				}, 0, failer, componentCodeLocation).Run()
			})

			It("should return the panic", func() {
				Ω(didRun).Should(BeTrue())

				Ω(outcome).Should(Equal(types.SpecStatePanicked))
				Ω(failure.ForwardedPanic).Should(Equal("ack!"))
			})
		})
	})
}

func AsynchronousSharedRunnerBehaviors(build func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable, componentType types.SpecComponentType, componentIndex int) {
	var (
		outcome types.SpecState
		failure types.SpecFailure

		failer *Failer.Failer

		componentCodeLocation types.CodeLocation
		innerCodeLocation     types.CodeLocation

		didRun bool
	)

	BeforeEach(func() {
		failer = Failer.New()
		componentCodeLocation = codelocation.New(0)
		innerCodeLocation = codelocation.New(0)

		didRun = false
	})

	Describe("asynchronous functions", func() {
		var timeoutDuration time.Duration

		BeforeEach(func() {
			timeoutDuration = time.Duration(1 * float64(time.Second))
		})

		Context("when running", func() {
			It("should run the function as a goroutine, and block until it's done", func() {
				initialNumberOfGoRoutines := runtime.NumGoroutine()
				numberOfGoRoutines := 0

				build(func(done Done) {
					didRun = true
					numberOfGoRoutines = runtime.NumGoroutine()
					close(done)
				}, timeoutDuration, failer, componentCodeLocation).Run()

				Ω(didRun).Should(BeTrue())
				Ω(numberOfGoRoutines).Should(BeNumerically(">=", initialNumberOfGoRoutines+1))
			})
		})

		Context("when the function passes", func() {
			BeforeEach(func() {
				outcome, failure = build(func(done Done) {
					didRun = true
					close(done)
				}, timeoutDuration, failer, componentCodeLocation).Run()
			})

			It("should have a succesful outcome", func() {
				Ω(didRun).Should(BeTrue())
				Ω(outcome).Should(Equal(types.SpecStatePassed))
				Ω(failure).Should(BeZero())
			})
		})

		Context("when the function fails", func() {
			BeforeEach(func() {
				outcome, failure = build(func(done Done) {
					didRun = true
					failer.Fail("bam", innerCodeLocation)
					time.Sleep(20 * time.Millisecond)
					panic("doesn't matter")
					close(done)
				}, 10*time.Millisecond, failer, componentCodeLocation).Run()
			})

			It("should return the failure", func() {
				Ω(didRun).Should(BeTrue())

				Ω(outcome).Should(Equal(types.SpecStateFailed))
				Ω(failure).Should(Equal(types.SpecFailure{
					Message:               "bam",
					Location:              innerCodeLocation,
					ForwardedPanic:        "",
					ComponentIndex:        componentIndex,
					ComponentType:         componentType,
					ComponentCodeLocation: componentCodeLocation,
				}))
			})
		})

		Context("when the function times out", func() {
			var guard chan struct{}

			BeforeEach(func() {
				guard = make(chan struct{})
				outcome, failure = build(func(done Done) {
					didRun = true
					time.Sleep(20 * time.Millisecond)
					close(guard)
					panic("doesn't matter")
					close(done)
				}, 10*time.Millisecond, failer, componentCodeLocation).Run()
			})

			It("should return the timeout", func() {
				<-guard
				Ω(didRun).Should(BeTrue())

				Ω(outcome).Should(Equal(types.SpecStateTimedOut))
				Ω(failure).Should(Equal(types.SpecFailure{
					Message:               "Timed out",
					Location:              componentCodeLocation,
					ForwardedPanic:        "",
					ComponentIndex:        componentIndex,
					ComponentType:         componentType,
					ComponentCodeLocation: componentCodeLocation,
				}))
			})
		})

		Context("when the function panics", func() {
			BeforeEach(func() {
				outcome, failure = build(func(done Done) {
					didRun = true
					innerCodeLocation = codelocation.New(0)
					panic("ack!")
				}, 100*time.Millisecond, failer, componentCodeLocation).Run()
			})

			It("should return the panic", func() {
				Ω(didRun).Should(BeTrue())

				Ω(outcome).Should(Equal(types.SpecStatePanicked))
				Ω(failure.ForwardedPanic).Should(Equal("ack!"))
			})
		})
	})
}

func InvalidSharedRunnerBehaviors(build func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable, componentType types.SpecComponentType) {
	var (
		failer                *Failer.Failer
		componentCodeLocation types.CodeLocation
		innerCodeLocation     types.CodeLocation
	)

	BeforeEach(func() {
		failer = Failer.New()
		componentCodeLocation = codelocation.New(0)
		innerCodeLocation = codelocation.New(0)
	})

	Describe("invalid functions", func() {
		Context("when passed something that's not a function", func() {
			It("should panic", func() {
				Ω(func() {
					build("not a function", 0, failer, componentCodeLocation)
				}).Should(Panic())
			})
		})

		Context("when the function takes the wrong kind of argument", func() {
			It("should panic", func() {
				Ω(func() {
					build(func(oops string) {}, 0, failer, componentCodeLocation)
				}).Should(Panic())
			})
		})

		Context("when the function takes more than one argument", func() {
			It("should panic", func() {
				Ω(func() {
					build(func(done Done, oops string) {}, 0, failer, componentCodeLocation)
				}).Should(Panic())
			})
		})
	})
}

var _ = Describe("Shared RunnableNode behavior", func() {
	Describe("It Nodes", func() {
		build := func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable {
			return NewItNode("", body, types.FlagTypeFocused, componentCodeLocation, timeout, failer, 3)
		}

		SynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeIt, 3)
		AsynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeIt, 3)
		InvalidSharedRunnerBehaviors(build, types.SpecComponentTypeIt)
	})

	Describe("Measure Nodes", func() {
		build := func(body interface{}, _ time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable {
			return NewMeasureNode("", func(Benchmarker) {
				reflect.ValueOf(body).Call([]reflect.Value{})
			}, types.FlagTypeFocused, componentCodeLocation, 10, failer, 3)
		}

		SynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeMeasure, 3)
	})

	Describe("BeforeEach Nodes", func() {
		build := func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable {
			return NewBeforeEachNode(body, componentCodeLocation, timeout, failer, 3)
		}

		SynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeBeforeEach, 3)
		AsynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeBeforeEach, 3)
		InvalidSharedRunnerBehaviors(build, types.SpecComponentTypeBeforeEach)
	})

	Describe("AfterEach Nodes", func() {
		build := func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable {
			return NewAfterEachNode(body, componentCodeLocation, timeout, failer, 3)
		}

		SynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeAfterEach, 3)
		AsynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeAfterEach, 3)
		InvalidSharedRunnerBehaviors(build, types.SpecComponentTypeAfterEach)
	})

	Describe("JustBeforeEach Nodes", func() {
		build := func(body interface{}, timeout time.Duration, failer *Failer.Failer, componentCodeLocation types.CodeLocation) runnable {
			return NewJustBeforeEachNode(body, componentCodeLocation, timeout, failer, 3)
		}

		SynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeJustBeforeEach, 3)
		AsynchronousSharedRunnerBehaviors(build, types.SpecComponentTypeJustBeforeEach, 3)
		InvalidSharedRunnerBehaviors(build, types.SpecComponentTypeJustBeforeEach)
	})
})
