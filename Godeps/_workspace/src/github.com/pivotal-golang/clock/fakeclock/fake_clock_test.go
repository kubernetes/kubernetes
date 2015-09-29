package fakeclock_test

import (
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/pivotal-golang/clock/fakeclock"
)

var _ = Describe("FakeClock", func() {
	const Δ time.Duration = 10 * time.Millisecond

	var (
		fakeClock   *fakeclock.FakeClock
		initialTime time.Time
	)

	BeforeEach(func() {
		initialTime = time.Date(2014, 1, 1, 3, 0, 30, 0, time.UTC)
		fakeClock = fakeclock.NewFakeClock(initialTime)
	})

	Describe("Now", func() {
		It("returns the current time, w/o race conditions", func() {
			go fakeClock.Increment(time.Minute)
			Eventually(fakeClock.Now).Should(Equal(initialTime.Add(time.Minute)))
		})
	})

	Describe("Sleep", func() {
		It("blocks until the given interval elapses", func() {
			doneSleeping := make(chan struct{})
			go func() {
				fakeClock.Sleep(10 * time.Second)
				close(doneSleeping)
			}()

			Consistently(doneSleeping, Δ).ShouldNot(BeClosed())

			fakeClock.Increment(5 * time.Second)
			Consistently(doneSleeping, Δ).ShouldNot(BeClosed())

			fakeClock.Increment(4 * time.Second)
			Consistently(doneSleeping, Δ).ShouldNot(BeClosed())

			fakeClock.Increment(1 * time.Second)
			Eventually(doneSleeping).Should(BeClosed())
		})
	})

	Describe("WatcherCount", func() {
		Context("when a timer is created", func() {
			It("increments the watcher count", func() {
				fakeClock.NewTimer(time.Second)
				Expect(fakeClock.WatcherCount()).To(Equal(1))

				fakeClock.NewTimer(2 * time.Second)
				Expect(fakeClock.WatcherCount()).To(Equal(2))
			})
		})

		Context("when a timer fires", func() {
			It("increments the watcher count", func() {
				fakeClock.NewTimer(time.Second)
				Expect(fakeClock.WatcherCount()).To(Equal(1))

				fakeClock.Increment(time.Second)
				Expect(fakeClock.WatcherCount()).To(Equal(0))
			})
		})
	})
})
