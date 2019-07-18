package fq

import (
	"fmt"
	"math"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

// FQScheduler is a fair queuing implementation designed for the kube-apiserver.
// FQ is designed for
// 1) dispatching requests to be served rather than packets to be transmitted
// 2) serving multiple requests at once
// 3) accounting for unknown and varying service time
type FQScheduler struct {
	lock         sync.Mutex
	Queues       []FQQueue
	clock        clock.Clock
	vt           float64
	C            int
	G            float64
	lastRealTime time.Time
	robinidx     int
}

func (fqs *FQScheduler) chooseQueue(packet FQPacket) FQQueue {
	if packet.GetQueueIdx() < 0 || packet.GetQueueIdx() > len(fqs.Queues) {
		panic("no matching queue for packet")
	}
	return fqs.Queues[packet.GetQueueIdx()]
}

func NewFQScheduler(queues []FQQueue, clock clock.Clock) *FQScheduler {
	fq := &FQScheduler{
		Queues:       queues,
		clock:        clock,
		vt:           0,
		lastRealTime: clock.Now(),
		C:            DEFAULT_C,
		G:            DEFAULT_G,
	}
	return fq
}

// Enqueue enqueues a packet into the fair queuing scheduler
func (fqs *FQScheduler) Enqueue(packet FQPacket) {
	fqs.lock.Lock()
	defer fqs.lock.Unlock()
	fqs.synctime()

	queue := fqs.chooseQueue(packet)
	queue.Enqueue(packet)
	fqs.updateTime(packet, queue)
}

func (fqs *FQScheduler) getVirtualTime() float64 {
	return fqs.vt
}

func (fqs *FQScheduler) synctime() {
	realNow := fqs.clock.Now()
	timesincelast := realNow.Sub(fqs.lastRealTime).Seconds()
	fqs.lastRealTime = realNow
	fqs.vt += timesincelast * fqs.getvirtualtimeratio()
}

func (fqs *FQScheduler) getvirtualtimeratio() float64 {
	NEQ := 0
	reqs := 0
	for _, queue := range fqs.Queues {
		reqs += queue.GetRequestsExecuting()
		// It might be best to delete this line. If everything is working
		//  correctly, there will be no waiting packets if reqs < C on current
		//  line 85; if something is going wrong, it is more accurate to say
		// that virtual time advanced due to the requests actually executing.

		// reqs += len(queue.GetPackets())
		if len(queue.GetPackets()) > 0 || queue.GetRequestsExecuting() > 0 {
			NEQ++
		}
	}
	// no active flows, virtual time does not advance (also avoid div by 0)
	if NEQ == 0 {
		return 0
	}
	return min(float64(reqs), float64(fqs.C)) / float64(NEQ)
}

func (fqs *FQScheduler) updateTime(packet FQPacket, queue FQQueue) {
	// When a request arrives to an empty queue with no requests executing
	// len(queue.GetPackets()) == 1 as enqueue has just happened prior (vs  == 0)
	if len(queue.GetPackets()) == 1 && queue.GetRequestsExecuting() == 0 {
		// the queue’s virtual start time is set to getVirtualTime().
		queue.SetVirStart(fqs.getVirtualTime())
	}
}

// FinishPacketAndDequeue is a convenience method used using the FQScheduler
// at the concurrency limit
func (fqs *FQScheduler) FinishPacketAndDeque(p FQPacket) (FQPacket, bool) {
	fqs.FinishPacket(p)
	return fqs.Dequeue()
}

// FinishPacket is a callback that should be used when a previously dequeud packet
// has completed it's service.  This callback updates imporatnt state in the
//  FQScheduler
func (fqs *FQScheduler) FinishPacket(p FQPacket) {
	fqs.lock.Lock()
	defer fqs.lock.Unlock()

	fqs.synctime()
	S := fqs.clock.Since(p.GetStartTime()).Seconds()

	// When a request finishes being served, and the actual service time was S,
	// the queue’s virtual start time is decremented by G - S.
	virstart := fqs.Queues[p.GetQueueIdx()].GetVirStart()
	virstart -= fqs.G - S
	fqs.Queues[p.GetQueueIdx()].SetVirStart(virstart)

	// request has finished, remove from requests executing
	requestsExecuting := fqs.Queues[p.GetQueueIdx()].GetRequestsExecuting()
	requestsExecuting--
	fqs.Queues[p.GetQueueIdx()].SetRequestsExecuting(requestsExecuting)

	// TODO(aaron-prindle) using curQueue seems to copy
	// // When a request finishes being served, and the actual service time was S,
	// // the queue’s virtual start time is decremented by G - S.
	// curQueue := fqs.Queues[p.GetQueueIdx()]
	// virstart := curQueue.GetVirStart()
	// virstart -= fqs.G - S
	// curQueue.SetVirStart(virstart)

	// // request has finished, remove from requests executing
	// requestsExecuting := curQueue.GetRequestsExecuting()
	// requestsExecuting--
	// curQueue.SetRequestsExecuting(requestsExecuting)

}

// Dequeue dequeues a packet from the fair queuing scheduler
func (fqs *FQScheduler) Dequeue() (FQPacket, bool) {
	fqs.lock.Lock()
	defer fqs.lock.Unlock()
	fqs.synctime()
	queue := fqs.selectQueue()

	fmt.Println("1")
	if queue == nil {
		fmt.Println("2")
		return nil, false
	}
	packet, ok := queue.Dequeue()

	if ok {
		fmt.Println("3")
		// When a request is dequeued for service -> fqs.VirStart += G
		virstart := queue.GetVirStart()
		virstart += fqs.G
		queue.SetVirStart(virstart)

		packet.SetStartTime(fqs.clock.Now())
		// request dequeued, service has started
		queue.SetRequestsExecuting(queue.GetRequestsExecuting() + 1)
	} else {
		// TODO(aaron-prindle) verify this statement is needed...
		return nil, false
	}

	fmt.Println("4")
	return packet, ok
}

func (fqs *FQScheduler) roundrobinqueue() int {
	fqs.robinidx = (fqs.robinidx + 1) % len(fqs.Queues)
	return fqs.robinidx
}

func (fqs *FQScheduler) selectQueue() FQQueue {
	minvirfinish := math.Inf(1)
	var minqueue FQQueue
	var minidx int
	for range fqs.Queues {
		idx := fqs.roundrobinqueue()
		queue := fqs.Queues[idx]
		if len(queue.GetPackets()) != 0 {
			curvirfinish := queue.GetVirtualFinish(0, fqs.G)
			if curvirfinish < minvirfinish {
				minvirfinish = curvirfinish
				minqueue = queue
				minidx = idx
			}
		}
	}
	fqs.robinidx = minidx
	return minqueue
}
