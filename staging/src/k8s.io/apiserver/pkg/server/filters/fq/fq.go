package fq

import (
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

func (q *FQScheduler) chooseQueue(packet FQPacket) FQQueue {
	if packet.GetQueueIdx() < 0 || packet.GetQueueIdx() > len(q.Queues) {
		panic("no matching queue for packet")
	}
	return q.Queues[packet.GetQueueIdx()]
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
func (q *FQScheduler) Enqueue(packet FQPacket) {
	q.lock.Lock()
	defer q.lock.Unlock()
	q.synctime()

	queue := q.chooseQueue(packet)
	queue.Enqueue(packet)
	q.updateTime(packet, queue)
}

func (q *FQScheduler) getVirtualTime() float64 {
	return q.vt
}

func (q *FQScheduler) synctime() {
	realNow := q.clock.Now()
	timesincelast := realNow.Sub(q.lastRealTime).Seconds()
	q.lastRealTime = realNow
	q.vt += timesincelast * q.getvirtualtimeratio()
}

func (q *FQScheduler) getvirtualtimeratio() float64 {
	NEQ := 0
	reqs := 0
	for _, queue := range q.Queues {
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
	return min(float64(reqs), float64(q.C)) / float64(NEQ)
}

func (q *FQScheduler) updateTime(packet FQPacket, queue FQQueue) {
	// When a request arrives to an empty queue with no requests executing
	// len(queue.GetPackets()) == 1 as enqueue has just happened prior (vs  == 0)
	if len(queue.GetPackets()) == 1 && queue.GetRequestsExecuting() == 0 {
		// the queue’s virtual start time is set to getVirtualTime().
		queue.SetVirStart(q.getVirtualTime())
	}
}

// FinishPacketAndDequeue is a convenience method used using the FQScheduler
// at the concurrency limit
func (q *FQScheduler) FinishPacketAndDeque(p FQPacket) (FQPacket, bool) {
	q.FinishPacket(p)
	return q.Dequeue()
}

// FinishPacket is a callback that should be used when a previously dequeud packet
// has completed it's service.  This callback updates imporatnt state in the
//  FQScheduler
func (q *FQScheduler) FinishPacket(p FQPacket) {
	q.lock.Lock()
	defer q.lock.Unlock()

	q.synctime()
	S := q.clock.Since(p.GetStartTime()).Seconds()

	// When a request finishes being served, and the actual service time was S,
	// the queue’s virtual start time is decremented by G - S.
	virstart := q.Queues[p.GetQueueIdx()].GetVirStart()
	virstart -= q.G - S
	q.Queues[p.GetQueueIdx()].SetVirStart(virstart)

	// request has finished, remove from requests executing
	requestsExecuting := q.Queues[p.GetQueueIdx()].GetRequestsExecuting()
	requestsExecuting--
	q.Queues[p.GetQueueIdx()].SetRequestsExecuting(requestsExecuting)
}

// Dequeue dequeues a packet from the fair queuing scheduler
func (q *FQScheduler) Dequeue() (FQPacket, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()
	q.synctime()
	queue := q.selectQueue()

	if queue == nil {
		return nil, false
	}
	packet, ok := queue.Dequeue()

	if ok {
		// When a request is dequeued for service -> q.VirStart += G
		virstart := queue.GetVirStart()
		virstart += q.G
		queue.SetVirStart(virstart)

		packet.SetStartTime(q.clock.Now())
		// request dequeued, service has started
		requestsExecuting := queue.GetRequestsExecuting()
		requestsExecuting++
		queue.SetRequestsExecuting(requestsExecuting)
	}
	return packet, ok
}

func (q *FQScheduler) roundrobinqueue() int {
	q.robinidx = (q.robinidx + 1) % len(q.Queues)
	return q.robinidx
}

func (q *FQScheduler) selectQueue() FQQueue {
	minvirfinish := math.Inf(1)
	var minqueue FQQueue
	var minidx int
	for range q.Queues {
		idx := q.roundrobinqueue()
		queue := q.Queues[idx]
		if len(queue.GetPackets()) != 0 {
			curvirfinish := queue.GetVirtualFinish(0, q.G)
			if curvirfinish < minvirfinish {
				minvirfinish = curvirfinish
				minqueue = queue
				minidx = idx
			}
		}
	}
	q.robinidx = minidx
	return minqueue
}
