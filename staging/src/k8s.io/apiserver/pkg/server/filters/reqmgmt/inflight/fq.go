package inflight

import (
	"fmt"
	"math"
	"sync"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/server/filters/fq"
)

// FQScheduler is a fair queuing implementation designed for the kube-apiserver.
// FQ is designed for
// 1) dispatching requests to be served rather than packets to be transmitted
// 2) serving multiple requests at once
// 3) accounting for unknown and varying service time
type FQScheduler struct {
	lock             sync.Mutex
	fq               *fq.FQScheduler
	actualnumqueues  int // TODO(aaron-prindle) currently unused...
	numqueues        int
	numPackets       int
	QueueLengthLimit int // TODO(aaron-prindle) currently unused...
}

func NewFQScheduler(queues []fq.FQQueue, clock clock.Clock) *FQScheduler {
	fq := &FQScheduler{
		fq: fq.NewFQScheduler(queues, clock),
	}
	return fq
}

func (q *FQScheduler) GetRequestsExecuting() int {
	q.lock.Lock()
	defer q.lock.Unlock()
	total := 0
	for _, queue := range q.fq.Queues {
		total += queue.GetRequestsExecuting()
	}
	return total
}

func (q *FQScheduler) GetQueues() []fq.FQQueue {
	return q.fq.Queues
}

func (q *FQScheduler) IsEmpty() bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	return q.numPackets == 0
}

func (q *FQScheduler) lengthOfQueue(i int) int {
	return len(q.fq.Queues[i].GetPackets())
}

func (q *FQScheduler) shuffleDealAndPick(v, nq uint64,
	mr func(int /*in [0, nq-1]*/) int, /*in [0, numQueues-1] and excluding previously determined members of I*/
	nRem, minLen, bestIdx int) int {
	// numQueues := uint64(q.numqueues)
	if nRem < 1 {
		return bestIdx
	}
	vNext := v / nq
	ai := int(v - nq*vNext)
	ii := mr(ai)
	// i := numQueues - nq // i is used only for debug printing
	mrNext := func(a int /*in [0, nq-2]*/) int /*in [0, numQueues-1] and excluding I[0], I[1], ... ii*/ {
		if a < ai {
			// fmt.Printf("mr[%v](%v) going low\n", i, a)
			return mr(a)
		}
		// fmt.Printf("mr[%v](%v) going high\n", i, a)
		return mr(a + 1)
	}
	lenI := q.lengthOfQueue(ii)
	// fmt.Printf("Considering A[%v]=%v, I[%v]=%v, qlen[%v]=%v\n\n", i, ai, i, ii, i, lenI)
	if lenI < minLen {
		minLen = lenI
		bestIdx = ii
	}
	return q.shuffleDealAndPick(vNext, nq-1, mrNext, nRem-1, minLen, bestIdx)
}

func (q *FQScheduler) ChooseQueueIdx(hashValue uint64, handSize int) int {
	// modifies a packet to set the queueidx
	// uses shuffle sharding
	return q.shuffleDealAndPick(hashValue, uint64(len(q.fq.Queues)), func(i int) int { return i }, handSize, math.MaxInt32, -1)
}

func (q *FQScheduler) Enqueue(pkt fq.FQPacket) {
	q.lock.Lock()
	defer q.lock.Unlock()

	q.numPackets++

	q.fq.Enqueue(pkt)
}

// FinishPacket is a callback that should be used when a previously dequeud packet
// has completed it's service.  This callback updates imporatnt state in the
//  FQScheduler
func (q *FQScheduler) FinishPacketAndDequeueNextPacket(p fq.FQPacket) (bool, fq.FQPacket) {
	q.FinishPacket(p)
	return q.Dequeue()
}

// FinishPacket is a callback that should be used when a previously dequeud packet
// has completed it's service.  This callback updates imporatnt state in the
//  FQScheduler
func (q *FQScheduler) FinishPacket(p fq.FQPacket) {
	q.fq.FinishPacket(p)
}

// Dequeue dequeues a packet from the fair queuing scheduler
func (q *FQScheduler) Dequeue() (bool, fq.FQPacket) {
	q.lock.Lock()
	defer q.lock.Unlock()

	pkt, ok := q.fq.Dequeue()
	if !ok {
		return false, nil
	}

	reqMgmtPkt := pkt.(*Packet)

	fmt.Printf("dequeue: %v, %v\n", reqMgmtPkt.DequeueChannel, reqMgmtPkt.Seq)

	reqMgmtPkt.DequeueChannel <- true
	q.numPackets--
	return true, pkt
}

// Dequeue dequeues a packet from the fair queuing scheduler
func (q *FQScheduler) DecrementPackets(i int) {
	q.lock.Lock()
	defer q.lock.Unlock()

	q.numPackets -= i
}
