package fq

import "time"

// TODO(aaron-prindle) currently testing with one concurrent request
const DEFAULT_C = 1 // const C = 300

const DEFAULT_G = 1 //   1 second (virtual time is in seconds)

type FQPacket interface {
	GetServiceTime() float64
	GetQueueIdx() int
	SetQueueIdx(int)
	GetStartTime() time.Time
	SetStartTime(time.Time)
}

// Packet is a temporary container for "requests" with additional tracking fields
// required for the functionality FQScheduler
type Packet struct {
	servicetime float64
	QueueIdx    int
	startTime   time.Time
}

func (p *Packet) GetServiceTime() float64 {
	return p.servicetime
}
func (p *Packet) GetQueueIdx() int {
	return p.QueueIdx
}

func (p *Packet) SetQueueIdx(queueIdx int) {
	p.QueueIdx = queueIdx
}

func (p *Packet) GetStartTime() time.Time {
	return p.startTime
}
func (p *Packet) SetStartTime(starttime time.Time) {
	p.startTime = starttime

}

type FQQueue interface {
	GetPackets() []FQPacket
	GetVirtualFinish(J int, G float64) float64
	GetVirStart() float64
	SetVirStart(float64)
	GetRequestsExecuting() int
	SetRequestsExecuting(int)
	Enqueue(packet FQPacket)
	Dequeue() (FQPacket, bool)
}

// Queue is an array of packets with additional metadata required for
// the FQScheduler
type Queue struct {
	Packets           []FQPacket
	VirStart          float64
	RequestsExecuting int
}

// GetPackets
func (q *Queue) GetPackets() []FQPacket {
	return q.Packets
}

// GetRequestsExecuting
func (q *Queue) GetRequestsExecuting() int {
	return q.RequestsExecuting
}

// GetRequestsExecuting
func (q *Queue) SetRequestsExecuting(requestsExecuting int) {
	q.RequestsExecuting = requestsExecuting
}

// GetVirStart
func (q *Queue) GetVirStart() float64 {
	return q.VirStart
}

// SetVirStart
func (q *Queue) SetVirStart(virstart float64) {
	q.VirStart = virstart
}

// Enqueue enqueues a packet into the queue
func (q *Queue) Enqueue(packet FQPacket) {
	q.Packets = append(q.GetPackets(), packet)
}

// Dequeue dequeues a packet from the queue
func (q *Queue) Dequeue() (FQPacket, bool) {
	if len(q.Packets) == 0 {
		return nil, false
	}
	packet := q.Packets[0]
	q.Packets = q.Packets[1:]

	return packet, true
}

// VirtualFinish returns the expected virtual finish time of the packet at
// index J in the queue with estimated finish time G
func (q *Queue) GetVirtualFinish(J int, G float64) float64 {
	// The virtual finish time of request number J in the queue
	// (counting from J=1 for the head) is J * G + (virtual start time).

	// counting from J=1 for the head (eg: queue.Packets[0] -> J=1) - J+1
	jg := float64(J+1) * float64(G)
	return jg + q.VirStart
}
