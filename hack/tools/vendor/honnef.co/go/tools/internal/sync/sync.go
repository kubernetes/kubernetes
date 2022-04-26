package sync

type Semaphore struct {
	ch chan struct{}
}

func NewSemaphore(size int) Semaphore {
	return Semaphore{
		ch: make(chan struct{}, size),
	}
}

func (sem Semaphore) Acquire() {
	sem.ch <- struct{}{}
}

func (sem Semaphore) AcquireMaybe() bool {
	select {
	case sem.ch <- struct{}{}:
		return true
	default:
		return false
	}
}

func (sem Semaphore) Release() {
	<-sem.ch
}

func (sem Semaphore) Len() int {
	return len(sem.ch)
}

func (sem Semaphore) Cap() int {
	return cap(sem.ch)
}
