package server


type channelFactory interface {
	new(name string) channelWrapper
	wrapReadOnly(<-chan struct{}) channelWaiter
}

type channelWaiter interface {
	wait() // blocking
}

type channelWrapper interface {
	channelWaiter
	close()
	rawChan() chan struct{}
}


type simpleFactory struct {}

func (f simpleFactory) new(name string) channelWrapper {
  return rwChan{delegate: make(chan struct{})}
}
func (f simpleFactory) wrapReadOnly(delegate <-chan struct{}) channelWaiter {
	return rChan{delegate: delegate}
}

type rwChan struct {
	delegate chan struct{}
}

func (s rwChan) wait() {
	<-s.delegate
}

func (s rwChan) close() {
	close(s.delegate)
}

func (s rwChan) rawChan() chan struct{} {
	return s.delegate
}

type rChan struct {
	delegate <-chan struct{}
}

func (sr rChan) wait() {
	<-sr.delegate
}