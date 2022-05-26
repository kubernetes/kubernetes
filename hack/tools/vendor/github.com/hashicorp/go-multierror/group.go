package multierror

import "sync"

// Group is a collection of goroutines which return errors that need to be
// coalesced.
type Group struct {
	mutex sync.Mutex
	err   *Error
	wg    sync.WaitGroup
}

// Go calls the given function in a new goroutine.
//
// If the function returns an error it is added to the group multierror which
// is returned by Wait.
func (g *Group) Go(f func() error) {
	g.wg.Add(1)

	go func() {
		defer g.wg.Done()

		if err := f(); err != nil {
			g.mutex.Lock()
			g.err = Append(g.err, err)
			g.mutex.Unlock()
		}
	}()
}

// Wait blocks until all function calls from the Go method have returned, then
// returns the multierror.
func (g *Group) Wait() *Error {
	g.wg.Wait()
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g.err
}
