package cli

import (
	"sync"
)

// ConcurrentUi is a wrapper around a Ui interface (and implements that
// interface) making the underlying Ui concurrency safe.
type ConcurrentUi struct {
	Ui Ui
	l  sync.Mutex
}

func (u *ConcurrentUi) Ask(query string) (string, error) {
	u.l.Lock()
	defer u.l.Unlock()

	return u.Ui.Ask(query)
}

func (u *ConcurrentUi) AskSecret(query string) (string, error) {
	u.l.Lock()
	defer u.l.Unlock()

	return u.Ui.AskSecret(query)
}

func (u *ConcurrentUi) Error(message string) {
	u.l.Lock()
	defer u.l.Unlock()

	u.Ui.Error(message)
}

func (u *ConcurrentUi) Info(message string) {
	u.l.Lock()
	defer u.l.Unlock()

	u.Ui.Info(message)
}

func (u *ConcurrentUi) Output(message string) {
	u.l.Lock()
	defer u.l.Unlock()

	u.Ui.Output(message)
}

func (u *ConcurrentUi) Warn(message string) {
	u.l.Lock()
	defer u.l.Unlock()

	u.Ui.Warn(message)
}
