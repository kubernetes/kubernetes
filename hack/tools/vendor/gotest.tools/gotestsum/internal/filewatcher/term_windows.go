package filewatcher

import "context"

type redoHandler struct{}

func newRedoHandler() *redoHandler {
	return nil
}

func (r *redoHandler) Run(_ context.Context) {}

func (r *redoHandler) Ch() <-chan RunOptions {
	return nil
}

func (r *redoHandler) SetupTerm() {}

func (r *redoHandler) ResetTerm() {}
