package libcontainer

import "errors"

var (
	ErrExist      = errors.New("container with given ID already exists")
	ErrInvalidID  = errors.New("invalid container ID format")
	ErrNotExist   = errors.New("container does not exist")
	ErrPaused     = errors.New("container paused")
	ErrRunning    = errors.New("container still running")
	ErrNotRunning = errors.New("container not running")
	ErrNotPaused  = errors.New("container not paused")
)
