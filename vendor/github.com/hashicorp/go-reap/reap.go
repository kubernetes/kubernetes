package reap

// ErrorCh is an error channel that lets you know when an error was
// encountered while reaping child processes.
type ErrorCh chan error

// PidCh returns the process IDs of reaped child processes.
type PidCh chan int
