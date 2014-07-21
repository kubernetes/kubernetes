package main

import "syscall"

func main() {
	// Halts execution, waiting on signal.
	syscall.Pause()
}
