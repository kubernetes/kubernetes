package execpath

import "os/exec"

type Error = exec.Error

// ErrNotFound is the error resulting if a path search failed to find an executable file.
var ErrNotFound = exec.ErrNotFound
