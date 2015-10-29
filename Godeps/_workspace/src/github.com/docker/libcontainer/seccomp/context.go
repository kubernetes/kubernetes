package seccomp

import (
	"errors"
	"syscall"
)

const labelTemplate = "lb-%d-%d"

// Action is the type of action that will be taken when a
// syscall is performed.
type Action int

const (
	Kill  Action = iota - 3 // Kill the calling process of the syscall.
	Trap                    // Trap and coredump the calling process of the syscall.
	Allow                   // Allow the syscall to be completed.
)

// Syscall is the specified syscall, action, and any type of arguments
// to filter on.
type Syscall struct {
	// Value is the syscall number.
	Value uint32
	// Action is the action to perform when the specified syscall is made.
	Action Action
	// Args are filters that can be specified on the arguments to the syscall.
	Args Args
}

func (s *Syscall) scmpAction() uint32 {
	switch s.Action {
	case Allow:
		return retAllow
	case Trap:
		return retTrap
	case Kill:
		return retKill
	}
	return actionErrno(uint32(s.Action))
}

// Arg represents an argument to the syscall with the argument's index,
// the operator to apply when matching, and the argument's value at that time.
type Arg struct {
	Index uint32   // index of args which start from zero
	Op    Operator // operation, such as EQ/NE/GE/LE
	Value uint     // the value of arg
}

type Args [][]Arg

var (
	ErrUnresolvedLabel      = errors.New("seccomp: unresolved label")
	ErrDuplicateLabel       = errors.New("seccomp: duplicate label use")
	ErrUnsupportedOperation = errors.New("seccomp: unsupported operation for argument")
)

// Error returns an Action that will be used to send the calling
// process the specified errno when the syscall is made.
func Error(code syscall.Errno) Action {
	return Action(code)
}

// New returns a new syscall context for use.
func New() *Context {
	return &Context{
		syscalls: make(map[uint32]*Syscall),
	}
}

// Context holds syscalls for the current process to limit the type of
// actions the calling process can make.
type Context struct {
	syscalls map[uint32]*Syscall
}

// Add will add the specified syscall, action, and arguments to the seccomp
// Context.
func (c *Context) Add(s *Syscall) {
	c.syscalls[s.Value] = s
}

// Remove removes the specified syscall configuration from the Context.
func (c *Context) Remove(call uint32) {
	delete(c.syscalls, call)
}

// Load will apply the Context to the calling process makeing any secccomp process changes
// apply after the context is loaded.
func (c *Context) Load() error {
	filter, err := c.newFilter()
	if err != nil {
		return err
	}
	if err := prctl(prSetNoNewPrivileges, 1, 0, 0, 0); err != nil {
		return err
	}
	prog := newSockFprog(filter)
	return prog.set()
}

func (c *Context) newFilter() ([]sockFilter, error) {
	var (
		labels bpfLabels
		f      = newFilter()
	)
	for _, s := range c.syscalls {
		f.addSyscall(s, &labels)
	}
	f.allow()
	// process args for the syscalls
	for _, s := range c.syscalls {
		if err := f.addArguments(s, &labels); err != nil {
			return nil, err
		}
	}
	// apply labels for arguments
	idx := int32(len(*f) - 1)
	for ; idx >= 0; idx-- {
		lf := &(*f)[idx]
		if lf.code != (syscall.BPF_JMP + syscall.BPF_JA) {
			continue
		}
		rel := int32(lf.jt)<<8 | int32(lf.jf)
		if ((jumpJT << 8) | jumpJF) == rel {
			if labels[lf.k].location == 0xffffffff {
				return nil, ErrUnresolvedLabel
			}
			lf.k = labels[lf.k].location - uint32(idx+1)
			lf.jt = 0
			lf.jf = 0
		} else if ((labelJT << 8) | labelJF) == rel {
			if labels[lf.k].location != 0xffffffff {
				return nil, ErrDuplicateLabel
			}
			labels[lf.k].location = uint32(idx)
			lf.k = 0
			lf.jt = 0
			lf.jf = 0
		}
	}
	return *f, nil
}
