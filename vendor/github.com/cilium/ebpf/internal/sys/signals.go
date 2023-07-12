package sys

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/cilium/ebpf/internal/unix"
)

// A sigset containing only SIGPROF.
var profSet unix.Sigset_t

func init() {
	// See sigsetAdd for details on the implementation. Open coded here so
	// that the compiler will check the constant calculations for us.
	profSet.Val[sigprofBit/wordBits] |= 1 << (sigprofBit % wordBits)
}

// maskProfilerSignal locks the calling goroutine to its underlying OS thread
// and adds SIGPROF to the thread's signal mask. This prevents pprof from
// interrupting expensive syscalls like e.g. BPF_PROG_LOAD.
//
// The caller must defer unmaskProfilerSignal() to reverse the operation.
func maskProfilerSignal() {
	runtime.LockOSThread()

	if err := unix.PthreadSigmask(unix.SIG_BLOCK, &profSet, nil); err != nil {
		runtime.UnlockOSThread()
		panic(fmt.Errorf("masking profiler signal: %w", err))
	}
}

// unmaskProfilerSignal removes SIGPROF from the underlying thread's signal
// mask, allowing it to be interrupted for profiling once again.
//
// It also unlocks the current goroutine from its underlying OS thread.
func unmaskProfilerSignal() {
	defer runtime.UnlockOSThread()

	if err := unix.PthreadSigmask(unix.SIG_UNBLOCK, &profSet, nil); err != nil {
		panic(fmt.Errorf("unmasking profiler signal: %w", err))
	}
}

const (
	// Signal is the nth bit in the bitfield.
	sigprofBit = int(unix.SIGPROF - 1)
	// The number of bits in one Sigset_t word.
	wordBits = int(unsafe.Sizeof(unix.Sigset_t{}.Val[0])) * 8
)

// sigsetAdd adds signal to set.
//
// Note: Sigset_t.Val's value type is uint32 or uint64 depending on the arch.
// This function must be able to deal with both and so must avoid any direct
// references to u32 or u64 types.
func sigsetAdd(set *unix.Sigset_t, signal unix.Signal) error {
	if signal < 1 {
		return fmt.Errorf("signal %d must be larger than 0", signal)
	}

	// For amd64, runtime.sigaddset() performs the following operation:
	// set[(signal-1)/32] |= 1 << ((uint32(signal) - 1) & 31)
	//
	// This trick depends on sigset being two u32's, causing a signal in the the
	// bottom 31 bits to be written to the low word if bit 32 is low, or the high
	// word if bit 32 is high.

	// Signal is the nth bit in the bitfield.
	bit := int(signal - 1)
	// Word within the sigset the bit needs to be written to.
	word := bit / wordBits

	if word >= len(set.Val) {
		return fmt.Errorf("signal %d does not fit within unix.Sigset_t", signal)
	}

	// Write the signal bit into its corresponding word at the corrected offset.
	set.Val[word] |= 1 << (bit % wordBits)

	return nil
}
