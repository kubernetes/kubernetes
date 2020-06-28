package vmcompute

import (
	gcontext "context"
	"syscall"
	"time"

	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/Microsoft/hcsshim/internal/log"
	"github.com/Microsoft/hcsshim/internal/logfields"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/Microsoft/hcsshim/internal/timeout"
	"go.opencensus.io/trace"
)

//go:generate go run ../../mksyscall_windows.go -output zsyscall_windows.go vmcompute.go

//sys hcsEnumerateComputeSystems(query string, computeSystems **uint16, result **uint16) (hr error) = vmcompute.HcsEnumerateComputeSystems?
//sys hcsCreateComputeSystem(id string, configuration string, identity syscall.Handle, computeSystem *HcsSystem, result **uint16) (hr error) = vmcompute.HcsCreateComputeSystem?
//sys hcsOpenComputeSystem(id string, computeSystem *HcsSystem, result **uint16) (hr error) = vmcompute.HcsOpenComputeSystem?
//sys hcsCloseComputeSystem(computeSystem HcsSystem) (hr error) = vmcompute.HcsCloseComputeSystem?
//sys hcsStartComputeSystem(computeSystem HcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsStartComputeSystem?
//sys hcsShutdownComputeSystem(computeSystem HcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsShutdownComputeSystem?
//sys hcsTerminateComputeSystem(computeSystem HcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsTerminateComputeSystem?
//sys hcsPauseComputeSystem(computeSystem HcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsPauseComputeSystem?
//sys hcsResumeComputeSystem(computeSystem HcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsResumeComputeSystem?
//sys hcsGetComputeSystemProperties(computeSystem HcsSystem, propertyQuery string, properties **uint16, result **uint16) (hr error) = vmcompute.HcsGetComputeSystemProperties?
//sys hcsModifyComputeSystem(computeSystem HcsSystem, configuration string, result **uint16) (hr error) = vmcompute.HcsModifyComputeSystem?
//sys hcsRegisterComputeSystemCallback(computeSystem HcsSystem, callback uintptr, context uintptr, callbackHandle *HcsCallback) (hr error) = vmcompute.HcsRegisterComputeSystemCallback?
//sys hcsUnregisterComputeSystemCallback(callbackHandle HcsCallback) (hr error) = vmcompute.HcsUnregisterComputeSystemCallback?

//sys hcsCreateProcess(computeSystem HcsSystem, processParameters string, processInformation *HcsProcessInformation, process *HcsProcess, result **uint16) (hr error) = vmcompute.HcsCreateProcess?
//sys hcsOpenProcess(computeSystem HcsSystem, pid uint32, process *HcsProcess, result **uint16) (hr error) = vmcompute.HcsOpenProcess?
//sys hcsCloseProcess(process HcsProcess) (hr error) = vmcompute.HcsCloseProcess?
//sys hcsTerminateProcess(process HcsProcess, result **uint16) (hr error) = vmcompute.HcsTerminateProcess?
//sys hcsSignalProcess(process HcsProcess, options string, result **uint16) (hr error) = vmcompute.HcsSignalProcess?
//sys hcsGetProcessInfo(process HcsProcess, processInformation *HcsProcessInformation, result **uint16) (hr error) = vmcompute.HcsGetProcessInfo?
//sys hcsGetProcessProperties(process HcsProcess, processProperties **uint16, result **uint16) (hr error) = vmcompute.HcsGetProcessProperties?
//sys hcsModifyProcess(process HcsProcess, settings string, result **uint16) (hr error) = vmcompute.HcsModifyProcess?
//sys hcsGetServiceProperties(propertyQuery string, properties **uint16, result **uint16) (hr error) = vmcompute.HcsGetServiceProperties?
//sys hcsRegisterProcessCallback(process HcsProcess, callback uintptr, context uintptr, callbackHandle *HcsCallback) (hr error) = vmcompute.HcsRegisterProcessCallback?
//sys hcsUnregisterProcessCallback(callbackHandle HcsCallback) (hr error) = vmcompute.HcsUnregisterProcessCallback?

// errVmcomputeOperationPending is an error encountered when the operation is being completed asynchronously
const errVmcomputeOperationPending = syscall.Errno(0xC0370103)

// HcsSystem is the handle associated with a created compute system.
type HcsSystem syscall.Handle

// HcsProcess is the handle associated with a created process in a compute
// system.
type HcsProcess syscall.Handle

// HcsCallback is the handle associated with the function to call when events
// occur.
type HcsCallback syscall.Handle

// HcsProcessInformation is the structure used when creating or getting process
// info.
type HcsProcessInformation struct {
	// ProcessId is the pid of the created process.
	ProcessId uint32
	reserved  uint32
	// StdInput is the handle associated with the stdin of the process.
	StdInput syscall.Handle
	// StdOutput is the handle associated with the stdout of the process.
	StdOutput syscall.Handle
	// StdError is the handle associated with the stderr of the process.
	StdError syscall.Handle
}

func execute(ctx gcontext.Context, timeout time.Duration, f func() error) error {
	if timeout > 0 {
		var cancel gcontext.CancelFunc
		ctx, cancel = gcontext.WithTimeout(ctx, timeout)
		defer cancel()
	}

	done := make(chan error, 1)
	go func() {
		done <- f()
	}()
	select {
	case <-ctx.Done():
		if ctx.Err() == gcontext.DeadlineExceeded {
			log.G(ctx).WithField(logfields.Timeout, timeout).
				Warning("Syscall did not complete within operation timeout. This may indicate a platform issue. If it appears to be making no forward progress, obtain the stacks and see if there is a syscall stuck in the platform API for a significant length of time.")
		}
		return ctx.Err()
	case err := <-done:
		return err
	}
}

func HcsEnumerateComputeSystems(ctx gcontext.Context, query string) (computeSystems, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsEnumerateComputeSystems")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("query", query))

	return computeSystems, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var (
			computeSystemsp *uint16
			resultp         *uint16
		)
		err := hcsEnumerateComputeSystems(query, &computeSystemsp, &resultp)
		if computeSystemsp != nil {
			computeSystems = interop.ConvertAndFreeCoTaskMemString(computeSystemsp)
		}
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsCreateComputeSystem(ctx gcontext.Context, id string, configuration string, identity syscall.Handle) (computeSystem HcsSystem, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsCreateComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		if hr != errVmcomputeOperationPending {
			oc.SetSpanStatus(span, hr)
		}
	}()
	span.AddAttributes(
		trace.StringAttribute("id", id),
		trace.StringAttribute("configuration", configuration))

	return computeSystem, result, execute(ctx, timeout.SystemCreate, func() error {
		var resultp *uint16
		err := hcsCreateComputeSystem(id, configuration, identity, &computeSystem, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsOpenComputeSystem(ctx gcontext.Context, id string) (computeSystem HcsSystem, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsOpenComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()

	return computeSystem, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsOpenComputeSystem(id, &computeSystem, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsCloseComputeSystem(ctx gcontext.Context, computeSystem HcsSystem) (hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsCloseComputeSystem")
	defer span.End()
	defer func() { oc.SetSpanStatus(span, hr) }()

	return execute(ctx, timeout.SyscallWatcher, func() error {
		return hcsCloseComputeSystem(computeSystem)
	})
}

func HcsStartComputeSystem(ctx gcontext.Context, computeSystem HcsSystem, options string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsStartComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		if hr != errVmcomputeOperationPending {
			oc.SetSpanStatus(span, hr)
		}
	}()
	span.AddAttributes(trace.StringAttribute("options", options))

	return result, execute(ctx, timeout.SystemStart, func() error {
		var resultp *uint16
		err := hcsStartComputeSystem(computeSystem, options, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsShutdownComputeSystem(ctx gcontext.Context, computeSystem HcsSystem, options string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsShutdownComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		if hr != errVmcomputeOperationPending {
			oc.SetSpanStatus(span, hr)
		}
	}()
	span.AddAttributes(trace.StringAttribute("options", options))

	return result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsShutdownComputeSystem(computeSystem, options, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsTerminateComputeSystem(ctx gcontext.Context, computeSystem HcsSystem, options string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsTerminateComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		if hr != errVmcomputeOperationPending {
			oc.SetSpanStatus(span, hr)
		}
	}()
	span.AddAttributes(trace.StringAttribute("options", options))

	return result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsTerminateComputeSystem(computeSystem, options, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsPauseComputeSystem(ctx gcontext.Context, computeSystem HcsSystem, options string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsPauseComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		if hr != errVmcomputeOperationPending {
			oc.SetSpanStatus(span, hr)
		}
	}()
	span.AddAttributes(trace.StringAttribute("options", options))

	return result, execute(ctx, timeout.SystemPause, func() error {
		var resultp *uint16
		err := hcsPauseComputeSystem(computeSystem, options, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsResumeComputeSystem(ctx gcontext.Context, computeSystem HcsSystem, options string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsResumeComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		if hr != errVmcomputeOperationPending {
			oc.SetSpanStatus(span, hr)
		}
	}()
	span.AddAttributes(trace.StringAttribute("options", options))

	return result, execute(ctx, timeout.SystemResume, func() error {
		var resultp *uint16
		err := hcsResumeComputeSystem(computeSystem, options, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsGetComputeSystemProperties(ctx gcontext.Context, computeSystem HcsSystem, propertyQuery string) (properties, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsGetComputeSystemProperties")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("propertyQuery", propertyQuery))

	return properties, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var (
			propertiesp *uint16
			resultp     *uint16
		)
		err := hcsGetComputeSystemProperties(computeSystem, propertyQuery, &propertiesp, &resultp)
		if propertiesp != nil {
			properties = interop.ConvertAndFreeCoTaskMemString(propertiesp)
		}
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsModifyComputeSystem(ctx gcontext.Context, computeSystem HcsSystem, configuration string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsModifyComputeSystem")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("configuration", configuration))

	return result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsModifyComputeSystem(computeSystem, configuration, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsRegisterComputeSystemCallback(ctx gcontext.Context, computeSystem HcsSystem, callback uintptr, context uintptr) (callbackHandle HcsCallback, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsRegisterComputeSystemCallback")
	defer span.End()
	defer func() { oc.SetSpanStatus(span, hr) }()

	return callbackHandle, execute(ctx, timeout.SyscallWatcher, func() error {
		return hcsRegisterComputeSystemCallback(computeSystem, callback, context, &callbackHandle)
	})
}

func HcsUnregisterComputeSystemCallback(ctx gcontext.Context, callbackHandle HcsCallback) (hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsUnregisterComputeSystemCallback")
	defer span.End()
	defer func() { oc.SetSpanStatus(span, hr) }()

	return execute(ctx, timeout.SyscallWatcher, func() error {
		return hcsUnregisterComputeSystemCallback(callbackHandle)
	})
}

func HcsCreateProcess(ctx gcontext.Context, computeSystem HcsSystem, processParameters string) (processInformation HcsProcessInformation, process HcsProcess, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsCreateProcess")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("processParameters", processParameters))

	return processInformation, process, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsCreateProcess(computeSystem, processParameters, &processInformation, &process, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsOpenProcess(ctx gcontext.Context, computeSystem HcsSystem, pid uint32) (process HcsProcess, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsOpenProcess")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.Int64Attribute("pid", int64(pid)))

	return process, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsOpenProcess(computeSystem, pid, &process, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsCloseProcess(ctx gcontext.Context, process HcsProcess) (hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsCloseProcess")
	defer span.End()
	defer func() { oc.SetSpanStatus(span, hr) }()

	return execute(ctx, timeout.SyscallWatcher, func() error {
		return hcsCloseProcess(process)
	})
}

func HcsTerminateProcess(ctx gcontext.Context, process HcsProcess) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsTerminateProcess")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()

	return result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsTerminateProcess(process, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsSignalProcess(ctx gcontext.Context, process HcsProcess, options string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsSignalProcess")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("options", options))

	return result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsSignalProcess(process, options, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsGetProcessInfo(ctx gcontext.Context, process HcsProcess) (processInformation HcsProcessInformation, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsGetProcessInfo")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()

	return processInformation, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsGetProcessInfo(process, &processInformation, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsGetProcessProperties(ctx gcontext.Context, process HcsProcess) (processProperties, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsGetProcessProperties")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()

	return processProperties, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var (
			processPropertiesp *uint16
			resultp            *uint16
		)
		err := hcsGetProcessProperties(process, &processPropertiesp, &resultp)
		if processPropertiesp != nil {
			processProperties = interop.ConvertAndFreeCoTaskMemString(processPropertiesp)
		}
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsModifyProcess(ctx gcontext.Context, process HcsProcess, settings string) (result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsModifyProcess")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("settings", settings))

	return result, execute(ctx, timeout.SyscallWatcher, func() error {
		var resultp *uint16
		err := hcsModifyProcess(process, settings, &resultp)
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsGetServiceProperties(ctx gcontext.Context, propertyQuery string) (properties, result string, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsGetServiceProperties")
	defer span.End()
	defer func() {
		if result != "" {
			span.AddAttributes(trace.StringAttribute("result", result))
		}
		oc.SetSpanStatus(span, hr)
	}()
	span.AddAttributes(trace.StringAttribute("propertyQuery", propertyQuery))

	return properties, result, execute(ctx, timeout.SyscallWatcher, func() error {
		var (
			propertiesp *uint16
			resultp     *uint16
		)
		err := hcsGetServiceProperties(propertyQuery, &propertiesp, &resultp)
		if propertiesp != nil {
			properties = interop.ConvertAndFreeCoTaskMemString(propertiesp)
		}
		if resultp != nil {
			result = interop.ConvertAndFreeCoTaskMemString(resultp)
		}
		return err
	})
}

func HcsRegisterProcessCallback(ctx gcontext.Context, process HcsProcess, callback uintptr, context uintptr) (callbackHandle HcsCallback, hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsRegisterProcessCallback")
	defer span.End()
	defer func() { oc.SetSpanStatus(span, hr) }()

	return callbackHandle, execute(ctx, timeout.SyscallWatcher, func() error {
		return hcsRegisterProcessCallback(process, callback, context, &callbackHandle)
	})
}

func HcsUnregisterProcessCallback(ctx gcontext.Context, callbackHandle HcsCallback) (hr error) {
	ctx, span := trace.StartSpan(ctx, "HcsUnregisterProcessCallback")
	defer span.End()
	defer func() { oc.SetSpanStatus(span, hr) }()

	return execute(ctx, timeout.SyscallWatcher, func() error {
		return hcsUnregisterProcessCallback(callbackHandle)
	})
}
