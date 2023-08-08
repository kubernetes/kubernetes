// This package provides utilities for efficiently performing Win32 IO operations in Go.
// Currently, this package is provides support for genreal IO and management of
//   - named pipes
//   - files
//   - [Hyper-V sockets]
//
// This code is similar to Go's [net] package, and uses IO completion ports to avoid
// blocking IO on system threads, allowing Go to reuse the thread to schedule other goroutines.
//
// This limits support to Windows Vista and newer operating systems.
//
// Additionally, this package provides support for:
//   - creating and managing GUIDs
//   - writing to [ETW]
//   - opening and manageing VHDs
//   - parsing [Windows Image files]
//   - auto-generating Win32 API code
//
// [Hyper-V sockets]: https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/user-guide/make-integration-service
// [ETW]: https://docs.microsoft.com/en-us/windows-hardware/drivers/devtest/event-tracing-for-windows--etw-
// [Windows Image files]: https://docs.microsoft.com/en-us/windows-hardware/manufacture/desktop/work-with-windows-images
package winio
