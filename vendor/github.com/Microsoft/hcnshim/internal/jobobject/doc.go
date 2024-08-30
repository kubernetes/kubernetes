// This package provides higher level constructs for the win32 job object API.
// Most of the core creation and management functions are already present in "golang.org/x/sys/windows"
// (CreateJobObject, AssignProcessToJobObject, etc.) as well as most of the limit information
// structs and associated limit flags. Whatever is not present from the job object API
// in golang.org/x/sys/windows is located in /internal/winapi.
//
// https://docs.microsoft.com/en-us/windows/win32/procthread/job-objects
package jobobject
