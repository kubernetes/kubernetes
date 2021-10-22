// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows_test

import (
	"bytes"
	"debug/pe"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func TestWin32finddata(t *testing.T) {
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	path := filepath.Join(dir, "long_name.and_extension")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create %v: %v", path, err)
	}
	f.Close()

	type X struct {
		fd  windows.Win32finddata
		got byte
		pad [10]byte // to protect ourselves

	}
	var want byte = 2 // it is unlikely to have this character in the filename
	x := X{got: want}

	pathp, _ := windows.UTF16PtrFromString(path)
	h, err := windows.FindFirstFile(pathp, &(x.fd))
	if err != nil {
		t.Fatalf("FindFirstFile failed: %v", err)
	}
	err = windows.FindClose(h)
	if err != nil {
		t.Fatalf("FindClose failed: %v", err)
	}

	if x.got != want {
		t.Fatalf("memory corruption: want=%d got=%d", want, x.got)
	}
}

func TestFormatMessage(t *testing.T) {
	dll := windows.MustLoadDLL("netevent.dll")

	const TITLE_SC_MESSAGE_BOX uint32 = 0xC0001B75
	const flags uint32 = syscall.FORMAT_MESSAGE_FROM_HMODULE | syscall.FORMAT_MESSAGE_ARGUMENT_ARRAY | syscall.FORMAT_MESSAGE_IGNORE_INSERTS
	buf := make([]uint16, 300)
	_, err := windows.FormatMessage(flags, uintptr(dll.Handle), TITLE_SC_MESSAGE_BOX, 0, buf, nil)
	if err != nil {
		t.Fatalf("FormatMessage for handle=%x and errno=%x failed: %v", dll.Handle, TITLE_SC_MESSAGE_BOX, err)
	}
}

func abort(funcname string, err error) {
	panic(funcname + " failed: " + err.Error())
}

func ExampleLoadLibrary() {
	h, err := windows.LoadLibrary("kernel32.dll")
	if err != nil {
		abort("LoadLibrary", err)
	}
	defer windows.FreeLibrary(h)
	proc, err := windows.GetProcAddress(h, "GetVersion")
	if err != nil {
		abort("GetProcAddress", err)
	}
	r, _, _ := syscall.Syscall(uintptr(proc), 0, 0, 0, 0)
	major := byte(r)
	minor := uint8(r >> 8)
	build := uint16(r >> 16)
	print("windows version ", major, ".", minor, " (Build ", build, ")\n")
}

func TestTOKEN_ALL_ACCESS(t *testing.T) {
	if windows.TOKEN_ALL_ACCESS != 0xF01FF {
		t.Errorf("TOKEN_ALL_ACCESS = %x, want 0xF01FF", windows.TOKEN_ALL_ACCESS)
	}
}

func TestCreateWellKnownSid(t *testing.T) {
	sid, err := windows.CreateWellKnownSid(windows.WinBuiltinAdministratorsSid)
	if err != nil {
		t.Fatalf("Unable to create well known sid for administrators: %v", err)
	}
	if got, want := sid.String(), "S-1-5-32-544"; got != want {
		t.Fatalf("Builtin Administrators SID = %s, want %s", got, want)
	}
}

func TestPseudoTokens(t *testing.T) {
	version, err := windows.GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if ((version&0xffff)>>8)|((version&0xff)<<8) < 0x0602 {
		return
	}

	realProcessToken, err := windows.OpenCurrentProcessToken()
	if err != nil {
		t.Fatal(err)
	}
	defer realProcessToken.Close()
	realProcessUser, err := realProcessToken.GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}

	pseudoProcessToken := windows.GetCurrentProcessToken()
	pseudoProcessUser, err := pseudoProcessToken.GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}
	if !windows.EqualSid(realProcessUser.User.Sid, pseudoProcessUser.User.Sid) {
		t.Fatal("The real process token does not have the same as the pseudo process token")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	err = windows.RevertToSelf()
	if err != nil {
		t.Fatal(err)
	}

	pseudoThreadToken := windows.GetCurrentThreadToken()
	_, err = pseudoThreadToken.GetTokenUser()
	if err != windows.ERROR_NO_TOKEN {
		t.Fatal("Expected an empty thread token")
	}
	pseudoThreadEffectiveToken := windows.GetCurrentThreadEffectiveToken()
	pseudoThreadEffectiveUser, err := pseudoThreadEffectiveToken.GetTokenUser()
	if err != nil {
		t.Fatal(nil)
	}
	if !windows.EqualSid(realProcessUser.User.Sid, pseudoThreadEffectiveUser.User.Sid) {
		t.Fatal("The real process token does not have the same as the pseudo thread effective token, even though we aren't impersonating")
	}

	err = windows.ImpersonateSelf(windows.SecurityImpersonation)
	if err != nil {
		t.Fatal(err)
	}
	defer windows.RevertToSelf()
	pseudoThreadUser, err := pseudoThreadToken.GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}
	if !windows.EqualSid(realProcessUser.User.Sid, pseudoThreadUser.User.Sid) {
		t.Fatal("The real process token does not have the same as the pseudo thread token after impersonating self")
	}
}

func TestGUID(t *testing.T) {
	guid, err := windows.GenerateGUID()
	if err != nil {
		t.Fatal(err)
	}
	if guid.Data1 == 0 && guid.Data2 == 0 && guid.Data3 == 0 && guid.Data4 == [8]byte{} {
		t.Fatal("Got an all zero GUID, which is overwhelmingly unlikely")
	}
	want := fmt.Sprintf("{%08X-%04X-%04X-%04X-%012X}", guid.Data1, guid.Data2, guid.Data3, guid.Data4[:2], guid.Data4[2:])
	got := guid.String()
	if got != want {
		t.Fatalf("String = %q; want %q", got, want)
	}
	guid2, err := windows.GUIDFromString(got)
	if err != nil {
		t.Fatal(err)
	}
	if guid2 != guid {
		t.Fatalf("Did not parse string back to original GUID = %q; want %q", guid2, guid)
	}
	_, err = windows.GUIDFromString("not-a-real-guid")
	if err != syscall.Errno(windows.CO_E_CLASSSTRING) {
		t.Fatalf("Bad GUID string error = %v; want CO_E_CLASSSTRING", err)
	}
}

func TestKnownFolderPath(t *testing.T) {
	token, err := windows.OpenCurrentProcessToken()
	if err != nil {
		t.Fatal(err)
	}
	defer token.Close()
	profileDir, err := token.GetUserProfileDirectory()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(profileDir, "Desktop")
	got, err := windows.KnownFolderPath(windows.FOLDERID_Desktop, windows.KF_FLAG_DEFAULT)
	if err != nil {
		t.Fatal(err)
	}
	if want != got {
		t.Fatalf("Path = %q; want %q", got, want)
	}
}

func TestRtlGetVersion(t *testing.T) {
	version := windows.RtlGetVersion()
	major, minor, build := windows.RtlGetNtVersionNumbers()
	// Go is not explictly added to the application compatibility database, so
	// these two functions should return the same thing.
	if version.MajorVersion != major || version.MinorVersion != minor || version.BuildNumber != build {
		t.Fatalf("%d.%d.%d != %d.%d.%d", version.MajorVersion, version.MinorVersion, version.BuildNumber, major, minor, build)
	}
}

func TestGetNamedSecurityInfo(t *testing.T) {
	path, err := windows.GetSystemDirectory()
	if err != nil {
		t.Fatal(err)
	}
	sd, err := windows.GetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION)
	if err != nil {
		t.Fatal(err)
	}
	if !sd.IsValid() {
		t.Fatal("Invalid security descriptor")
	}
	sdOwner, _, err := sd.Owner()
	if err != nil {
		t.Fatal(err)
	}
	if !sdOwner.IsValid() {
		t.Fatal("Invalid security descriptor owner")
	}
}

func TestGetSecurityInfo(t *testing.T) {
	sd, err := windows.GetSecurityInfo(windows.CurrentProcess(), windows.SE_KERNEL_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatal(err)
	}
	if !sd.IsValid() {
		t.Fatal("Invalid security descriptor")
	}
	sdStr := sd.String()
	if !strings.HasPrefix(sdStr, "D:(A;") {
		t.Fatalf("DACL = %q; want D:(A;...", sdStr)
	}
}

func TestSddlConversion(t *testing.T) {
	sd, err := windows.SecurityDescriptorFromString("O:BA")
	if err != nil {
		t.Fatal(err)
	}
	if !sd.IsValid() {
		t.Fatal("Invalid security descriptor")
	}
	sdOwner, _, err := sd.Owner()
	if err != nil {
		t.Fatal(err)
	}
	if !sdOwner.IsValid() {
		t.Fatal("Invalid security descriptor owner")
	}
	if !sdOwner.IsWellKnown(windows.WinBuiltinAdministratorsSid) {
		t.Fatalf("Owner = %q; want S-1-5-32-544", sdOwner)
	}
}

func TestBuildSecurityDescriptor(t *testing.T) {
	const want = "O:SYD:(A;;GA;;;BA)"

	adminSid, err := windows.CreateWellKnownSid(windows.WinBuiltinAdministratorsSid)
	if err != nil {
		t.Fatal(err)
	}
	systemSid, err := windows.CreateWellKnownSid(windows.WinLocalSystemSid)
	if err != nil {
		t.Fatal(err)
	}

	access := []windows.EXPLICIT_ACCESS{{
		AccessPermissions: windows.GENERIC_ALL,
		AccessMode:        windows.GRANT_ACCESS,
		Trustee: windows.TRUSTEE{
			TrusteeForm:  windows.TRUSTEE_IS_SID,
			TrusteeType:  windows.TRUSTEE_IS_GROUP,
			TrusteeValue: windows.TrusteeValueFromSID(adminSid),
		},
	}}
	owner := &windows.TRUSTEE{
		TrusteeForm:  windows.TRUSTEE_IS_SID,
		TrusteeType:  windows.TRUSTEE_IS_USER,
		TrusteeValue: windows.TrusteeValueFromSID(systemSid),
	}

	sd, err := windows.BuildSecurityDescriptor(owner, nil, access, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	sd, err = sd.ToAbsolute()
	if err != nil {
		t.Fatal(err)
	}
	err = sd.SetSACL(nil, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := sd.String(); got != want {
		t.Fatalf("SD = %q; want %q", got, want)
	}
	sd, err = sd.ToSelfRelative()
	if err != nil {
		t.Fatal(err)
	}
	if got := sd.String(); got != want {
		t.Fatalf("SD = %q; want %q", got, want)
	}

	sd, err = windows.NewSecurityDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	acl, err := windows.ACLFromEntries(access, nil)
	if err != nil {
		t.Fatal(err)
	}
	err = sd.SetDACL(acl, true, false)
	if err != nil {
		t.Fatal(err)
	}
	err = sd.SetOwner(systemSid, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := sd.String(); got != want {
		t.Fatalf("SD = %q; want %q", got, want)
	}
	sd, err = sd.ToSelfRelative()
	if err != nil {
		t.Fatal(err)
	}
	if got := sd.String(); got != want {
		t.Fatalf("SD = %q; want %q", got, want)
	}
}

func TestGetDiskFreeSpaceEx(t *testing.T) {
	cwd, err := windows.UTF16PtrFromString(".")
	if err != nil {
		t.Fatalf(`failed to call UTF16PtrFromString("."): %v`, err)
	}
	var freeBytesAvailableToCaller, totalNumberOfBytes, totalNumberOfFreeBytes uint64
	if err := windows.GetDiskFreeSpaceEx(cwd, &freeBytesAvailableToCaller, &totalNumberOfBytes, &totalNumberOfFreeBytes); err != nil {
		t.Fatalf("failed to call GetDiskFreeSpaceEx: %v", err)
	}

	if freeBytesAvailableToCaller == 0 {
		t.Errorf("freeBytesAvailableToCaller: got 0; want > 0")
	}
	if totalNumberOfBytes == 0 {
		t.Errorf("totalNumberOfBytes: got 0; want > 0")
	}
	if totalNumberOfFreeBytes == 0 {
		t.Errorf("totalNumberOfFreeBytes: got 0; want > 0")
	}
}

func TestGetPreferredUILanguages(t *testing.T) {
	tab := map[string]func(flags uint32) ([]string, error){
		"GetProcessPreferredUILanguages": windows.GetProcessPreferredUILanguages,
		"GetThreadPreferredUILanguages":  windows.GetThreadPreferredUILanguages,
		"GetUserPreferredUILanguages":    windows.GetUserPreferredUILanguages,
		"GetSystemPreferredUILanguages":  windows.GetSystemPreferredUILanguages,
	}
	for fName, f := range tab {
		lang, err := f(windows.MUI_LANGUAGE_ID)
		if err != nil {
			t.Errorf(`failed to call %v(MUI_LANGUAGE_ID): %v`, fName, err)
		}
		for _, l := range lang {
			_, err := strconv.ParseUint(l, 16, 16)
			if err != nil {
				t.Errorf(`%v(MUI_LANGUAGE_ID) returned unexpected LANGID: %v`, fName, l)
			}
		}

		lang, err = f(windows.MUI_LANGUAGE_NAME)
		if err != nil {
			t.Errorf(`failed to call %v(MUI_LANGUAGE_NAME): %v`, fName, err)
		}
	}
}

func TestProcessWorkingSetSizeEx(t *testing.T) {
	// Grab a handle to the current process
	hProcess := windows.CurrentProcess()

	// Allocate memory to store the result of the query
	var minimumWorkingSetSize, maximumWorkingSetSize uintptr

	// Make the system-call
	var flag uint32
	windows.GetProcessWorkingSetSizeEx(hProcess, &minimumWorkingSetSize, &maximumWorkingSetSize, &flag)

	// Set the new limits to the current ones
	if err := windows.SetProcessWorkingSetSizeEx(hProcess, minimumWorkingSetSize, maximumWorkingSetSize, flag); err != nil {
		t.Error(err)
	}
}

func TestJobObjectInfo(t *testing.T) {
	jo, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		t.Fatalf("CreateJobObject failed: %v", err)
	}
	defer windows.CloseHandle(jo)

	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION

	err = windows.QueryInformationJobObject(jo, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)), nil)
	if err != nil {
		t.Fatalf("QueryInformationJobObject failed: %v", err)
	}

	const wantMemLimit = 4 * 1024

	info.BasicLimitInformation.LimitFlags |= windows.JOB_OBJECT_LIMIT_PROCESS_MEMORY
	info.ProcessMemoryLimit = wantMemLimit
	_, err = windows.SetInformationJobObject(jo, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)))
	if err != nil {
		t.Fatalf("SetInformationJobObject failed: %v", err)
	}

	err = windows.QueryInformationJobObject(jo, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)), nil)
	if err != nil {
		t.Fatalf("QueryInformationJobObject failed: %v", err)
	}

	if have := info.ProcessMemoryLimit; wantMemLimit != have {
		t.Errorf("ProcessMemoryLimit is wrong: want %v have %v", wantMemLimit, have)
	}
}

func TestIsWow64Process2(t *testing.T) {
	var processMachine, nativeMachine uint16
	err := windows.IsWow64Process2(windows.CurrentProcess(), &processMachine, &nativeMachine)
	if errors.Is(err, windows.ERROR_PROC_NOT_FOUND) {
		maj, min, build := windows.RtlGetNtVersionNumbers()
		if maj < 10 || (maj == 10 && min == 0 && build < 17763) {
			t.Skip("not available on older versions of Windows")
			return
		}
	}
	if err != nil {
		t.Fatalf("IsWow64Process2 failed: %v", err)
	}
	if processMachine == pe.IMAGE_FILE_MACHINE_UNKNOWN {
		processMachine = nativeMachine
	}
	switch {
	case processMachine == pe.IMAGE_FILE_MACHINE_AMD64 && runtime.GOARCH == "amd64":
	case processMachine == pe.IMAGE_FILE_MACHINE_I386 && runtime.GOARCH == "386":
	case processMachine == pe.IMAGE_FILE_MACHINE_ARMNT && runtime.GOARCH == "arm":
	case processMachine == pe.IMAGE_FILE_MACHINE_ARM64 && runtime.GOARCH == "arm64":
	default:
		t.Errorf("IsWow64Process2 is wrong: want %v have %v", runtime.GOARCH, processMachine)
	}
}

func TestNTStatusString(t *testing.T) {
	want := "The name limit for the local computer network adapter card was exceeded."
	got := windows.STATUS_TOO_MANY_NAMES.Error()
	if want != got {
		t.Errorf("NTStatus.Error did not return an expected error string - want %q; got %q", want, got)
	}
}

func TestNTStatusConversion(t *testing.T) {
	want := windows.ERROR_TOO_MANY_NAMES
	got := windows.STATUS_TOO_MANY_NAMES.Errno()
	if want != got {
		t.Errorf("NTStatus.Errno = %q (0x%x); want %q (0x%x)", got.Error(), got, want.Error(), want)
	}
}

func TestPEBFilePath(t *testing.T) {
	peb := windows.RtlGetCurrentPeb()
	if peb == nil || peb.Ldr == nil {
		t.Error("unable to retrieve PEB with valid Ldr")
	}
	var entry *windows.LDR_DATA_TABLE_ENTRY
	for cur := peb.Ldr.InMemoryOrderModuleList.Flink; cur != &peb.Ldr.InMemoryOrderModuleList; cur = cur.Flink {
		e := (*windows.LDR_DATA_TABLE_ENTRY)(unsafe.Pointer(uintptr(unsafe.Pointer(cur)) - unsafe.Offsetof(windows.LDR_DATA_TABLE_ENTRY{}.InMemoryOrderLinks)))
		if e.DllBase == peb.ImageBaseAddress {
			entry = e
			break
		}
	}
	if entry == nil {
		t.Error("unable to find Ldr entry for current process")
	}
	osPath, err := os.Executable()
	if err != nil {
		t.Errorf("unable to get path to current executable: %v", err)
	}
	pebPath := entry.FullDllName.String()
	if osPath != pebPath {
		t.Errorf("peb.Ldr.{entry}.FullDllName = %#q; want %#q", pebPath, osPath)
	}
	paramPath := peb.ProcessParameters.ImagePathName.String()
	if osPath != paramPath {
		t.Errorf("peb.ProcessParameters.ImagePathName.{entry}.ImagePathName = %#q; want %#q", paramPath, osPath)
	}
	osCwd, err := os.Getwd()
	if err != nil {
		t.Errorf("unable to get working directory: %v", err)
	}
	osCwd = filepath.Clean(osCwd)
	paramCwd := filepath.Clean(peb.ProcessParameters.CurrentDirectory.DosPath.String())
	if paramCwd != osCwd {
		t.Errorf("peb.ProcessParameters.CurrentDirectory.DosPath = %#q; want %#q", paramCwd, osCwd)
	}
}

func TestResourceExtraction(t *testing.T) {
	system32, err := windows.GetSystemDirectory()
	if err != nil {
		t.Errorf("unable to find system32 directory: %v", err)
	}
	cmd, err := windows.LoadLibrary(filepath.Join(system32, "cmd.exe"))
	if err != nil {
		t.Errorf("unable to load cmd.exe: %v", err)
	}
	defer windows.FreeLibrary(cmd)
	rsrc, err := windows.FindResource(cmd, windows.CREATEPROCESS_MANIFEST_RESOURCE_ID, windows.RT_MANIFEST)
	if err != nil {
		t.Errorf("unable to find cmd.exe manifest resource: %v", err)
	}
	manifest, err := windows.LoadResourceData(cmd, rsrc)
	if err != nil {
		t.Errorf("unable to load cmd.exe manifest resource data: %v", err)
	}
	if !bytes.Contains(manifest, []byte("</assembly>")) {
		t.Errorf("did not find </assembly> in manifest")
	}
}

func TestCommandLineRecomposition(t *testing.T) {
	const (
		maxCharsPerArg  = 35
		maxArgsPerTrial = 80
		doubleQuoteProb = 4
		singleQuoteProb = 1
		backSlashProb   = 3
		spaceProb       = 1
		trials          = 1000
	)
	randString := func(l int) []rune {
		s := make([]rune, l)
		for i := range s {
			s[i] = rand.Int31()
		}
		return s
	}
	mungeString := func(s []rune, char rune, timesInTen int) {
		if timesInTen < rand.Intn(10)+1 || len(s) == 0 {
			return
		}
		s[rand.Intn(len(s))] = char
	}
	argStorage := make([]string, maxArgsPerTrial+1)
	for i := 0; i < trials; i++ {
		args := argStorage[:rand.Intn(maxArgsPerTrial)+2]
		args[0] = "valid-filename-for-arg0"
		for j := 1; j < len(args); j++ {
			arg := randString(rand.Intn(maxCharsPerArg + 1))
			mungeString(arg, '"', doubleQuoteProb)
			mungeString(arg, '\'', singleQuoteProb)
			mungeString(arg, '\\', backSlashProb)
			mungeString(arg, ' ', spaceProb)
			args[j] = string(arg)
		}
		commandLine := windows.ComposeCommandLine(args)
		decomposedArgs, err := windows.DecomposeCommandLine(commandLine)
		if err != nil {
			t.Errorf("Unable to decompose %#q made from %v: %v", commandLine, args, err)
			continue
		}
		if len(decomposedArgs) != len(args) {
			t.Errorf("Incorrect decomposition length from %v to %#q to %v", args, commandLine, decomposedArgs)
			continue
		}
		badMatches := make([]int, 0, len(args))
		for i := range args {
			if args[i] != decomposedArgs[i] {
				badMatches = append(badMatches, i)
			}
		}
		if len(badMatches) != 0 {
			t.Errorf("Incorrect decomposition at indices %v from %v to %#q to %v", badMatches, args, commandLine, decomposedArgs)
			continue
		}
	}
}
