//
// Copyright (c) 2014 David Mzareulyan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// +build windows

package sshagent

// see https://github.com/Yasushi/putty/blob/master/windows/winpgntc.c#L155
// see https://github.com/paramiko/paramiko/blob/master/paramiko/win_pageant.py

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	. "syscall"
	. "unsafe"
)

// Maximum size of message can be sent to pageant
const MaxMessageLen = 8192

var (
	ErrPageantNotFound = errors.New("pageant process not found")
	ErrSendMessage     = errors.New("error sending message")

	ErrMessageTooLong       = errors.New("message too long")
	ErrInvalidMessageFormat = errors.New("invalid message format")
	ErrResponseTooLong      = errors.New("response too long")
)

const (
	agentCopydataID = 0x804e50ba
	wmCopydata      = 74
)

type copyData struct {
	dwData uintptr
	cbData uint32
	lpData Pointer
}

var (
	lock sync.Mutex

	winFindWindow         = winAPI("user32.dll", "FindWindowW")
	winGetCurrentThreadID = winAPI("kernel32.dll", "GetCurrentThreadId")
	winSendMessage        = winAPI("user32.dll", "SendMessageW")
)

func winAPI(dllName, funcName string) func(...uintptr) (uintptr, uintptr, error) {
	proc := MustLoadDLL(dllName).MustFindProc(funcName)
	return func(a ...uintptr) (uintptr, uintptr, error) { return proc.Call(a...) }
}

// Available returns true if Pageant is running
func Available() bool { return pageantWindow() != 0 }

// Query sends message msg to Pageant and returns response or error.
// 'msg' is raw agent request with length prefix
// Response is raw agent response with length prefix
func query(msg []byte) ([]byte, error) {
	if len(msg) > MaxMessageLen {
		return nil, ErrMessageTooLong
	}

	msgLen := binary.BigEndian.Uint32(msg[:4])
	if len(msg) != int(msgLen)+4 {
		return nil, ErrInvalidMessageFormat
	}

	lock.Lock()
	defer lock.Unlock()

	paWin := pageantWindow()

	if paWin == 0 {
		return nil, ErrPageantNotFound
	}

	thID, _, _ := winGetCurrentThreadID()
	mapName := fmt.Sprintf("PageantRequest%08x", thID)
	pMapName, _ := UTF16PtrFromString(mapName)

	mmap, err := CreateFileMapping(InvalidHandle, nil, PAGE_READWRITE, 0, MaxMessageLen+4, pMapName)
	if err != nil {
		return nil, err
	}
	defer CloseHandle(mmap)

	ptr, err := MapViewOfFile(mmap, FILE_MAP_WRITE, 0, 0, 0)
	if err != nil {
		return nil, err
	}
	defer UnmapViewOfFile(ptr)

	mmSlice := (*(*[MaxMessageLen]byte)(Pointer(ptr)))[:]

	copy(mmSlice, msg)

	mapNameBytesZ := append([]byte(mapName), 0)

	cds := copyData{
		dwData: agentCopydataID,
		cbData: uint32(len(mapNameBytesZ)),
		lpData: Pointer(&(mapNameBytesZ[0])),
	}

	resp, _, _ := winSendMessage(paWin, wmCopydata, 0, uintptr(Pointer(&cds)))

	if resp == 0 {
		return nil, ErrSendMessage
	}

	respLen := binary.BigEndian.Uint32(mmSlice[:4])
	if respLen > MaxMessageLen-4 {
		return nil, ErrResponseTooLong
	}

	respData := make([]byte, respLen+4)
	copy(respData, mmSlice)

	return respData, nil
}

func pageantWindow() uintptr {
	nameP, _ := UTF16PtrFromString("Pageant")
	h, _, _ := winFindWindow(uintptr(Pointer(nameP)), uintptr(Pointer(nameP)))
	return h
}
