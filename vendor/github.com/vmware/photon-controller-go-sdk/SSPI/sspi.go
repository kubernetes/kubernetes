// +build windows

package SSPI

import (
	"fmt"
	"strings"
	"syscall"
	"unsafe"
)

var (
	secur32_dll           = syscall.NewLazyDLL("secur32.dll")
	initSecurityInterface = secur32_dll.NewProc("InitSecurityInterfaceW")
	sec_fn                *SecurityFunctionTable
)

func init() {
	ptr, _, _ := initSecurityInterface.Call()
	sec_fn = (*SecurityFunctionTable)(unsafe.Pointer(ptr))
}

const (
	SEC_E_OK                        = 0
	SECPKG_CRED_OUTBOUND            = 2
	SEC_WINNT_AUTH_IDENTITY_UNICODE = 2
	ISC_REQ_DELEGATE                = 0x00000001
	ISC_REQ_REPLAY_DETECT           = 0x00000004
	ISC_REQ_SEQUENCE_DETECT         = 0x00000008
	ISC_REQ_CONFIDENTIALITY         = 0x00000010
	ISC_REQ_CONNECTION              = 0x00000800
	SECURITY_NETWORK_DREP           = 0
	SEC_I_CONTINUE_NEEDED           = 0x00090312
	SEC_I_COMPLETE_NEEDED           = 0x00090313
	SEC_I_COMPLETE_AND_CONTINUE     = 0x00090314
	SECBUFFER_VERSION               = 0
	SECBUFFER_TOKEN                 = 2
	NTLMBUF_LEN                     = 12000
)

const ISC_REQ = ISC_REQ_CONFIDENTIALITY |
	ISC_REQ_REPLAY_DETECT |
	ISC_REQ_SEQUENCE_DETECT |
	ISC_REQ_CONNECTION |
	ISC_REQ_DELEGATE

type SecurityFunctionTable struct {
	dwVersion                  uint32
	EnumerateSecurityPackages  uintptr
	QueryCredentialsAttributes uintptr
	AcquireCredentialsHandle   uintptr
	FreeCredentialsHandle      uintptr
	Reserved2                  uintptr
	InitializeSecurityContext  uintptr
	AcceptSecurityContext      uintptr
	CompleteAuthToken          uintptr
	DeleteSecurityContext      uintptr
	ApplyControlToken          uintptr
	QueryContextAttributes     uintptr
	ImpersonateSecurityContext uintptr
	RevertSecurityContext      uintptr
	MakeSignature              uintptr
	VerifySignature            uintptr
	FreeContextBuffer          uintptr
	QuerySecurityPackageInfo   uintptr
	Reserved3                  uintptr
	Reserved4                  uintptr
	Reserved5                  uintptr
	Reserved6                  uintptr
	Reserved7                  uintptr
	Reserved8                  uintptr
	QuerySecurityContextToken  uintptr
	EncryptMessage             uintptr
	DecryptMessage             uintptr
}

type SEC_WINNT_AUTH_IDENTITY struct {
	User           *uint16
	UserLength     uint32
	Domain         *uint16
	DomainLength   uint32
	Password       *uint16
	PasswordLength uint32
	Flags          uint32
}

type TimeStamp struct {
	LowPart  uint32
	HighPart int32
}

type SecHandle struct {
	dwLower uintptr
	dwUpper uintptr
}

type SecBuffer struct {
	cbBuffer   uint32
	BufferType uint32
	pvBuffer   *byte
}

type SecBufferDesc struct {
	ulVersion uint32
	cBuffers  uint32
	pBuffers  *SecBuffer
}

type SSPIAuth struct {
	Domain   string
	UserName string
	Password string
	Service  string
	cred     SecHandle
	ctxt     SecHandle
}

type Auth interface {
	InitialBytes() ([]byte, error)
	NextBytes([]byte) ([]byte, error)
	Free()
}

// GetAuth returns SSPI auth object initialized with given params and true for success
// In case of error, it will return nil SSPI object and false for failure
func GetAuth(user, password, service, workstation string) (Auth, bool) {
	if user == "" {
		return &SSPIAuth{Service: service}, true
	}
	if !strings.ContainsRune(user, '\\') {
		return nil, false
	}
	domain_user := strings.SplitN(user, "\\", 2)
	return &SSPIAuth{
		Domain:   domain_user[0],
		UserName: domain_user[1],
		Password: password,
		Service:  service,
	}, true
}

func (auth *SSPIAuth) InitialBytes() ([]byte, error) {
	var identity *SEC_WINNT_AUTH_IDENTITY
	if auth.UserName != "" {
		identity = &SEC_WINNT_AUTH_IDENTITY{
			Flags:          SEC_WINNT_AUTH_IDENTITY_UNICODE,
			Password:       syscall.StringToUTF16Ptr(auth.Password),
			PasswordLength: uint32(len(auth.Password)),
			Domain:         syscall.StringToUTF16Ptr(auth.Domain),
			DomainLength:   uint32(len(auth.Domain)),
			User:           syscall.StringToUTF16Ptr(auth.UserName),
			UserLength:     uint32(len(auth.UserName)),
		}
	}
	var ts TimeStamp
	sec_ok, _, _ := syscall.Syscall9(sec_fn.AcquireCredentialsHandle,
		9,
		0,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr("Negotiate"))),
		SECPKG_CRED_OUTBOUND,
		0,
		uintptr(unsafe.Pointer(identity)),
		0,
		0,
		uintptr(unsafe.Pointer(&auth.cred)),
		uintptr(unsafe.Pointer(&ts)))
	if sec_ok != SEC_E_OK {
		return nil, fmt.Errorf("AcquireCredentialsHandle failed %x", sec_ok)
	}

	var buf SecBuffer
	var desc SecBufferDesc
	desc.ulVersion = SECBUFFER_VERSION
	desc.cBuffers = 1
	desc.pBuffers = &buf

	outbuf := make([]byte, NTLMBUF_LEN)
	buf.cbBuffer = NTLMBUF_LEN
	buf.BufferType = SECBUFFER_TOKEN
	buf.pvBuffer = &outbuf[0]

	var attrs uint32
	sec_ok, _, _ = syscall.Syscall12(sec_fn.InitializeSecurityContext,
		12,
		uintptr(unsafe.Pointer(&auth.cred)),
		0,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(auth.Service))),
		ISC_REQ,
		0,
		SECURITY_NETWORK_DREP,
		0,
		0,
		uintptr(unsafe.Pointer(&auth.ctxt)),
		uintptr(unsafe.Pointer(&desc)),
		uintptr(unsafe.Pointer(&attrs)),
		uintptr(unsafe.Pointer(&ts)))
	if sec_ok == SEC_I_COMPLETE_AND_CONTINUE ||
		sec_ok == SEC_I_COMPLETE_NEEDED {
		syscall.Syscall6(sec_fn.CompleteAuthToken,
			2,
			uintptr(unsafe.Pointer(&auth.ctxt)),
			uintptr(unsafe.Pointer(&desc)),
			0, 0, 0, 0)
	} else if sec_ok != SEC_E_OK &&
		sec_ok != SEC_I_CONTINUE_NEEDED {
		syscall.Syscall6(sec_fn.FreeCredentialsHandle,
			1,
			uintptr(unsafe.Pointer(&auth.cred)),
			0, 0, 0, 0, 0)
		return nil, fmt.Errorf("InitialBytes InitializeSecurityContext failed %x", sec_ok)
	}
	return outbuf[:buf.cbBuffer], nil
}

func (auth *SSPIAuth) NextBytes(bytes []byte) ([]byte, error) {
	var in_buf, out_buf SecBuffer
	var in_desc, out_desc SecBufferDesc

	in_desc.ulVersion = SECBUFFER_VERSION
	in_desc.cBuffers = 1
	in_desc.pBuffers = &in_buf

	out_desc.ulVersion = SECBUFFER_VERSION
	out_desc.cBuffers = 1
	out_desc.pBuffers = &out_buf

	in_buf.BufferType = SECBUFFER_TOKEN
	in_buf.pvBuffer = &bytes[0]
	in_buf.cbBuffer = uint32(len(bytes))

	outbuf := make([]byte, NTLMBUF_LEN)
	out_buf.BufferType = SECBUFFER_TOKEN
	out_buf.pvBuffer = &outbuf[0]
	out_buf.cbBuffer = NTLMBUF_LEN

	var attrs uint32
	var ts TimeStamp
	sec_ok, _, _ := syscall.Syscall12(sec_fn.InitializeSecurityContext,
		12,
		uintptr(unsafe.Pointer(&auth.cred)),
		uintptr(unsafe.Pointer(&auth.ctxt)),
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(auth.Service))),
		ISC_REQ,
		0,
		SECURITY_NETWORK_DREP,
		uintptr(unsafe.Pointer(&in_desc)),
		0,
		uintptr(unsafe.Pointer(&auth.ctxt)),
		uintptr(unsafe.Pointer(&out_desc)),
		uintptr(unsafe.Pointer(&attrs)),
		uintptr(unsafe.Pointer(&ts)))
	if sec_ok == SEC_I_COMPLETE_AND_CONTINUE ||
		sec_ok == SEC_I_COMPLETE_NEEDED {
		syscall.Syscall6(sec_fn.CompleteAuthToken,
			2,
			uintptr(unsafe.Pointer(&auth.ctxt)),
			uintptr(unsafe.Pointer(&out_desc)),
			0, 0, 0, 0)
	} else if sec_ok != SEC_E_OK &&
		sec_ok != SEC_I_CONTINUE_NEEDED {
		return nil, fmt.Errorf("NextBytes InitializeSecurityContext failed %x", sec_ok)
	}

	return outbuf[:out_buf.cbBuffer], nil
}

func (auth *SSPIAuth) Free() {
	syscall.Syscall6(sec_fn.DeleteSecurityContext,
		1,
		uintptr(unsafe.Pointer(&auth.ctxt)),
		0, 0, 0, 0, 0)
	syscall.Syscall6(sec_fn.FreeCredentialsHandle,
		1,
		uintptr(unsafe.Pointer(&auth.cred)),
		0, 0, 0, 0, 0)
}