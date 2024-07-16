//go:build windows

package winapi

import (
	"syscall"

	"golang.org/x/sys/windows"
)

const UserNameCharLimit = 20

const (
	USER_PRIV_GUEST uint32 = iota
	USER_PRIV_USER
	USER_PRIV_ADMIN
)

const (
	UF_NORMAL_ACCOUNT     = 0x00200
	UF_DONT_EXPIRE_PASSWD = 0x10000
)

const NERR_UserNotFound = syscall.Errno(0x8AD)

//	typedef struct _LOCALGROUP_MEMBERS_INFO_0 {
//		PSID lgrmi0_sid;
//	} LOCALGROUP_MEMBERS_INFO_0, *PLOCALGROUP_MEMBERS_INFO_0, *LPLOCALGROUP_MEMBERS_INFO_0;
type LocalGroupMembersInfo0 struct {
	Sid *windows.SID
}

//	typedef struct _LOCALGROUP_INFO_1 {
//		LPWSTR lgrpi1_name;
//		LPWSTR lgrpi1_comment;
//	} LOCALGROUP_INFO_1, *PLOCALGROUP_INFO_1, *LPLOCALGROUP_INFO_1;
type LocalGroupInfo1 struct {
	Name    *uint16
	Comment *uint16
}

//	typedef struct _USER_INFO_1 {
//		LPWSTR usri1_name;
//		LPWSTR usri1_password;
//		DWORD  usri1_password_age;
//		DWORD  usri1_priv;
//		LPWSTR usri1_home_dir;
//		LPWSTR usri1_comment;
//		DWORD  usri1_flags;
//		LPWSTR usri1_script_path;
//	} USER_INFO_1, *PUSER_INFO_1, *LPUSER_INFO_1;
type UserInfo1 struct {
	Name        *uint16
	Password    *uint16
	PasswordAge uint32
	Priv        uint32
	HomeDir     *uint16
	Comment     *uint16
	Flags       uint32
	ScriptPath  *uint16
}

// NET_API_STATUS NET_API_FUNCTION NetLocalGroupGetInfo(
// 	[in]  LPCWSTR servername,
// 	[in]  LPCWSTR groupname,
// 	[in]  DWORD   level,
// 	[out] LPBYTE  *bufptr
// );
//
//sys netLocalGroupGetInfo(serverName *uint16, groupName *uint16, level uint32, bufptr **byte) (status error) = netapi32.NetLocalGroupGetInfo

// NetLocalGroupGetInfo is a slightly go friendlier wrapper around the NetLocalGroupGetInfo function. Instead of taking in *uint16's, it takes in
// go strings and does the conversion internally.
func NetLocalGroupGetInfo(serverName, groupName string, level uint32, bufPtr **byte) (err error) {
	var (
		serverNameUTF16 *uint16
		groupNameUTF16  *uint16
	)
	if serverName != "" {
		serverNameUTF16, err = windows.UTF16PtrFromString(serverName)
		if err != nil {
			return err
		}
	}
	if groupName != "" {
		groupNameUTF16, err = windows.UTF16PtrFromString(groupName)
		if err != nil {
			return err
		}
	}
	return netLocalGroupGetInfo(
		serverNameUTF16,
		groupNameUTF16,
		level,
		bufPtr,
	)
}

// NET_API_STATUS NET_API_FUNCTION NetUserAdd(
// 	[in]  LPCWSTR servername,
// 	[in]  DWORD   level,
// 	[in]  LPBYTE  buf,
// 	[out] LPDWORD parm_err
// );
//
//sys netUserAdd(serverName *uint16, level uint32, buf *byte, parm_err *uint32) (status error) = netapi32.NetUserAdd

// NetUserAdd is a slightly go friendlier wrapper around the NetUserAdd function. Instead of taking in *uint16's, it takes in
// go strings and does the conversion internally.
func NetUserAdd(serverName string, level uint32, buf *byte, parm_err *uint32) (err error) {
	var serverNameUTF16 *uint16
	if serverName != "" {
		serverNameUTF16, err = windows.UTF16PtrFromString(serverName)
		if err != nil {
			return err
		}
	}
	return netUserAdd(
		serverNameUTF16,
		level,
		buf,
		parm_err,
	)
}

// NET_API_STATUS NET_API_FUNCTION NetUserDel(
// 	[in] LPCWSTR servername,
// 	[in] LPCWSTR username
// );
//
//sys netUserDel(serverName *uint16, username *uint16) (status error) = netapi32.NetUserDel

// NetUserDel is a slightly go friendlier wrapper around the NetUserDel function. Instead of taking in *uint16's, it takes in
// go strings and does the conversion internally.
func NetUserDel(serverName, userName string) (err error) {
	var (
		serverNameUTF16 *uint16
		userNameUTF16   *uint16
	)
	if serverName != "" {
		serverNameUTF16, err = windows.UTF16PtrFromString(serverName)
		if err != nil {
			return err
		}
	}
	if userName != "" {
		userNameUTF16, err = windows.UTF16PtrFromString(userName)
		if err != nil {
			return err
		}
	}
	return netUserDel(
		serverNameUTF16,
		userNameUTF16,
	)
}

// NET_API_STATUS NET_API_FUNCTION NetLocalGroupAddMembers(
// 	[in] LPCWSTR servername,
// 	[in] LPCWSTR groupname,
// 	[in] DWORD   level,
// 	[in] LPBYTE  buf,
// 	[in] DWORD   totalentries
// );
//
//sys netLocalGroupAddMembers(serverName *uint16, groupName *uint16, level uint32, buf *byte, totalEntries uint32) (status error) = netapi32.NetLocalGroupAddMembers

// NetLocalGroupAddMembers is a slightly go friendlier wrapper around the NetLocalGroupAddMembers function. Instead of taking in *uint16's, it takes in
// go strings and does the conversion internally.
func NetLocalGroupAddMembers(serverName, groupName string, level uint32, buf *byte, totalEntries uint32) (err error) {
	var (
		serverNameUTF16 *uint16
		groupNameUTF16  *uint16
	)
	if serverName != "" {
		serverNameUTF16, err = windows.UTF16PtrFromString(serverName)
		if err != nil {
			return err
		}
	}
	if groupName != "" {
		groupNameUTF16, err = windows.UTF16PtrFromString(groupName)
		if err != nil {
			return err
		}
	}
	return netLocalGroupAddMembers(
		serverNameUTF16,
		groupNameUTF16,
		level,
		buf,
		totalEntries,
	)
}
