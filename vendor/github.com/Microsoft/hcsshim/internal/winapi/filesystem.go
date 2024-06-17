package winapi

//sys NtCreateFile(handle *uintptr, accessMask uint32, oa *ObjectAttributes, iosb *IOStatusBlock, allocationSize *uint64, fileAttributes uint32, shareAccess uint32, createDisposition uint32, createOptions uint32, eaBuffer *byte, eaLength uint32) (status uint32) = ntdll.NtCreateFile
//sys NtSetInformationFile(handle uintptr, iosb *IOStatusBlock, information uintptr, length uint32, class uint32) (status uint32) = ntdll.NtSetInformationFile

//sys NtOpenDirectoryObject(handle *uintptr, accessMask uint32, oa *ObjectAttributes) (status uint32) = ntdll.NtOpenDirectoryObject
//sys NtQueryDirectoryObject(handle uintptr, buffer *byte, length uint32, singleEntry bool, restartScan bool, context *uint32, returnLength *uint32)(status uint32) = ntdll.NtQueryDirectoryObject

const (
	FileLinkInformationClass          = 11
	FileDispositionInformationExClass = 64

	FILE_READ_ATTRIBUTES  = 0x0080
	FILE_WRITE_ATTRIBUTES = 0x0100
	DELETE                = 0x10000

	FILE_OPEN   = 1
	FILE_CREATE = 2

	FILE_LIST_DIRECTORY          = 0x00000001
	FILE_DIRECTORY_FILE          = 0x00000001
	FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
	FILE_OPEN_FOR_BACKUP_INTENT  = 0x00004000
	FILE_OPEN_REPARSE_POINT      = 0x00200000

	FILE_DISPOSITION_DELETE = 0x00000001

	OBJ_DONT_REPARSE = 0x1000

	STATUS_MORE_ENTRIES    = 0x105
	STATUS_NO_MORE_ENTRIES = 0x8000001a
)

// Select entries from FILE_INFO_BY_HANDLE_CLASS.
//
// C declaration:
//   typedef enum _FILE_INFO_BY_HANDLE_CLASS {
//       FileBasicInfo,
//       FileStandardInfo,
//       FileNameInfo,
//       FileRenameInfo,
//       FileDispositionInfo,
//       FileAllocationInfo,
//       FileEndOfFileInfo,
//       FileStreamInfo,
//       FileCompressionInfo,
//       FileAttributeTagInfo,
//       FileIdBothDirectoryInfo,
//       FileIdBothDirectoryRestartInfo,
//       FileIoPriorityHintInfo,
//       FileRemoteProtocolInfo,
//       FileFullDirectoryInfo,
//       FileFullDirectoryRestartInfo,
//       FileStorageInfo,
//       FileAlignmentInfo,
//       FileIdInfo,
//       FileIdExtdDirectoryInfo,
//       FileIdExtdDirectoryRestartInfo,
//       FileDispositionInfoEx,
//       FileRenameInfoEx,
//       FileCaseSensitiveInfo,
//       FileNormalizedNameInfo,
//       MaximumFileInfoByHandleClass
//   } FILE_INFO_BY_HANDLE_CLASS, *PFILE_INFO_BY_HANDLE_CLASS;
//
// Documentation: https://docs.microsoft.com/en-us/windows/win32/api/minwinbase/ne-minwinbase-file_info_by_handle_class
const (
	FileIdInfo = 18
)

type FileDispositionInformationEx struct {
	Flags uintptr
}

type IOStatusBlock struct {
	Status, Information uintptr
}

type ObjectAttributes struct {
	Length             uintptr
	RootDirectory      uintptr
	ObjectName         *UnicodeString
	Attributes         uintptr
	SecurityDescriptor uintptr
	SecurityQoS        uintptr
}

type ObjectDirectoryInformation struct {
	Name     UnicodeString
	TypeName UnicodeString
}

type FileLinkInformation struct {
	ReplaceIfExists bool
	RootDirectory   uintptr
	FileNameLength  uint32
	FileName        [1]uint16
}

// C declaration:
//   typedef struct _FILE_ID_INFO {
//       ULONGLONG   VolumeSerialNumber;
//       FILE_ID_128 FileId;
//   } FILE_ID_INFO, *PFILE_ID_INFO;
//
// Documentation: https://docs.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_info
type FILE_ID_INFO struct {
	VolumeSerialNumber uint64
	FileID             [16]byte
}
