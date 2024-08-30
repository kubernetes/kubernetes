package winapi

// Offline registry management API

type ORHKey uintptr

type RegType uint32

const (
	// Registry value types: https://docs.microsoft.com/en-us/windows/win32/sysinfo/registry-value-types
	REG_TYPE_NONE                       RegType = 0
	REG_TYPE_SZ                         RegType = 1
	REG_TYPE_EXPAND_SZ                  RegType = 2
	REG_TYPE_BINARY                     RegType = 3
	REG_TYPE_DWORD                      RegType = 4
	REG_TYPE_DWORD_LITTLE_ENDIAN        RegType = 4
	REG_TYPE_DWORD_BIG_ENDIAN           RegType = 5
	REG_TYPE_LINK                       RegType = 6
	REG_TYPE_MULTI_SZ                   RegType = 7
	REG_TYPE_RESOURCE_LIST              RegType = 8
	REG_TYPE_FULL_RESOURCE_DESCRIPTOR   RegType = 9
	REG_TYPE_RESOURCE_REQUIREMENTS_LIST RegType = 10
	REG_TYPE_QWORD                      RegType = 11
	REG_TYPE_QWORD_LITTLE_ENDIAN        RegType = 11
)

//sys ORCreateHive(key *ORHKey) (win32err error) = offreg.ORCreateHive
//sys ORMergeHives(hiveHandles []ORHKey, result *ORHKey) (win32err error) = offreg.ORMergeHives
//sys OROpenHive(hivePath string, result *ORHKey) (win32err error) = offreg.OROpenHive
//sys ORCloseHive(handle ORHKey) (win32err error) = offreg.ORCloseHive
//sys ORSaveHive(handle ORHKey, hivePath string, osMajorVersion uint32, osMinorVersion uint32) (win32err error) = offreg.ORSaveHive
//sys OROpenKey(handle ORHKey, subKey string, result *ORHKey) (win32err error) = offreg.OROpenKey
//sys ORCloseKey(handle ORHKey) (win32err error) = offreg.ORCloseKey
//sys ORCreateKey(handle ORHKey, subKey string, class uintptr, options uint32, securityDescriptor uintptr, result *ORHKey, disposition *uint32) (win32err error) = offreg.ORCreateKey
//sys ORDeleteKey(handle ORHKey, subKey string) (win32err error) = offreg.ORDeleteKey
//sys ORGetValue(handle ORHKey, subKey string, value string, valueType *uint32, data *byte, dataLen *uint32) (win32err error) = offreg.ORGetValue
//sys ORSetValue(handle ORHKey, valueName string, valueType uint32, data *byte, dataLen uint32) (win32err error) = offreg.ORSetValue
