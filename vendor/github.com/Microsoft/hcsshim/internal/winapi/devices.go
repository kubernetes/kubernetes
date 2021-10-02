package winapi

import "github.com/Microsoft/go-winio/pkg/guid"

//sys CMGetDeviceIDListSize(pulLen *uint32, pszFilter *byte, uFlags uint32) (hr error) = cfgmgr32.CM_Get_Device_ID_List_SizeA
//sys CMGetDeviceIDList(pszFilter *byte, buffer *byte, bufferLen uint32, uFlags uint32) (hr error)= cfgmgr32.CM_Get_Device_ID_ListA
//sys CMLocateDevNode(pdnDevInst *uint32, pDeviceID string, uFlags uint32) (hr error) = cfgmgr32.CM_Locate_DevNodeW
//sys CMGetDevNodeProperty(dnDevInst uint32, propertyKey *DevPropKey, propertyType *uint32, propertyBuffer *uint16, propertyBufferSize *uint32, uFlags uint32) (hr error) = cfgmgr32.CM_Get_DevNode_PropertyW

type DevPropKey struct {
	Fmtid guid.GUID
	Pid   uint32
}
