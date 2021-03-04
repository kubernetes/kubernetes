package goscaleio

type Error struct {
	Message                 string `xml:"message,attr"`
	MajorErrorCode          int    `xml:"majorErrorCode,attr"`
	MinorErrorCode          string `xml:"minorErrorCode,attr"`
	VendorSpecificErrorCode string `xml:"vendorSpecificErrorCode,attr,omitempty"`
	StackTrace              string `xml:"stackTrace,attr,omitempty"`
}

// type session struct {
// 	Link []*types.Link `xml:"Link"`
// }

type System struct {
	MdmMode                               string   `json:"mdmMode"`
	MdmClusterState                       string   `json:"mdmClusterState"`
	SecondaryMdmActorIPList               []string `json:"secondaryMdmActorIpList"`
	InstallID                             string   `json:"installId"`
	PrimaryActorIPList                    []string `json:"primaryMdmActorIpList"`
	SystemVersionName                     string   `json:"systemVersionName"`
	CapacityAlertHighThresholdPercent     int      `json:"capacityAlertHighThresholdPercent"`
	CapacityAlertCriticalThresholdPercent int      `json:"capacityAlertCriticalThresholdPercent"`
	RemoteReadOnlyLimitState              bool     `json:"remoteReadOnlyLimitState"`
	PrimaryMdmActorPort                   int      `json:"primaryMdmActorPort"`
	SecondaryMdmActorPort                 int      `json:"secondaryMdmActorPort"`
	TiebreakerMdmActorPort                int      `json:"tiebreakerMdmActorPort"`
	MdmManagementPort                     int      `json:"mdmManagementPort"`
	TiebreakerMdmIPList                   []string `json:"tiebreakerMdmIpList"`
	MdmManagementIPList                   []string `json:"mdmManagementIPList"`
	DefaultIsVolumeObfuscated             bool     `json:"defaultIsVolumeObfuscated"`
	RestrictedSdcModeEnabled              bool     `json:"restrictedSdcModeEnabled"`
	Swid                                  string   `json:"swid"`
	DaysInstalled                         int      `json:"daysInstalled"`
	MaxCapacityInGb                       string   `json:"maxCapacityInGb"`
	CapacityTimeLeftInDays                string   `json:"capacityTimeLeftInDays"`
	EnterpriseFeaturesEnabled             bool     `json:"enterpriseFeaturesEnabled"`
	IsInitialLicense                      bool     `json:"isInitialLicense"`
	Name                                  string   `json:"name"`
	ID                                    string   `json:"id"`
	Links                                 []*Link  `json:"links"`
}

type Link struct {
	Rel  string `json:"rel"`
	HREF string `json:"href"`
}

type BWC struct {
	TotalWeightInKb int `json:"totalWeightInKb"`
	NumOccured      int `json:"numOccured"`
	NumSeconds      int `json:"numSeconds"`
}

type Statistics struct {
	PrimaryReadFromDevBwc                    BWC `json:"primaryReadFromDevBwc"`
	NumOfStoragePools                        int `json:"numOfStoragePools"`
	ProtectedCapacityInKb                    int `json:"protectedCapacityInKb"`
	MovingCapacityInKb                       int `json:"movingCapacityInKb"`
	SnapCapacityInUseOccupiedInKb            int `json:"snapCapacityInUseOccupiedInKb"`
	SnapCapacityInUseInKb                    int `json:"snapCapacityInUseInKb"`
	ActiveFwdRebuildCapacityInKb             int `json:"activeFwdRebuildCapacityInKb"`
	DegradedHealthyVacInKb                   int `json:"degradedHealthyVacInKb"`
	ActiveMovingRebalanceJobs                int `json:"activeMovingRebalanceJobs"`
	TotalReadBwc                             BWC `json:"totalReadBwc"`
	MaxCapacityInKb                          int `json:"maxCapacityInKb"`
	PendingBckRebuildCapacityInKb            int `json:"pendingBckRebuildCapacityInKb"`
	ActiveMovingOutFwdRebuildJobs            int `json:"activeMovingOutFwdRebuildJobs"`
	CapacityLimitInKb                        int `json:"capacityLimitInKb"`
	SecondaryVacInKb                         int `json:"secondaryVacInKb"`
	PendingFwdRebuildCapacityInKb            int `json:"pendingFwdRebuildCapacityInKb"`
	ThinCapacityInUseInKb                    int `json:"thinCapacityInUseInKb"`
	AtRestCapacityInKb                       int `json:"atRestCapacityInKb"`
	ActiveMovingInBckRebuildJobs             int `json:"activeMovingInBckRebuildJobs"`
	DegradedHealthyCapacityInKb              int `json:"degradedHealthyCapacityInKb"`
	NumOfScsiInitiators                      int `json:"numOfScsiInitiators"`
	NumOfUnmappedVolumes                     int `json:"numOfUnmappedVolumes"`
	FailedCapacityInKb                       int `json:"failedCapacityInKb"`
	SecondaryReadFromDevBwc                  BWC `json:"secondaryReadFromDevBwc"`
	NumOfVolumes                             int `json:"numOfVolumes"`
	SecondaryWriteBwc                        BWC `json:"secondaryWriteBwc"`
	ActiveBckRebuildCapacityInKb             int `json:"activeBckRebuildCapacityInKb"`
	FailedVacInKb                            int `json:"failedVacInKb"`
	PendingMovingCapacityInKb                int `json:"pendingMovingCapacityInKb"`
	ActiveMovingInRebalanceJobs              int `json:"activeMovingInRebalanceJobs"`
	PendingMovingInRebalanceJobs             int `json:"pendingMovingInRebalanceJobs"`
	BckRebuildReadBwc                        BWC `json:"bckRebuildReadBwc"`
	DegradedFailedVacInKb                    int `json:"degradedFailedVacInKb"`
	NumOfSnapshots                           int `json:"numOfSnapshots"`
	RebalanceCapacityInKb                    int `json:"rebalanceCapacityInKb"`
	fwdRebuildReadBwc                        BWC `json:"fwdRebuildReadBwc"`
	NumOfSdc                                 int `json:"numOfSdc"`
	ActiveMovingInFwdRebuildJobs             int `json:"activeMovingInFwdRebuildJobs"`
	NumOfVtrees                              int `json:"numOfVtrees"`
	ThickCapacityInUseInKb                   int `json:"thickCapacityInUseInKb"`
	ProtectedVacInKb                         int `json:"protectedVacInKb"`
	PendingMovingInBckRebuildJobs            int `json:"pendingMovingInBckRebuildJobs"`
	CapacityAvailableForVolumeAllocationInKb int `json:"capacityAvailableForVolumeAllocationInKb"`
	PendingRebalanceCapacityInKb             int `json:"pendingRebalanceCapacityInKb"`
	PendingMovingRebalanceJobs               int `json:"pendingMovingRebalanceJobs"`
	NumOfProtectionDomains                   int `json:"numOfProtectionDomains"`
	NumOfSds                                 int `json:"numOfSds"`
	CapacityInUseInKb                        int `json:"capacityInUseInKb"`
	BckRebuildWriteBwc                       BWC `json:"bckRebuildWriteBwc"`
	DegradedFailedCapacityInKb               int `json:"degradedFailedCapacityInKb"`
	NumOfThinBaseVolumes                     int `json:"numOfThinBaseVolumes"`
	PendingMovingOutFwdRebuildJobs           int `json:"pendingMovingOutFwdRebuildJobs"`
	SecondaryReadBwc                         BWC `json:"secondaryReadBwc"`
	PendingMovingOutBckRebuildJobs           int `json:"pendingMovingOutBckRebuildJobs"`
	RebalanceWriteBwc                        BWC `json:"rebalanceWriteBwc"`
	PrimaryReadBwc                           BWC `json:"primaryReadBwc"`
	NumOfVolumesInDeletion                   int `json:"numOfVolumesInDeletion"`
	NumOfDevices                             int `json:"numOfDevices"`
	RebalanceReadBwc                         BWC `json:"rebalanceReadBwc"`
	InUseVacInKb                             int `json:"inUseVacInKb"`
	UnreachableUnusedCapacityInKb            int `json:"unreachableUnusedCapacityInKb"`
	TotalWriteBwc                            BWC `json:"totalWriteBwc"`
	SpareCapacityInKb                        int `json:"spareCapacityInKb"`
	ActiveMovingOutBckRebuildJobs            int `json:"activeMovingOutBckRebuildJobs"`
	PrimaryVacInKb                           int `json:"primaryVacInKb"`
	NumOfThickBaseVolumes                    int `json:"numOfThickBaseVolumes"`
	BckRebuildCapacityInKb                   int `json:"bckRebuildCapacityInKb"`
	NumOfMappedToAllVolumes                  int `json:"numOfMappedToAllVolumes"`
	ActiveMovingCapacityInKb                 int `json:"activeMovingCapacityInKb"`
	PendingMovingInFwdRebuildJobs            int `json:"pendingMovingInFwdRebuildJobs"`
	ActiveRebalanceCapacityInKb              int `json:"activeRebalanceCapacityInKb"`
	RmcacheSizeInKb                          int `json:"rmcacheSizeInKb"`
	FwdRebuildCapacityInKb                   int `json:"fwdRebuildCapacityInKb"`
	FwdRebuildWriteBwc                       BWC `json:"fwdRebuildWriteBwc"`
	PrimaryWriteBwc                          BWC `json:"primaryWriteBwc"`
}

type User struct {
	SystemID              string  `json:"systemId"`
	UserRole              string  `json:"userRole"`
	PasswordChangeRequire bool    `json:"passwordChangeRequired"`
	Name                  string  `json:"name"`
	ID                    string  `json:"id"`
	Links                 []*Link `json:"links"`
}

type ScsiInitiator struct {
	Name     string  `json:"name"`
	IQN      string  `json:"iqn"`
	SystemID string  `json:"systemID"`
	Links    []*Link `json:"links"`
}

type ProtectionDomain struct {
	SystemID                          string  `json:"systemId"`
	RebuildNetworkThrottlingInKbps    int     `json:"rebuildNetworkThrottlingInKbps"`
	RebalanceNetworkThrottlingInKbps  int     `json:"rebalanceNetworkThrottlingInKbps"`
	OverallIoNetworkThrottlingInKbps  int     `json:"overallIoNetworkThrottlingInKbps"`
	OverallIoNetworkThrottlingEnabled bool    `json:"overallIoNetworkThrottlingEnabled"`
	RebuildNetworkThrottlingEnabled   bool    `json:"rebuildNetworkThrottlingEnabled"`
	RebalanceNetworkThrottlingEnabled bool    `json:"rebalanceNetworkThrottlingEnabled"`
	ProtectionDomainState             string  `json:"protectionDomainState"`
	Name                              string  `json:"name"`
	ID                                string  `json:"id"`
	Links                             []*Link `json:"links"`
}

type ProtectionDomainParam struct {
	Name string `json:"name"`
}

type ProtectionDomainResp struct {
	ID string `json:"id"`
}

type Sdc struct {
	SystemID           string  `json:"systemId"`
	SdcApproved        bool    `json:"sdcApproved"`
	SdcIp              string  `json:"SdcIp"`
	OnVmWare           bool    `json:"onVmWare"`
	SdcGuid            string  `json:"sdcGuid"`
	MdmConnectionState string  `json:"mdmConnectionState"`
	Name               string  `json:"name"`
	ID                 string  `json:"id"`
	Links              []*Link `json:"links"`
}

type SdsIp struct {
	IP   string `json:"ip"`
	Role string `json:"role"`
}

type SdsIpList struct {
	SdsIP SdsIp `json:"SdsIp"`
}

type Sds struct {
	ID                           string       `json:"id"`
	Name                         string       `json:"name,omitempty"`
	ProtectionDomainID           string       `json:"protectionDomainId"`
	IPList                       []*SdsIpList `json:"ipList"`
	Port                         int          `json:"port,omitempty"`
	SdsState                     string       `json:"sdsState"`
	MembershipState              string       `json:"membershipState"`
	MdmConnectionState           string       `json:"mdmConnectionState"`
	DrlMode                      string       `json:"drlMode,omitempty"`
	RmcacheEnabled               bool         `json:"rmcacheEnabled,omitempty"`
	RmcacheSizeInKb              int          `json:"rmcacheSizeInKb,omitempty"`
	RmcacheFrozen                bool         `json:"rmcacheFrozen,omitempty"`
	IsOnVMware                   bool         `json:"isOnVmWare,omitempty"`
	FaultSetID                   string       `json:"faultSetId,omitempty"`
	NumOfIoBuffers               int          `json:"numOfIoBuffers,omitempty"`
	RmcacheMemoryAllocationState string       `json:"RmcacheMemoryAllocationState,omitempty"`
}

type DeviceInfo struct {
	DevicePath    string `json:"devicePath"`
	StoragePoolID string `json:"storagePoolId"`
	DeviceName    string `json:"deviceName,omitempty"`
}

type SdsParam struct {
	Name               string        `json:"name,omitempty"`
	IPList             []*SdsIpList  `json:"sdsIpList"`
	Port               int           `json:"sdsPort,omitempty"`
	DrlMode            string        `json:"drlMode,omitempty"`
	RmcacheEnabled     bool          `json:"rmcacheEnabled,omitempty"`
	RmcacheSizeInKb    int           `json:"rmcacheSizeInKb,omitempty"`
	RmcacheFrozen      bool          `json:"rmcacheFrozen,omitempty"`
	ProtectionDomainID string        `json:"protectionDomainId"`
	FaultSetID         string        `json:"faultSetId,omitempty"`
	NumOfIoBuffers     int           `json:"numOfIoBuffers,omitempty"`
	DeviceInfoList     []*DeviceInfo `json:"deviceInfoList,omitempty"`
	ForceClean         bool          `json:"forceClean,omitempty"`
	DeviceTestTimeSecs int           `json:"deviceTestTimeSecs ,omitempty"`
	DeviceTestMode     string        `json:"deviceTestMode,omitempty"`
}

type SdsResp struct {
	ID string `json:"id"`
}

type Device struct {
	ID                     string `json:"id,omitempty"`
	Name                   string `json:"name,omitempty"`
	DeviceCurrentPathname  string `json:"deviceCurrentPathname"`
	DeviceOriginalPathname string `json:"deviceOriginalPathname,omitempty"`
	DeviceState            string `json:"deviceState,omitempty"`
	ErrorState             string `json:"errorState,omitempty"`
	CapacityLimitInKb      int    `json:"capacityLimitInKb,omitempty"`
	MaxCapacityInKb        int    `json:"maxCapacityInKb,omitempty"`
	StoragePoolID          string `json:"storagePoolId"`
	SdsID                  string `json:"sdsId"`
}

type DeviceParam struct {
	Name                  string `json:"name,omitempty"`
	DeviceCurrentPathname string `json:"deviceCurrentPathname"`
	CapacityLimitInKb     int    `json:"capacityLimitInKb,omitempty"`
	StoragePoolID         string `json:"storagePoolId"`
	SdsID                 string `json:"sdsId"`
	TestTimeSecs          int    `json:"testTimeSecs,omitempty"`
	TestMode              string `json:"testMode,omitempty"`
}

type DeviceResp struct {
	ID string `json:"id"`
}

type StoragePool struct {
	ProtectionDomainID                               string  `json:"protectionDomainId"`
	RebalanceioPriorityPolicy                        string  `json:"rebalanceIoPriorityPolicy"`
	RebuildioPriorityPolicy                          string  `json:"rebuildIoPriorityPolicy"`
	RebuildioPriorityBwLimitPerDeviceInKbps          int     `json:"rebuildIoPriorityBwLimitPerDeviceInKbps"`
	RebuildioPriorityNumOfConcurrentIosPerDevice     int     `json:"rebuildIoPriorityNumOfConcurrentIosPerDevice"`
	RebalanceioPriorityNumOfConcurrentIosPerDevice   int     `json:"rebalanceIoPriorityNumOfConcurrentIosPerDevice"`
	RebalanceioPriorityBwLimitPerDeviceInKbps        int     `json:"rebalanceIoPriorityBwLimitPerDeviceInKbps"`
	RebuildioPriorityAppIopsPerDeviceThreshold       int     `json:"rebuildIoPriorityAppIopsPerDeviceThreshold"`
	RebalanceioPriorityAppIopsPerDeviceThreshold     int     `json:"rebalanceIoPriorityAppIopsPerDeviceThreshold"`
	RebuildioPriorityAppBwPerDeviceThresholdInKbps   int     `json:"rebuildIoPriorityAppBwPerDeviceThresholdInKbps"`
	RebalanceioPriorityAppBwPerDeviceThresholdInKbps int     `json:"rebalanceIoPriorityAppBwPerDeviceThresholdInKbps"`
	RebuildioPriorityQuietPeriodInMsec               int     `json:"rebuildIoPriorityQuietPeriodInMsec"`
	RebalanceioPriorityQuietPeriodInMsec             int     `json:"rebalanceIoPriorityQuietPeriodInMsec"`
	ZeroPaddingEnabled                               bool    `json:"zeroPaddingEnabled"`
	UseRmcache                                       bool    `json:"useRmcache"`
	SparePercentage                                  int     `json:"sparePercentage"`
	RmCacheWriteHandlingMode                         string  `json:"rmcacheWriteHandlingMode"`
	RebuildEnabled                                   bool    `json:"rebuildEnabled"`
	RebalanceEnabled                                 bool    `json:"rebalanceEnabled"`
	NumofParallelRebuildRebalanceJobsPerDevice       int     `json:"numOfParallelRebuildRebalanceJobsPerDevice"`
	Name                                             string  `json:"name"`
	ID                                               string  `json:"id"`
	Links                                            []*Link `json:"links"`
}

type StoragePoolParam struct {
	Name                     string `json:"name"`
	SparePercentage          int    `json:"sparePercentage,omitempty"`
	RebuildEnabled           bool   `json:"rebuildEnabled,omitempty"`
	RebalanceEnabled         bool   `json:"rebalanceEnabled,omitempty"`
	ProtectionDomainID       string `json:"protectionDomainId"`
	ZeroPaddingEnabled       bool   `json:"zeroPaddingEnabled,omitempty"`
	UseRmcache               bool   `json:"useRmcache,omitempty"`
	RmcacheWriteHandlingMode string `json:"rmcacheWriteHandlingMode,omitempty"`
}

type StoragePoolResp struct {
	ID string `json:"id"`
}

type MappedSdcInfo struct {
	SdcID         string `json:"sdcId"`
	SdcIP         string `json:"sdcIp"`
	LimitIops     int    `json:"limitIops"`
	LimitBwInMbps int    `json:"limitBwInMbps"`
}

type Volume struct {
	StoragePoolID           string           `json:"storagePoolId"`
	UseRmCache              bool             `json:"useRmcache"`
	MappingToAllSdcsEnabled bool             `json:"mappingToAllSdcsEnabled"`
	MappedSdcInfo           []*MappedSdcInfo `json:"mappedSdcInfo"`
	IsObfuscated            bool             `json:"isObfuscated"`
	VolumeType              string           `json:"volumeType"`
	ConsistencyGroupID      string           `json:"consistencyGroupId"`
	VTreeID                 string           `json:"vtreeId"`
	AncestorVolumeID        string           `json:"ancestorVolumeId"`
	MappedScsiInitiatorInfo string           `json:"mappedScsiInitiatorInfo"`
	SizeInKb                int              `json:"sizeInKb"`
	CreationTime            int              `json:"creationTime"`
	Name                    string           `json:"name"`
	ID                      string           `json:"id"`
	Links                   []*Link          `json:"links"`
}

type VolumeParam struct {
	ProtectionDomainID string `json:"protectionDomainId,omitempty"`
	StoragePoolID      string `json:"storagePoolId,omitempty"`
	UseRmCache         string `json:"useRmcache,omitempty"`
	VolumeType         string `json:"volumeType,omitempty"`
	VolumeSizeInKb     string `json:"volumeSizeInKb,omitempty"`
	Name               string `json:"name,omitempty"`
}

type VolumeResp struct {
	ID string `json:"id"`
}

type VolumeQeryIdByKeyParam struct {
	Name string `json:"name"`
}

type VolumeQeryBySelectedIdsParam struct {
	IDs []string `json:"ids"`
}

type MapVolumeSdcParam struct {
	SdcID                 string `json:"sdcId,omitempty"`
	AllowMultipleMappings string `json:"allowMultipleMappings,omitempty"`
	AllSdcs               string `json:"allSdcs,omitempty"`
}

type UnmapVolumeSdcParam struct {
	SdcID                string `json:"sdcId,omitempty"`
	IgnoreScsiInitiators string `json:"ignoreScsiInitiators,omitempty"`
	AllSdcs              string `json:"allSdcs,omitempty"`
}

type SnapshotDef struct {
	VolumeID     string `json:"volumeId,omitempty"`
	SnapshotName string `json:"snapshotName,omitempty"`
}

type SnapshotVolumesParam struct {
	SnapshotDefs []*SnapshotDef `json:"snapshotDefs"`
}

type SnapshotVolumesResp struct {
	VolumeIDList    []string `json:"volumeIdList"`
	SnapshotGroupID string   `json:"snapshotGroupId"`
}

type VTree struct {
	ID            string  `json:"id"`
	Name          string  `json:"name"`
	BaseVolumeID  string  `json:"baseVolumeId"`
	StoragePoolID string  `json:"storagePoolId"`
	Links         []*Link `json:"links"`
}

type RemoveVolumeParam struct {
	RemoveMode string `json:"removeMode"`
}
