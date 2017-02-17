package ecs

import (
	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

type DescribeInstanceMonitorDataArgs struct {
	InstanceId string
	StartTime  util.ISO6801Time
	EndTime    util.ISO6801Time
	Period     int //Default 60s
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&instancemonitordatatype
type InstanceMonitorDataType struct {
	InstanceId        string
	CPU               int
	IntranetRX        int
	IntranetTX        int
	IntranetBandwidth int
	InternetRX        int
	InternetTX        int
	InternetBandwidth int
	IOPSRead          int
	IOPSWrite         int
	BPSRead           int
	BPSWrite          int
	TimeStamp         util.ISO6801Time
}

type DescribeInstanceMonitorDataResponse struct {
	common.Response
	MonitorData struct {
		InstanceMonitorData []InstanceMonitorDataType
	}
}

// DescribeInstanceMonitorData describes instance monitoring data
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/monitor&describeinstancemonitordata
func (client *Client) DescribeInstanceMonitorData(args *DescribeInstanceMonitorDataArgs) (monitorData []InstanceMonitorDataType, err error) {
	if args.Period == 0 {
		args.Period = 60
	}
	response := DescribeInstanceMonitorDataResponse{}
	err = client.Invoke("DescribeInstanceMonitorData", args, &response)
	if err != nil {
		return nil, err
	}
	return response.MonitorData.InstanceMonitorData, err
}

type DescribeEipMonitorDataArgs struct {
	AllocationId string
	StartTime    util.ISO6801Time
	EndTime      util.ISO6801Time
	Period       int //Default 60s
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&eipmonitordatatype
type EipMonitorDataType struct {
	EipRX        int
	EipTX        int
	EipFlow      int
	EipBandwidth int
	EipPackets   int
	TimeStamp    util.ISO6801Time
}

type DescribeEipMonitorDataResponse struct {
	common.Response
	EipMonitorDatas struct {
		EipMonitorData []EipMonitorDataType
	}
}

// DescribeEipMonitorData describes EIP monitoring data
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/monitor&describeeipmonitordata
func (client *Client) DescribeEipMonitorData(args *DescribeEipMonitorDataArgs) (monitorData []EipMonitorDataType, err error) {
	if args.Period == 0 {
		args.Period = 60
	}
	response := DescribeEipMonitorDataResponse{}
	err = client.Invoke("DescribeEipMonitorData", args, &response)
	if err != nil {
		return nil, err
	}
	return response.EipMonitorDatas.EipMonitorData, err
}

type DescribeDiskMonitorDataArgs struct {
	DiskId    string
	StartTime util.ISO6801Time
	EndTime   util.ISO6801Time
	Period    int //Default 60s
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&diskmonitordatatype
type DiskMonitorDataType struct {
	DiskId    string
	IOPSRead  int
	IOPSWrite int
	IOPSTotal int
	BPSRead   int
	BPSWrite  int
	BPSTotal  int
	TimeStamp util.ISO6801Time
}

type DescribeDiskMonitorDataResponse struct {
	common.Response
	TotalCount  int
	MonitorData struct {
		DiskMonitorData []DiskMonitorDataType
	}
}

// DescribeDiskMonitorData describes disk monitoring data
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/monitor&describediskmonitordata
func (client *Client) DescribeDiskMonitorData(args *DescribeDiskMonitorDataArgs) (monitorData []DiskMonitorDataType, totalCount int, err error) {
	if args.Period == 0 {
		args.Period = 60
	}
	response := DescribeDiskMonitorDataResponse{}
	err = client.Invoke("DescribeDiskMonitorData", args, &response)
	if err != nil {
		return nil, 0, err
	}
	return response.MonitorData.DiskMonitorData, response.TotalCount, err
}
