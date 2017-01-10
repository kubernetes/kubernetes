package ecs

import (
	"time"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

type DescribeRouteTablesArgs struct {
	VRouterId    string
	RouteTableId string
	common.Pagination
}

type RouteTableType string

const (
	RouteTableSystem = RouteTableType("System")
	RouteTableCustom = RouteTableType("Custom")
)

type RouteEntryStatus string

const (
	RouteEntryStatusPending   = RouteEntryStatus("Pending")
	RouteEntryStatusAvailable = RouteEntryStatus("Available")
	RouteEntryStatusModifying = RouteEntryStatus("Modifying")
)

type NextHopListType struct {
	NextHopList struct {
		NextHopItem []NextHopItemType
	}
}

type NextHopItemType struct {
	NextHopType string
	NextHopId   string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&routeentrysettype
type RouteEntrySetType struct {
	RouteTableId         string
	DestinationCidrBlock string
	Type                 RouteTableType
	NextHopType          string
	NextHopId            string
	NextHopList          NextHopListType
	InstanceId           string
	Status               RouteEntryStatus // enum Pending | Available | Modifying
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&routetablesettype
type RouteTableSetType struct {
	VRouterId    string
	RouteTableId string
	RouteEntrys  struct {
		RouteEntry []RouteEntrySetType
	}
	RouteTableType RouteTableType
	CreationTime   util.ISO6801Time
}

type DescribeRouteTablesResponse struct {
	common.Response
	common.PaginationResult
	RouteTables struct {
		RouteTable []RouteTableSetType
	}
}

// DescribeRouteTables describes Virtual Routers
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/routertable&describeroutetables
func (client *Client) DescribeRouteTables(args *DescribeRouteTablesArgs) (routeTables []RouteTableSetType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeRouteTablesResponse{}

	err = client.Invoke("DescribeRouteTables", args, &response)

	if err == nil {
		return response.RouteTables.RouteTable, &response.PaginationResult, nil
	}

	return nil, nil, err
}

type NextHopType string

const (
	NextHopIntance = NextHopType("Instance") //Default
	NextHopTunnel  = NextHopType("Tunnel")
)

type CreateRouteEntryArgs struct {
	RouteTableId         string
	DestinationCidrBlock string
	NextHopType          NextHopType
	NextHopId            string
	ClientToken          string
}

type CreateRouteEntryResponse struct {
	common.Response
}

// CreateRouteEntry creates route entry
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/routertable&createrouteentry
func (client *Client) CreateRouteEntry(args *CreateRouteEntryArgs) error {
	response := CreateRouteEntryResponse{}
	return client.Invoke("CreateRouteEntry", args, &response)
}

type DeleteRouteEntryArgs struct {
	RouteTableId         string
	DestinationCidrBlock string
	NextHopId            string
}

type DeleteRouteEntryResponse struct {
	common.Response
}

// DeleteRouteEntry deletes route entry
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/routertable&deleterouteentry
func (client *Client) DeleteRouteEntry(args *DeleteRouteEntryArgs) error {
	response := DeleteRouteEntryResponse{}
	return client.Invoke("DeleteRouteEntry", args, &response)
}

// WaitForAllRouteEntriesAvailable waits for all route entries to Available status
func (client *Client) WaitForAllRouteEntriesAvailable(vrouterId string, routeTableId string, timeout int) error {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	args := DescribeRouteTablesArgs{
		VRouterId:    vrouterId,
		RouteTableId: routeTableId,
	}
	for {

		routeTables, _, err := client.DescribeRouteTables(&args)

		if err != nil {
			return err
		}
		if len(routeTables) == 0 {
			return common.GetClientErrorFromString("Not found")
		}
		success := true

	loop:
		for _, routeTable := range routeTables {
			for _, routeEntry := range routeTable.RouteEntrys.RouteEntry {
				if routeEntry.Status != RouteEntryStatusAvailable {
					success = false
					break loop
				}
			}
		}
		if success {
			break
		}
		timeout = timeout - DefaultWaitForInterval
		if timeout <= 0 {
			return common.GetClientErrorFromString("Timeout")
		}
		time.Sleep(DefaultWaitForInterval * time.Second)
	}
	return nil
}
