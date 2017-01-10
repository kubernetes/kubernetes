package slb

import "github.com/denverdino/aliyungo/common"

type DescribeRegionsArgs struct {
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&regiontype
type RegionType struct {
	RegionId  common.Region
	LocalName string
}

type DescribeRegionsResponse struct {
	common.Response
	Regions struct {
		Region []RegionType
	}
}

// DescribeRegions describes regions
//
// You can read doc at http://docs.aliyun.com/#/pub/slb/api-reference/api-related-loadbalancer&DescribeRegions
func (client *Client) DescribeRegions() (regions []RegionType, err error) {
	response := DescribeRegionsResponse{}

	err = client.Invoke("DescribeRegions", &DescribeRegionsArgs{}, &response)

	if err != nil {
		return []RegionType{}, err
	}
	return response.Regions.Region, nil
}
