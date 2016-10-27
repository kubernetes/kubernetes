package zone

import (
	"github.com/magicshui/qingcloud-go"
)

// DescribeZonesRequest 请求
type DescribeZonesRequest struct {
	ZonesN qingcloud.NumberedString
	StausN qingcloud.NumberedString
}

// DescribeZonesResponse 结果
type DescribeZonesResponse struct {
	ZoneSet    []Zone `json:"zone_set"`
	TotalCount int    `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeZones 获取可访问的区域列表。
func DescribeZones(c *qingcloud.Client, params DescribeZonesRequest) (DescribeZonesResponse, error) {
	var result DescribeZonesResponse
	err := c.Get("DescribeZones", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
