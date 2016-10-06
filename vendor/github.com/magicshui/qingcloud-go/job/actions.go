package job

import (
	"github.com/magicshui/qingcloud-go"
)

// JOB 日志服务
type JOB struct {
	*qingcloud.Client
}

// NewClient 创建日志服务
func NewClient(clt *qingcloud.Client) *JOB {
	return &JOB{
		Client: clt,
	}
}

// DescribeJobsRequest 请求
type DescribeJobsRequest struct {
	JobsN     qingcloud.NumberedString
	StatusN   qingcloud.NumberedString
	JobAction qingcloud.String
	Verbose   qingcloud.Integer
	Offset    qingcloud.Integer
	Limit     qingcloud.Integer
}

// DescribeJobsResponse 返回
type DescribeJobsResponse struct {
	qingcloud.CommonResponse
	TotalCount int   `json:"total_count"`
	JobSet     []Job `json:"job_set"`
}

// DescribeJobs 获取一个或多个操作日志
// 可根据日志ID，动作，状态来获取日志列表。 如果不指定任何过滤条件，默认返回你触发的所有操作日志。
// 如果指定不存在的日志ID，或非法状态值，则会返回错误信息。
func DescribeJobs(c *qingcloud.Client, params DescribeJobsRequest) (DescribeJobsResponse, error) {
	var result DescribeJobsResponse
	err := c.Get("DescribeJobs", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
