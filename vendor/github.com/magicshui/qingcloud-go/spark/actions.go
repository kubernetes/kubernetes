package spark

import (
	"github.com/magicshui/qingcloud-go"
)

type AddSparkNodesRequest struct {
	Spark                 qingcloud.String
	NodeCount             qingcloud.Integer
	SparkNodeName         qingcloud.String
	PrivateIpsNRole       qingcloud.NumberedString
	PrivateIpsNPrivateIps qingcloud.NumberedString
}
type AddSparkNodesResponse struct {
	SparkId         string   `json:"spark_id"`
	SparkNewNodeIds []string `json:"spark_new_node_ids"`
	qingcloud.CommonResponse
}

// AddSparkNodes 给 Spark 服务添加一个或多个 worker 节点。
func AddSparkNodes(c *qingcloud.Client, params AddSparkNodesRequest) (AddSparkNodesResponse, error) {
	var result AddSparkNodesResponse
	err := c.Get("AddSparkNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteSparkNodesRequest struct {
	Spark       qingcloud.String
	SparkNodesN qingcloud.NumberedString
}
type DeleteSparkNodesResponse struct {
	SparkId string `json:"spark_id"`
	qingcloud.CommonResponse
}

// DeleteSparkNodes 删除 Spark 服务 worker 节点。
func DeleteSparkNodes(c *qingcloud.Client, params DeleteSparkNodesRequest) (DeleteSparkNodesResponse, error) {
	var result DeleteSparkNodesResponse
	err := c.Get("AddSparkNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StartSparksRequest struct {
	SparksN qingcloud.NumberedString
}
type StartSparksResponse qingcloud.CommonResponse

// StartSparksResponse 启动一台或多台 Spark 服务。
func StartSparks(c *qingcloud.Client, params StartSparksRequest) (StartSparksResponse, error) {
	var result StartSparksResponse
	err := c.Get("StartSparks", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StopSparksRequest struct {
	SparksN qingcloud.NumberedString
}
type StopSparksResponse qingcloud.CommonResponse

// StopSparks 关闭一台或多台 Spark 服务。该操作将关闭 Spark 服务的所有 Spark 节点。
func StopSparks(c *qingcloud.Client, params StopSparksRequest) (StopSparksResponse, error) {
	var result StopSparksResponse
	err := c.Get("StopSparks", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
