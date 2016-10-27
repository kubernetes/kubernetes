package mongo

import (
	"github.com/magicshui/qingcloud-go"
)

type MONGO struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *MONGO {
	return &MONGO{
		Client: clt,
	}
}

type DescribeMongoNodesRequest struct {
	Mongo  qingcloud.String
	Offset qingcloud.Integer
	Limit  qingcloud.Integer
}
type DescribeMongoNodesResponse struct {
	qingcloud.CommonResponse
	MongoNodeSet []MongoNode `json:"mongo_node_set"`
	TotalCount   int         `json:"total_count"`
}

// DescribeMongoNodes 获取 Mongo 节点相关的信息。
// 获取指定 Mongo 的所有节点信息。
func (c *MONGO) DescribeMongoNodes(params DescribeMongoNodesRequest) (DescribeMongoNodesResponse, error) {
	var result DescribeMongoNodesResponse
	err := c.Get("DescribeMongoNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeMongoParametersRequest struct {
	Mongo  qingcloud.String
	Offset qingcloud.Integer
	Limit  qingcloud.Integer
}
type DescribeMongoParametersResponse struct {
	qingcloud.CommonResponse
	TotalCount   int         `json:"total_count"`
	ParameterSet []Parameter `json:"parameter_set"`
}

// DescribeMongoParameters 获取指定 Mongo 的配置信息。
func (c *MONGO) DescribeMongoParameters(params DescribeMongoParametersRequest) (DescribeMongoParametersResponse, error) {
	var result DescribeMongoParametersResponse
	err := c.Get("DescribeMongoParameters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ResizeMongosRequest struct {
	MongosN     qingcloud.NumberedString
	MongoType   qingcloud.String
	StorageSize qingcloud.Integer
}
type ResizeMongosResponse struct {
	qingcloud.CommonResponse
	Mongos []string `json:"mongos"`
}

// ResizeMongos 扩容指定的 Mongo 集群。
func (c *MONGO) ResizeMongos(params ResizeMongosRequest) (ResizeMongosResponse, error) {
	var result ResizeMongosResponse
	err := c.Get("ResizeMongos", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateMongoRequest struct {
	Vxnet          qingcloud.String
	MongoVersion   qingcloud.String
	MongoType      qingcloud.Integer
	StorageSize    qingcloud.Integer
	MongoName      qingcloud.String
	Description    qingcloud.String
	AutoBackupTime qingcloud.Integer
	PrivateIps     qingcloud.Dict
}
type CreateMongoResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// CreateMongo 创建 Mongo 集群。
func (c *MONGO) CreateMongo(params CreateMongoRequest) (CreateMongoResponse, error) {
	var result CreateMongoResponse
	err := c.Get("CreateMongo", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StopMongosRequest struct {
	MongosN qingcloud.NumberedString
}
type StopMongosResponse struct {
	qingcloud.CommonResponse
	Mongos []string `json:"mongos"`
}

// StopMongos 关闭指定的 Mongo 集群。
func (c *MONGO) StopMongos(params StopMongosRequest) (StopMongosResponse, error) {
	var result StopMongosResponse
	err := c.Get("StopMongos", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StartMongosRequest struct {
	MongosN qingcloud.NumberedString
}
type StartMongosResponse struct {
	qingcloud.CommonResponse
	Mongos []string `json:"mongos"`
}

// StartMongos 启动指定的 Mongo 集群。
func (c *MONGO) StartMongos(params StartMongosRequest) (StartMongosResponse, error) {
	var result StartMongosResponse
	err := c.Get("StartMongos", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeMongosRequest struct {
	MongosN   qingcloud.NumberedString
	StatusN   qingcloud.NumberedString
	MongoName qingcloud.String

	TagsN   qingcloud.NumberedString
	Verbose qingcloud.Integer
	Offset  qingcloud.Integer
	Limit   qingcloud.Integer
}
type DescribeMongosResponse struct {
	qingcloud.CommonResponse
	TotalCount int     `json:"total_count"`
	MongoSet   []Mongo `json:"mongo_set"`
}

// DescribeMongos 获取 Mongo 集群相关的信息。
// 可根据 Mongo ID，状态，Mongo 名称作过滤条件，来获取 Mongo 集群列表。 如果不指定任何过滤条件，默认返回你所拥有的所有 Mongo 集群。 如果指定不存在的 Mongo ID，或非法状态值，则会返回错误信息。
func (c *MONGO) DescribeMongos(params DescribeMongosRequest) (DescribeMongosResponse, error) {
	var result DescribeMongosResponse
	err := c.Get("DescribeMongos", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteMongosRequest struct {
	MongosN qingcloud.NumberedString
}
type DeleteMongosResponse struct {
	qingcloud.CommonResponse
	Mongos []string `json:"mongos"`
}

// DeleteMongos 删除指定 Mongo 集群。
func (c *MONGO) DeleteMongos(params DeleteMongosRequest) (DeleteMongosResponse, error) {
	var result DeleteMongosResponse
	err := c.Get("DeleteMongos", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateMongoFromSnapshotRequest struct {
	Vxnet          qingcloud.String
	MongoType      qingcloud.Integer
	MongoName      qingcloud.String
	Description    qingcloud.String
	AutoBackupTime qingcloud.Integer
}
type CreateMongoFromSnapshotResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// CreateMongoFromSnapshot 从指定备份创建一个新的 Mongo 集群。
func (c *MONGO) CreateMongoFromSnapshot(params CreateMongoFromSnapshotRequest) (CreateMongoFromSnapshotResponse, error) {
	var result CreateMongoFromSnapshotResponse
	err := c.Get("CreateMongoFromSnapshot", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ChangeMongoVxnetRequest struct {
	Mongo      qingcloud.String
	Vxnet      qingcloud.String
	PrivateIps qingcloud.Dict
}
type ChangeMongoVxnetResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// ChangeMongoVxnet 变更 Mongo 集群的私有网络，即离开原有私有网络并加入新的私有网络。
func (c *MONGO) ChangeMongoVxnet(params ChangeMongoVxnetRequest) (ChangeMongoVxnetResponse, error) {
	var result ChangeMongoVxnetResponse
	err := c.Get("ChangeMongoVxnet", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddMongoInstancesRequest struct {
	Mongo      qingcloud.String
	NodeCount  qingcloud.Integer
	PrivateIps qingcloud.Dict
}
type AddMongoInstancesResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// AddMongoInstances 添加新节点到指定 Mongo 集群。
func (c *MONGO) AddMongoInstances(params AddMongoInstancesRequest) (AddMongoInstancesResponse, error) {
	var result AddMongoInstancesResponse
	err := c.Get("AddMongoInstances", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type RemoveMongoInstancesRequest struct {
	Mongo           qingcloud.String
	MongoInstancesN qingcloud.NumberedString
}
type RemoveMongoInstancesResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// RemoveMongoInstances 添加新节点到指定 Mongo 集群。
func (c *MONGO) RemoveMongoInstances(params RemoveMongoInstancesRequest) (RemoveMongoInstancesResponse, error) {
	var result RemoveMongoInstancesResponse
	err := c.Get("RemoveMongoInstances", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyMongoAttributesRequest struct {
	Mongo          qingcloud.String
	MongoName      qingcloud.String
	Description    qingcloud.String
	AutoBackupTime qingcloud.Integer
}
type ModifyMongoAttributesResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// ModifyMongoAttributes 修改 Mongo 集群相关属性值。
func (c *MONGO) ModifyMongoAttributes(params ModifyMongoAttributesRequest) (ModifyMongoAttributesResponse, error) {
	var result ModifyMongoAttributesResponse
	err := c.Get("ModifyMongoAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyMongoInstancesRequest struct {
	Mongo      qingcloud.String
	PrivateIps qingcloud.Dict
}
type ModifyMongoInstancesResponse struct {
	qingcloud.CommonResponse
	Mongo string `json:"mongo"`
}

// ModifyMongoInstances 修改 Mongo 集群相关属性值。
func (c *MONGO) ModifyMongoInstances(params ModifyMongoInstancesRequest) (ModifyMongoInstancesResponse, error) {
	var result ModifyMongoInstancesResponse
	err := c.Get("ModifyMongoInstances", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

// TODO: 如下还没有实现
type GetMongoMonitorRequest struct {
	Resource  qingcloud.String
	MetersN   qingcloud.NumberedString
	Step      qingcloud.String
	StartTime qingcloud.String
	EndTime   qingcloud.String
}

type GetMongoMonitorResponse struct {
	qingcloud.CommonResponse
}

// GetMongoMonitor 获取指定 Mongo 节点的监控信息。
func (c *MONGO) GetMongoMonitor(params GetMongoMonitorRequest) (GetMongoMonitorResponse, error) {
	var result GetMongoMonitorResponse
	err := c.Get("GetMongoMonitor", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
