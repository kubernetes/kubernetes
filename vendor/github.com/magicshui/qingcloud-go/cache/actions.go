package cache

import (
	"github.com/magicshui/qingcloud-go"
)

type CACHE struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *CACHE {
	return &CACHE{
		Client: clt,
	}
}

type DescribeCachesRequest struct {
	CachesN    qingcloud.NumberedString
	StatusN    qingcloud.NumberedString
	CacheTypeN qingcloud.NumberedString
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeCachesResponse struct {
	qingcloud.CommonResponse
	CacheSet   []Cache `json:"cache_set"`
	TotalCount int     `json:"total_count"`
}

// DescribeCaches 获取一个或多个缓存服务。
// 可根据缓存服务ID，状态，缓存服务名称作过滤条件，来获取缓存服务列表。 如果不指定任何过滤条件，默认返回你的所有缓存服务。
func (c *CACHE) DescribeCaches(params DescribeCachesRequest) (DescribeCachesResponse, error) {
	var result DescribeCachesResponse
	err := c.Get("DescribeCaches", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateCacheRequest struct {
	Vxnet                 qingcloud.String
	CacheSize             qingcloud.Integer
	CacheType             qingcloud.String
	NodeCount             qingcloud.Integer
	CacheName             qingcloud.String
	CacheParameterGroup   qingcloud.String
	PrivateIpsNCacheRole  qingcloud.NumberedString
	AutoBackupTime        qingcloud.Integer
	PrivateIpsNPrivateIps qingcloud.String
	CacheClass            qingcloud.String
}
type CreateCacheResponse struct {
	qingcloud.CommonResponse
	CacheId    string   `json:"cache_id"`
	CacheNodes []string `json:"cache_nodes"`
}

// CreateCache
// 创建一个缓存服务。目前缓存服务只能运行于私有网络中，在创建缓存之前，您需要有一个已连接到路由器的私有网络。
func (c *CACHE) CreateCache(params CreateCacheRequest) (CreateCacheResponse, error) {
	var result CreateCacheResponse
	err := c.Get("CreateCache", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StopCachesRequest struct {
	CachesN qingcloud.NumberedString
}
type StopCachesResponse qingcloud.CommonResponse

// StopCaches
// 关闭一台或多台缓存服务。该操作将关闭缓存服务的所有缓存节点。
func (c *CACHE) StopCaches(params StopCachesRequest) (StopCachesResponse, error) {
	var result StopCachesResponse
	err := c.Get("StopCaches", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StartCachesRequest struct {
	CachesN qingcloud.NumberedString
}
type StartCachesResponse qingcloud.CommonResponse

// StartCaches
// 启动一台或多台缓存服务。
func (c *CACHE) StartCaches(params StartCachesRequest) (StartCachesResponse, error) {
	var result StartCachesResponse
	err := c.Get("StartCaches", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type RestartCachesRequest struct {
	CachesN qingcloud.NumberedString
}
type RestartCachesResponse qingcloud.CommonResponse

// RestartCaches
// 重启一台或多台缓存服务。该操作将重启缓存服务的所有缓存节点。
func (c *CACHE) RestartCaches(params RestartCachesRequest) (RestartCachesResponse, error) {
	var result RestartCachesResponse
	err := c.Get("RestartCaches", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteCachesRequest struct {
	CachesN qingcloud.NumberedString
}
type DeleteCachesResponse qingcloud.CommonResponse

// DeleteCaches
// 删除一台或多台缓存服务。
func (c *CACHE) DeleteCaches(params DeleteCachesRequest) (DeleteCachesResponse, error) {
	var result DeleteCachesResponse
	err := c.Get("DeleteCaches", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ResizeCachesRequest struct {
	CachesN   qingcloud.NumberedString
	CacheSize qingcloud.Integer
}
type ResizeCachesResponse qingcloud.CommonResponse

// ResizeCaches
// 修改缓存服务节点内存大小。缓存服务状态必须是关闭的 stopped ，不然会返回错误。
func (c *CACHE) ResizeCaches(params ResizeCachesRequest) (ResizeCachesResponse, error) {
	var result ResizeCachesResponse
	err := c.Get("ResizeCaches", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type UpdateCacheRequest struct {
	Cache                  qingcloud.String
	PrivateIpsNCacheNodeId qingcloud.NumberedString
	PrivateIpsNPrivateIp   qingcloud.NumberedString
}
type UpdateCacheResponse qingcloud.CommonResponse

// UpdateCache
// 更新缓存服务的配置，例如更新缓存节点 IP 地址。
func (c *CACHE) UpdateCache(params UpdateCacheRequest) (UpdateCacheResponse, error) {
	var result UpdateCacheResponse
	err := c.Get("UpdateCache", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ChangeCacheVxnetRequest struct {
	Cache qingcloud.String
	Vxnet qingcloud.String

	PrivateIpsNCacheNodeId qingcloud.NumberedString
	PrivateIpsNPrivateIp   qingcloud.NumberedString
}
type ChangeCacheVxnetResponse qingcloud.CommonResponse

// ChangeCacheVxnet
// 变更缓存服务的私有网络，即离开原有私有网络并加入新的私有网络。
func (c *CACHE) ChangeCacheVxnet(params ChangeCacheVxnetRequest) (ChangeCacheVxnetResponse, error) {
	var result ChangeCacheVxnetResponse
	err := c.Get("ChangeCacheVxnet", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyCacheAttributesRequest struct {
	Cache          qingcloud.String
	CacheName      qingcloud.String
	Description    qingcloud.String
	AutoBackupTime qingcloud.Integer
}
type ModifyCacheAttributesResponse qingcloud.CommonResponse

// ModifyCacheAttributes
// 修改一台缓存服务的名称和描述。
func (c *CACHE) ModifyCacheAttributes(params ModifyCacheAttributesRequest) (ModifyCacheAttributesResponse, error) {
	var result ModifyCacheAttributesResponse
	err := c.Get("ModifyCacheAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeCacheNodesRequest struct {
	Cache      qingcloud.String
	CacheTypeN qingcloud.NumberedString
	StatusN    qingcloud.NumberedString
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeCacheNodesResponse struct {
	qingcloud.CommonResponse
	CacheNodeSet []CacheNode `json:"cache_node_set"`
	TotalCount   int         `json:"total_count"`
}

// DescribeCacheNodes 获取一个或多个缓存服务节点信息。
// 可根据缓存服务ID，缓存服务节点ID，状态，缓存服务节点名称作过滤条件，来获取缓存服务节点列表。 如果不指定任何过滤条件，默认返回你的所有缓存服务节点。
func (c *CACHE) DescribeCacheNodes(params DescribeCacheNodesRequest) (DescribeCacheNodesResponse, error) {
	var result DescribeCacheNodesResponse
	err := c.Get("DescribeCacheNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddCacheNodesRequest struct {
	Cache     qingcloud.String
	NodeCount qingcloud.Integer

	PrivateIpsNCacheRole  qingcloud.NumberedString
	PrivateIpsNPrivateIps qingcloud.NumberedString
}
type AddCacheNodesResponse struct {
	qingcloud.CommonResponse
	CacheNodes []string `json:"cache_nodes"`
}

// AddCacheNodes
// 给缓存服务添加一个或多个缓存节点。
func (c *CACHE) AddCacheNodes(params AddCacheNodesRequest) (AddCacheNodesResponse, error) {
	var result AddCacheNodesResponse
	err := c.Get("AddCacheNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteCacheNodesRequest struct {
	Cache       qingcloud.String
	CacheNodesN qingcloud.NumberedString
}
type DeleteCacheNodesResponse qingcloud.CommonResponse

// DeleteCacheNodes 删除缓存服务节点。
func (c *CACHE) DeleteCacheNodes(params DeleteCacheNodesRequest) (DeleteCacheNodesResponse, error) {
	var result DeleteCacheNodesResponse
	err := c.Get("DeleteCacheNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type RestartCacheNodesRequest struct {
	Cache       qingcloud.String
	CacheNodesN qingcloud.NumberedString
}
type RestartCacheNodesResponse qingcloud.CommonResponse

// RestartCacheNodes
// 重启一台或多台缓存服务节点。
func (c *CACHE) RestartCacheNodes(params RestartCacheNodesRequest) (RestartCacheNodesResponse, error) {
	var result RestartCacheNodesResponse
	err := c.Get("RestartCacheNodes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyCacheNodeAttributesRequest struct {
	CacheNode     qingcloud.String
	CacheNodeName qingcloud.String
}
type ModifyCacheNodeAttributesResponse qingcloud.CommonResponse

// ModifyCacheNodeAttributes
// 修改一台缓存服务节点的名称。
func (c *CACHE) ModifyCacheNodeAttributes(params ModifyCacheNodeAttributesRequest) (ModifyCacheNodeAttributesResponse, error) {
	var result ModifyCacheNodeAttributesResponse
	err := c.Get("ModifyCacheNodeAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateCacheFromSnapshotRequest struct {
	Snapshot              qingcloud.String
	Vxnet                 qingcloud.String
	NodeCount             qingcloud.Integer
	CacheName             qingcloud.String
	CacheParameterGroup   qingcloud.String
	AutoBackupTime        qingcloud.Integer
	PrivateIpsNCacheRole  qingcloud.NumberedString
	PrivateIpsNPrivateIps qingcloud.NumberedString
	CacheClass            qingcloud.Integer
}
type CreateCacheFromSnapshotResponse qingcloud.CommonResponse

// CreateCacheFromSnapshot
// 由备份节点创建一个新的缓存服务。
func (c *CACHE) CreateCacheFromSnapshot(params CreateCacheFromSnapshotRequest) (CreateCacheFromSnapshotResponse, error) {
	var result CreateCacheFromSnapshotResponse
	err := c.Get("CreateCacheFromSnapshot", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeCacheParameterGroupsRequest struct {
	CacheParameterGroupsN qingcloud.NumberedString
	CacheType             qingcloud.String

	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeCacheParameterGroupsResponse struct {
	qingcloud.CommonResponse
	CacheParameterGroupSet []CacheParameterGroup `json:"cache_parameter_group_set"`
}

// DescribeCacheParameterGroups 获取一个或多个缓存配置组信息。
// 可根据缓存配置组ID，缓存配置组名称作过滤条件，来获取缓存配置组列表。 如果不指定任何过滤条件，默认返回你的所有缓存配置组。
func (c *CACHE) DescribeCacheParameterGroups(params DescribeCacheParameterGroupsRequest) (DescribeCacheParameterGroupsResponse, error) {
	var result DescribeCacheParameterGroupsResponse
	err := c.Get("DescribeCacheParameterGroups", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateCacheParameterGroupRequest struct {
	CacheType               qingcloud.String
	CacheParameterGroupName qingcloud.String
}
type CreateCacheParameterGroupResponse struct {
	qingcloud.CommonResponse
	CacheParameterGroupId string `json:"cache_parameter_group_id"`
}

// CreateCacheParameterGroup
// 创建一个缓存服务配置组。
func (c *CACHE) CreateCacheParameterGroup(params CreateCacheParameterGroupRequest) (CreateCacheParameterGroupResponse, error) {
	var result CreateCacheParameterGroupResponse
	err := c.Get("CreateCacheParameterGroup", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ApplyCacheParameterGroupRequest struct {
	CacheParameterGroup qingcloud.String
	CachesN             qingcloud.NumberedString
}
type ApplyCacheParameterGroupResponse qingcloud.CommonResponse

// ApplyCacheParameterGroup
// 将缓存配置组应用于缓存服务。
func (c *CACHE) ApplyCacheParameterGroup(params ApplyCacheParameterGroupRequest) (ApplyCacheParameterGroupResponse, error) {
	var result ApplyCacheParameterGroupResponse
	err := c.Get("ApplyCacheParameterGroup", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteCacheParameterGroupsRequest struct {
	CacheParameterGroupsN qingcloud.NumberedString
}
type DeleteCacheParameterGroupsResponse struct {
	qingcloud.CommonResponse
	CacheParameterGroups []string `json:"cache_parameter_groups"`
}

// DeleteCacheParameterGroups 删除缓存服务配置组。
func (c *CACHE) DeleteCacheParameterGroups(params DeleteCacheParameterGroupsRequest) (DeleteCacheParameterGroupsResponse, error) {
	var result DeleteCacheParameterGroupsResponse
	err := c.Get("DeleteCacheParameterGroups", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyCacheParameterGroupAttributesRequest struct {
	CacheParameterGroup     qingcloud.String
	CacheParameterGroupName qingcloud.NumberedString
	Description             qingcloud.String
}
type ModifyCacheParameterGroupAttributesResponse qingcloud.CommonResponse

// ModifyCacheParameterGroupAttributes
// 修改缓存服务配置组的名称或描述。
func (c *CACHE) ModifyCacheParameterGroupAttributes(params ModifyCacheParameterGroupAttributesRequest) (ModifyCacheParameterGroupAttributesResponse, error) {
	var result ModifyCacheParameterGroupAttributesResponse
	err := c.Get("ModifyCacheParameterGroupAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeCacheParametersRequest struct {
	CacheParameterGroup qingcloud.String
}
type DescribeCacheParametersResponse struct {
	qingcloud.CommonResponse
	TotalCount        int                   `json:"total_count"`
	CacheParameterSet []CacheParameterGroup `json:"cache_parameter_set"`
}

// DescribeCacheParameters 返回指定缓存配置组的所有配置项信息。
func (c *CACHE) DescribeCacheParameters(params DescribeCacheParametersRequest) (DescribeCacheParametersResponse, error) {
	var result DescribeCacheParametersResponse
	err := c.Get("DescribeCacheParameters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type UpdateCacheParametersRequest struct {
	CacheParameterGroup            qingcloud.String
	ParametersNCacheParameterName  qingcloud.NumberedString
	ParametersNCacheParameterValue qingcloud.NumberedString
}
type UpdateCacheParametersResponse struct {
	qingcloud.CommonResponse
	CacheParameterGroupId string `json:"cache_parameter_group_id"`
}

// UpdateCacheParameters 更改缓存配置项。
func (c *CACHE) UpdateCacheParameters(params UpdateCacheParametersRequest) (UpdateCacheParametersResponse, error) {
	var result UpdateCacheParametersResponse
	err := c.Get("UpdateCacheParameters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ResetCacheParametersRequest struct {
	CacheParameterGroup  qingcloud.String
	CacheParameterNamesN qingcloud.NumberedString
}
type ResetCacheParametersResponse struct {
	qingcloud.CommonResponse
	CacheParameterGroupId string `json:"cache_parameter_group_id"`
}

// ResetCacheParameters 重置缓存配置项为系统默认值。
func (c *CACHE) ResetCacheParameters(params ResetCacheParametersRequest) (ResetCacheParametersResponse, error) {
	var result ResetCacheParametersResponse
	err := c.Get("ResetCacheParameters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
