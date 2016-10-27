package volume

import (
	"github.com/magicshui/qingcloud-go"
)

type VOLUME struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *VOLUME {
	return &VOLUME{
		Client: clt,
	}
}

type DescribeVolumesRequest struct {
	VolumesN   qingcloud.NumberedString
	VolumeType qingcloud.Integer
	StatusN    qingcloud.NumberedString
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeVolumesResponse struct {
	VolumeSet  []Volume `json:"volume_set"`
	TotalCount int      `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeVolumes 获取一个或多个硬盘
// 可根据硬盘ID，状态，硬盘名称作过滤条件，来获取硬盘列表。 如果不指定任何过滤条件，默认返回你所拥有的所有硬盘。 如果指定不存在的硬盘ID，或非法状态值，则会返回错误信息。
func (c *VOLUME) DescribeVolumes(params DescribeVolumesRequest) (DescribeVolumesResponse, error) {
	var result DescribeVolumesResponse
	// 硬盘状态: pending, available, in-use, suspended, deleted, ceased
	params.StatusN.Enum("pending", "available", "in-use", "suspended", "deleted", "ceased")
	err := c.Get("DescribeVolumes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateVolumesRequest struct {
	Size       qingcloud.Integer
	VolumeName qingcloud.String
	VolumeType qingcloud.Integer
	Count      qingcloud.Integer
}
type CreateVolumesResponse struct {
	Volumes []string `json:"volumes"`
	qingcloud.CommonResponse
}

// CreateVolumes 创建一块或多块硬盘，每块硬盘都可加载到任意一台主机中。
func (c *VOLUME) CreateVolumes(params CreateVolumesRequest) (CreateVolumesResponse, error) {
	var result CreateVolumesResponse
	err := c.Get("CreateVolumes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteVolumesRequest struct {
	VolumesN qingcloud.NumberedString
}
type DeleteVolumesReponse qingcloud.CommonResponse

// DeleteVolumes 删除一块或多块硬盘。硬盘须在可用（ available ）状态下才能被删除， 已加载到主机的硬盘需先卸载后才能删除。
func (c *VOLUME) DeleteVolumes(params DeleteVolumesRequest) (DeleteVolumesReponse, error) {
	var result DeleteVolumesReponse
	err := c.Get("DeleteVolumes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AttachVolumesRequest struct {
	VolumesN qingcloud.NumberedString
	Instance qingcloud.String
}
type AttachVolumesResponse qingcloud.CommonResponse

// AttachVolumes  将一块或多块“可用”（ available ）状态的硬盘加载到某台”运行”（ running ） 或”关机”（ stopped ）状态的主机。
func (c *VOLUME) AttachVolumes(params AttachVolumesRequest) (AttachVolumesResponse, error) {
	var result AttachVolumesResponse
	err := c.Get("AttachVolumes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DetachVolumesRequest struct {
	VolumesN qingcloud.NumberedString
	Instance qingcloud.String
}
type DetachVolumesResponse qingcloud.CommonResponse

// DetachVolumes 将一块或多块“使用中”（ in-use ）状态的硬盘从某台主机中卸载。
// 卸载前要保证已先从操作系统中 unmount 了硬盘，不然会返回错误信息。
// 不管卸载是否成功，都不会对硬盘内的数据产生影响。
func (c *VOLUME) DetachVolumes(params DetachVolumesRequest) (DetachVolumesResponse, error) {
	var result DetachVolumesResponse
	err := c.Get("DetachVolumes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ResizeVolumesRequest struct {
	VolumesN qingcloud.NumberedString
	Size     qingcloud.Integer
}
type ResizeVolumesResponse qingcloud.CommonResponse

// ResizeVolumes 给一块或多块“可用”（ available ）状态的硬盘扩大容量。
// 只允许扩大容量，不支持减小。
func (c *VOLUME) ResizeVolumes(params ResizeVolumesRequest) (ResizeVolumesResponse, error) {
	var result ResizeVolumesResponse
	err := c.Get("ResizeVolumes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyVolumeAttributesRequest struct {
	Volume      qingcloud.String
	VolumeName  qingcloud.String
	Description qingcloud.String
}
type ModifyVolumeAttributesResponse qingcloud.CommonResponse

// ModifyVolumeAttributes 修改一块硬盘的名称和描述。
// 修改时不受硬盘状态限制。
// 一次只能修改一块硬盘。
func (c *VOLUME) ModifyVolumeAttributes(params ModifyVolumeAttributesRequest) (ModifyVolumeAttributesResponse, error) {
	var result ModifyVolumeAttributesResponse
	err := c.Get("ModifyVolumeAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
