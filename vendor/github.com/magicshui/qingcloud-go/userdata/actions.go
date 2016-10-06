package userdata

import (
	"encoding/base64"
	"github.com/magicshui/qingcloud-go"
	"io/ioutil"
)

type USERDATA struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *USERDATA {
	return &USERDATA{
		Client: clt,
	}
}

type UploadUserDataAttachmentRequest struct {
	AttachmentContent qingcloud.String
	AttachmentName    qingcloud.String
}

func (c *UploadUserDataAttachmentRequest) Read(p string) {
	rawData, err := ioutil.ReadFile(p)
	if err != nil {
		return
	}
	str := base64.StdEncoding.EncodeToString(rawData)
	c.AttachmentContent.Set(str)
}

type UploadUserDataAttachmentResponse struct {
	qingcloud.CommonResponse
	AttachmentId string `json:"attachment_id"`
}

func (c *USERDATA) UploadUserDataAttachment(params UploadUserDataAttachmentRequest) (UploadUserDataAttachmentResponse, error) {
	var result UploadUserDataAttachmentResponse
	err := c.Get("UploadUserDataAttachment", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
