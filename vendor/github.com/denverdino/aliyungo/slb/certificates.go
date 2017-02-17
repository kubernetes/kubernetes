package slb

import "github.com/denverdino/aliyungo/common"

type UploadServerCertificateArgs struct {
	RegionId              common.Region
	ServerCertificate     string
	ServerCertificateName string
	PrivateKey            string
}

type UploadServerCertificateResponse struct {
	common.Response
	ServerCertificateId   string
	ServerCertificateName string
	Fingerprint           string
}

// UploadServerCertificate Upload server certificate
//
// You can read doc at http://docs.aliyun.com/#pub/slb/api-reference/api-servercertificate&UploadServerCertificate
func (client *Client) UploadServerCertificate(args *UploadServerCertificateArgs) (response *UploadServerCertificateResponse, err error) {
	response = &UploadServerCertificateResponse{}
	err = client.Invoke("UploadServerCertificate", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

type DeleteServerCertificateArgs struct {
	RegionId            common.Region
	ServerCertificateId string
}

type DeleteServerCertificateResponse struct {
	common.Response
}

// DeleteServerCertificate Delete server certificate
//
// You can read doc at http://docs.aliyun.com/#pub/slb/api-reference/api-servercertificate&DeleteServerCertificate
func (client *Client) DeleteServerCertificate(regionId common.Region, serverCertificateId string) (err error) {
	args := &DeleteServerCertificateArgs{
		RegionId:            regionId,
		ServerCertificateId: serverCertificateId,
	}
	response := &DeleteServerCertificateResponse{}
	return client.Invoke("DeleteServerCertificate", args, response)
}

type SetServerCertificateNameArgs struct {
	RegionId              common.Region
	ServerCertificateId   string
	ServerCertificateName string
}

type SetServerCertificateNameResponse struct {
	common.Response
}

// SetServerCertificateName Set name of server certificate
//
// You can read doc at http://docs.aliyun.com/#pub/slb/api-reference/api-servercertificate&SetServerCertificateName
func (client *Client) SetServerCertificateName(regionId common.Region, serverCertificateId string, name string) (err error) {
	args := &SetServerCertificateNameArgs{
		RegionId:              regionId,
		ServerCertificateId:   serverCertificateId,
		ServerCertificateName: name,
	}
	response := &SetServerCertificateNameResponse{}
	return client.Invoke("SetServerCertificateName", args, response)
}

type DescribeServerCertificatesArgs struct {
	RegionId            common.Region
	ServerCertificateId string
}

type ServerCertificateType struct {
	RegionId              common.Region
	ServerCertificateId   string
	ServerCertificateName string
	Fingerprint           string
}

type DescribeServerCertificatesResponse struct {
	common.Response
	ServerCertificates struct {
		ServerCertificate []ServerCertificateType
	}
}

// DescribeServerCertificates Describe server certificates
//
// You can read doc at http://docs.aliyun.com/#pub/slb/api-reference/api-servercertificate&DescribeServerCertificates
func (client *Client) DescribeServerCertificatesArgs(regionId common.Region, serverCertificateId string) (serverCertificates []ServerCertificateType, err error) {
	args := &DescribeServerCertificatesArgs{
		RegionId:            regionId,
		ServerCertificateId: serverCertificateId,
	}
	response := &DescribeServerCertificatesResponse{}
	err = client.Invoke("DescribeServerCertificates", args, response)
	if err != nil {
		return nil, err
	}
	return response.ServerCertificates.ServerCertificate, err
}
