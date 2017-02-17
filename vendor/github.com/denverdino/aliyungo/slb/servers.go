package slb

import (
	"encoding/json"

	"github.com/denverdino/aliyungo/common"
)

type AddBackendServersArgs struct {
	LoadBalancerId string
	BackendServers string
}

type SetBackendServersArgs AddBackendServersArgs

type AddBackendServersResponse struct {
	common.Response
	LoadBalancerId string
	BackendServers struct {
		BackendServer []BackendServerType
	}
}

type SetBackendServersResponse AddBackendServersResponse


// SetBackendServers set weight of backend servers

func (client *Client) SetBackendServers(loadBalancerId string, backendServers []BackendServerType) (result []BackendServerType, err error) {
	bytes, _ := json.Marshal(backendServers)

	args := &SetBackendServersArgs{
		LoadBalancerId: loadBalancerId,
		BackendServers: string(bytes),
	}
	response := &SetBackendServersResponse{}

	err = client.Invoke("SetBackendServers", args, response)
	if err != nil {
		return nil, err
	}
	return response.BackendServers.BackendServer, err
}


// AddBackendServers Add backend servers
//
// You can read doc at http://docs.aliyun.com/#/pub/slb/api-reference/api-related-backendserver&AddBackendServers
func (client *Client) AddBackendServers(loadBalancerId string, backendServers []BackendServerType) (result []BackendServerType, err error) {

	bytes, _ := json.Marshal(backendServers)

	args := &AddBackendServersArgs{
		LoadBalancerId: loadBalancerId,
		BackendServers: string(bytes),
	}
	response := &AddBackendServersResponse{}

	err = client.Invoke("AddBackendServers", args, response)
	if err != nil {
		return nil, err
	}
	return response.BackendServers.BackendServer, err
}

type RemoveBackendServersArgs struct {
	LoadBalancerId string
	BackendServers []string
}

type RemoveBackendServersResponse struct {
	common.Response
	LoadBalancerId string
	BackendServers struct {
		BackendServer []BackendServerType
	}
}

// RemoveBackendServers Remove backend servers
//
// You can read doc at http://docs.aliyun.com/#/pub/slb/api-reference/api-related-backendserver&RemoveBackendServers
func (client *Client) RemoveBackendServers(loadBalancerId string, backendServers []string) (result []BackendServerType, err error) {
	args := &RemoveBackendServersArgs{
		LoadBalancerId: loadBalancerId,
		BackendServers: backendServers,
	}
	response := &RemoveBackendServersResponse{}
	err = client.Invoke("RemoveBackendServers", args, response)
	if err != nil {
		return nil, err
	}
	return response.BackendServers.BackendServer, err
}

type HealthStatusType struct {
	ServerId           string
	ServerHealthStatus string
}

type DescribeHealthStatusArgs struct {
	LoadBalancerId string
	ListenerPort   int
}

type DescribeHealthStatusResponse struct {
	common.Response
	BackendServers struct {
		BackendServer []HealthStatusType
	}
}

// DescribeHealthStatus Describe health status
//
// You can read doc at http://docs.aliyun.com/#/pub/slb/api-reference/api-related-backendserver&DescribeHealthStatus
func (client *Client) DescribeHealthStatus(args *DescribeHealthStatusArgs) (response *DescribeHealthStatusResponse, err error) {
	response = &DescribeHealthStatusResponse{}
	err = client.Invoke("DescribeHealthStatus", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}
