package cluster

import (
	"errors"
	"strconv"
	"time"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/api/client"
	"github.com/libopenstorage/openstorage/cluster"
)

const (
	clusterPath = "/cluster"
	loggingurl = "/loggingurl"
	managementurl = "/managementurl"
	fluentdhost = "/fluentdconfig"
	tunnelconfigurl = "/tunnelconfig"
)

type clusterClient struct {
	c *client.Client
}

func newClusterClient(c *client.Client) cluster.Cluster {
	return &clusterClient{c: c}
}

// String description of this driver.
func (c *clusterClient) Name() string {
	return "ClusterManager"
}

func (c *clusterClient) Enumerate() (api.Cluster, error) {
	cluster := api.Cluster{}

	if err := c.c.Get().Resource(clusterPath + "/enumerate").Do().Unmarshal(&cluster); err != nil {
		return cluster, err
	}
	return cluster, nil
}

func (c *clusterClient) SetSize(size int) error {
	resp := api.ClusterResponse{}

	request := c.c.Get().Resource(clusterPath + "/setsize")
	request.QueryOption("size", strconv.FormatInt(int64(size), 16))
	if err := request.Do().Unmarshal(&resp); err != nil {
		return err
	}

	if resp.Error != "" {
		return errors.New(resp.Error)
	}

	return nil
}

func (c *clusterClient) Inspect(nodeID string) (api.Node, error) {
	var resp api.Node
	request := c.c.Get().Resource(clusterPath + "/inspect/" + nodeID)
	if err := request.Do().Unmarshal(&resp); err != nil {
		return api.Node{}, err
	}
	return resp, nil
}

func (c *clusterClient) AddEventListener(cluster.ClusterListener) error {
	return nil
}

func (c *clusterClient) UpdateData(nodeData map[string]interface{}) error {
	return nil
}

func (c *clusterClient) UpdateLabels(nodeLabels map[string]string) error {
	return nil
}

func (c *clusterClient) GetData() (map[string]*api.Node, error) {
	return nil, nil
}

func (c *clusterClient) NodeStatus() (api.Status, error) {
	var resp api.Status
	request := c.c.Get().Resource(clusterPath + "/nodestatus")
	if err := request.Do().Unmarshal(&resp); err != nil {
		return api.Status_STATUS_NONE, err
	}
	return resp, nil
}

func (c *clusterClient) PeerStatus(listenerName string) (map[string]api.Status, error) {
	var resp map[string]api.Status
	request := c.c.Get().Resource(clusterPath + "/peerstatus")
	request.QueryOption("name", listenerName)
	if err := request.Do().Unmarshal(&resp); err != nil {
		return nil, err
	}
	return resp, nil
}

func (c *clusterClient) Remove(nodes []api.Node, forceRemove bool) error {
	resp := api.ClusterResponse{}

	request := c.c.Delete().Resource(clusterPath + "/")

	for _, n := range nodes {
		request.QueryOption("id", n.Id)
	}
	request.QueryOption("forceRemove", strconv.FormatBool(forceRemove))

	if err := request.Do().Unmarshal(&resp); err != nil {
		return err
	}

	if resp.Error != "" {
		return errors.New(resp.Error)
	}

	return nil
}

func (c *clusterClient) NodeRemoveDone(nodeID string, result error) {
}

func (c *clusterClient) Shutdown() error {
	return nil
}

func (c *clusterClient) Start(int, bool) error {
	return nil
}

func (c *clusterClient) DisableUpdates() error {
	c.c.Put().Resource(clusterPath + "/disablegossip").Do()
	return nil
}

func (c *clusterClient) EnableUpdates() error {
	c.c.Put().Resource(clusterPath + "/enablegossip").Do()
	return nil
}

func (c *clusterClient) SetLoggingURL(loggingURL string) error {

	resp := api.ClusterResponse{}

	request := c.c.Put().Resource(clusterPath + loggingurl)
	request.QueryOption("url", loggingURL)
	if err := request.Do().Unmarshal(&resp); err != nil {
		return err
	}

	if resp.Error != "" {
		return errors.New(resp.Error)
	}

	return nil

}

func (c *clusterClient) SetManagementURL(managementURL string) error {

	resp := api.ClusterResponse{}

	request := c.c.Put().Resource(clusterPath + managementurl)
	request.QueryOption("url", managementURL)
	if err := request.Do().Unmarshal(&resp); err != nil {
		return err
	}

	if resp.Error != "" {
		return errors.New(resp.Error)
	}

	return nil

}

func (c *clusterClient) SetFluentDConfig(fluentDConfig api.FluentDConfig) error {
	resp := api.ClusterResponse{}
	request := c.c.Put().Resource(clusterPath + fluentdhost)
	request.Body(&fluentDConfig)

	if err := request.Do().Unmarshal(&resp); err != nil {
		return err
	}

	if resp.Error != "" {
		return errors.New(resp.Error)
	}

	return nil
}

func (c *clusterClient) GetFluentDConfig() api.FluentDConfig {
	tc := api.FluentDConfig{}

	if err := c.c.Get().Resource(clusterPath + fluentdhost).Do().Unmarshal(&tc); err != nil {
		return api.FluentDConfig{}
	}

	return tc
}

func (c *clusterClient) SetTunnelConfig(tunnelConfig api.TunnelConfig) error {
	resp := api.ClusterResponse{}

	request := c.c.Put().Resource(clusterPath + tunnelconfigurl)
	request.Body(&tunnelConfig)
	if err := request.Do().Unmarshal(&resp); err != nil {
		return err
	}

	if resp.Error != "" {
		return errors.New(resp.Error)
	}

	return nil
}

func (c *clusterClient) GetTunnelConfig() api.TunnelConfig {
	tc := api.TunnelConfig{}

	if err := c.c.Get().Resource(clusterPath + tunnelconfigurl).Do().Unmarshal(&tc); err != nil {
		return api.TunnelConfig{}
	}

	return tc
}

func (c *clusterClient) GetGossipState() *cluster.ClusterState {
	var status *cluster.ClusterState

	if err := c.c.Get().Resource(clusterPath + "/gossipstate").Do().Unmarshal(&status); err != nil {
		return nil
	}
	return status
}

func (c *clusterClient) EnumerateAlerts(ts, te time.Time, resource api.ResourceType) (*api.Alerts, error) {
	a := api.Alerts{}
	request := c.c.Get().Resource(clusterPath+"/alerts/" + strconv.FormatInt(int64(resource), 10))
	if !te.IsZero() {
		request.QueryOption("timestart", ts.Format(api.TimeLayout))
		request.QueryOption("timeend", te.Format(api.TimeLayout))
	}
	if err := request.Do().Unmarshal(&a); err != nil {
		return nil, err
	}
	return &a, nil
}

func (c *clusterClient) ClearAlert(resource api.ResourceType, alertID int64) error {
	path := clusterPath + "/alerts/" + strconv.FormatInt(int64(resource), 10) + "/" + strconv.FormatInt(alertID, 10)
	request := c.c.Put().Resource(path)
	resp := request.Do()
	if resp.Error() != nil {
		return resp.FormatError()
	}
	return nil
}

func (c *clusterClient) EraseAlert(resource api.ResourceType, alertID int64) error {
	path := clusterPath + "/alerts/" + strconv.FormatInt(int64(resource), 10) + "/" + strconv.FormatInt(alertID, 10)
	request := c.c.Delete().Resource(path)
	resp := request.Do()
	if resp.Error() != nil {
		return resp.FormatError()
	}
	return nil
}
