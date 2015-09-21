package gophercloud

import (
	"errors"
	"fmt"
	"github.com/racker/perigee"
)

func (gsp *genericServersProvider) ListFloatingIps() ([]FloatingIp, error) {
	var fips []FloatingIp

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/os-floating-ips"
		return perigee.Get(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			Results: &struct {
				FloatingIps *[]FloatingIp `json:"floating_ips"`
			}{&fips},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return fips, err
}

func (gsp *genericServersProvider) CreateFloatingIp(pool string) (FloatingIp, error) {
	fip := new(FloatingIp)

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/os-floating-ips"
		return perigee.Post(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			ReqBody: map[string]string{
				"pool": pool,
			},
			Results: &struct {
				FloatingIp **FloatingIp `json:"floating_ip"`
			}{&fip},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})

	if fip.Ip == "" {
		return *fip, errors.New("Error creating floating IP")
	}

	return *fip, err
}

func (gsp *genericServersProvider) AssociateFloatingIp(serverId string, ip FloatingIp) error {
	return gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, serverId)
		return perigee.Post(ep, perigee.Options{
			CustomClient: gsp.context.httpClient,
			ReqBody: map[string](map[string]string){
				"addFloatingIp": map[string]string{"address": ip.Ip},
			},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})
}

func (gsp *genericServersProvider) DeleteFloatingIp(ip FloatingIp) error {
	return gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-floating-ips/%d", gsp.endpoint, ip.Id)
		return perigee.Delete(ep, perigee.Options{
			CustomClient: gsp.context.httpClient,
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})
}

type FloatingIp struct {
	Id         int    `json:"id"`
	Pool       string `json:"pool"`
	Ip         string `json:"ip"`
	FixedIp    string `json:"fixed_ip"`
	InstanceId string `json:"instance_id"`
}
