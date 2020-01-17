package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"

	types "github.com/thecodeteam/goscaleio/types/v1"
)

type ProtectionDomain struct {
	ProtectionDomain *types.ProtectionDomain
	client           *Client
}

func NewProtectionDomain(client *Client) *ProtectionDomain {
	return &ProtectionDomain{
		ProtectionDomain: new(types.ProtectionDomain),
		client:           client,
	}
}

func NewProtectionDomainEx(client *Client, pd *types.ProtectionDomain) *ProtectionDomain {
	return &ProtectionDomain{
		ProtectionDomain: pd,
		client:           client,
	}
}

func (system *System) CreateProtectionDomain(name string) (string, error) {
	endpoint := system.client.SIOEndpoint

	protectionDomainParam := &types.ProtectionDomainParam{}
	protectionDomainParam.Name = name

	jsonOutput, err := json.Marshal(&protectionDomainParam)
	if err != nil {
		return "", fmt.Errorf("error marshaling: %s", err)
	}
	endpoint.Path = fmt.Sprintf("/api/types/ProtectionDomain/instances")

	req := system.client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", system.client.Token)
	req.Header.Add("Accept", "application/json;version="+system.client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+system.client.configConnect.Version)

	resp, err := system.client.retryCheckResp(&system.client.Http, req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.New("error reading body")
	}

	var pd types.ProtectionDomainResp
	err = json.Unmarshal(bs, &pd)
	if err != nil {
		return "", err
	}

	return pd.ID, nil
}

func (system *System) GetProtectionDomain(protectiondomainhref string) (protectionDomains []*types.ProtectionDomain, err error) {

	endpoint := system.client.SIOEndpoint

	if protectiondomainhref == "" {
		link, err := GetLink(system.System.Links, "/api/System/relationship/ProtectionDomain")
		if err != nil {
			return []*types.ProtectionDomain{}, errors.New("Error: problem finding link")
		}

		endpoint.Path = link.HREF
	} else {
		endpoint.Path = protectiondomainhref
	}

	req := system.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", system.client.Token)
	req.Header.Add("Accept", "application/json;version="+system.client.configConnect.Version)

	resp, err := system.client.retryCheckResp(&system.client.Http, req)
	if err != nil {
		return []*types.ProtectionDomain{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if protectiondomainhref == "" {
		if err = system.client.decodeBody(resp, &protectionDomains); err != nil {
			return []*types.ProtectionDomain{}, fmt.Errorf("error decoding instances response: %s", err)
		}
	} else {
		protectionDomain := &types.ProtectionDomain{}
		if err = system.client.decodeBody(resp, &protectionDomain); err != nil {
			return []*types.ProtectionDomain{}, fmt.Errorf("error decoding instances response: %s", err)
		}
		protectionDomains = append(protectionDomains, protectionDomain)

	}
	//
	// bs, err := ioutil.ReadAll(resp.Body)
	// if err != nil {
	// 	return []types.ProtectionDomain{}, errors.New("error reading body")
	// }
	//
	// fmt.Println(string(bs))
	// log.Fatalf("here")
	// return []types.ProtectionDomain{}, nil
	return protectionDomains, nil
}

func (system *System) FindProtectionDomain(id, name, href string) (protectionDomain *types.ProtectionDomain, err error) {
	protectionDomains, err := system.GetProtectionDomain(href)
	if err != nil {
		return &types.ProtectionDomain{}, fmt.Errorf("Error getting protection domains %s", err)
	}

	for _, protectionDomain = range protectionDomains {
		if protectionDomain.ID == id || protectionDomain.Name == name || href != "" {
			return protectionDomain, nil
		}
	}

	return &types.ProtectionDomain{}, errors.New("Couldn't find protection domain")
}
