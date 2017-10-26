package dnspod

import (
	"fmt"
	"strconv"
	// "time"
)

// DomainsService handles communication with the domain related
// methods of the dnspod API.
//
// dnspod API docs: https://www.dnspod.cn/docs/domains.html
type DomainsService struct {
	client *Client
}

type DomainInfo struct {
	DomainTotal   int    `json:"domain_total,omitempty"`
	AllTotal      int    `json:"all_total,omitempty"`
	MineTotal     int    `json:"mine_total,omitempty"`
	ShareTotal    string `json:"share_total,omitempty"`
	VipTotal      int    `json:"vip_total,omitempty"`
	IsMarkTotal   int    `json:"ismark_total,omitempty"`
	PauseTotal    int    `json:"pause_total,omitempty"`
	ErrorTotal    int    `json:"error_total,omitempty"`
	LockTotal     int    `json:"lock_total,omitempty"`
	SpamTotal     int    `json:"spam_total,omitempty"`
	VipExpire     int    `json:"vip_expire,omitempty"`
	ShareOutTotal int    `json:"share_out_total,omitempty"`
}

type Domain struct {
	ID               int    `json:"id,omitempty"`
	Name             string `json:"name,omitempty"`
	PunyCode         string `json:"punycode,omitempty"`
	Grade            string `json:"grade,omitempty"`
	GradeTitle       string `json:"grade_title,omitempty"`
	Status           string `json:"status,omitempty"`
	ExtStatus        string `json:"ext_status,omitempty"`
	Records          string `json:"records,omitempty"`
	GroupID          string `json:"group_id,omitempty"`
	IsMark           string `json:"is_mark,omitempty"`
	Remark           string `json:"remark,omitempty"`
	IsVIP            string `json:"is_vip,omitempty"`
	SearchenginePush string `json:"searchengine_push,omitempty"`
	UserID           string `json:"user_id,omitempty"`
	CreatedOn        string `json:"created_on,omitempty"`
	UpdatedOn        string `json:"updated_on,omitempty"`
	TTL              string `json:"ttl,omitempty"`
	CNameSpeedUp     string `json:"cname_speedup,omitempty"`
	Owner            string `json:"owner,omitempty"`
	AuthToAnquanBao  bool   `json:"auth_to_anquanbao,omitempty"`
}

type domainListWrapper struct {
	Status  Status     `json:"status"`
	Info    DomainInfo `json:"info"`
	Domains []Domain   `json:"domains"`
}

type domainWrapper struct {
	Status Status     `json:"status"`
	Info   DomainInfo `json:"info"`
	Domain Domain     `json:"domain"`
}

// domainRequest represents a generic wrapper for a domain request,
// when domainWrapper cannot be used because of type constraint on Domain.
type domainRequest struct {
	Domain interface{} `json:"domain"`
}

// domainAction generates the resource path for given domain.
func domainAction(action string) string {
	if len(action) > 0 {
		return fmt.Sprintf("Domain.%s", action)
	}
	return "Domain.List"
}

// List the domains.
//
// dnspod API docs: https://www.dnspod.cn/docs/domains.html#domain-list
func (s *DomainsService) List() ([]Domain, *Response, error) {
	path := domainAction("List")
	returnedDomains := domainListWrapper{}

	payload := newPayLoad(s.client.CommonParams)
	res, err := s.client.post(path, payload, &returnedDomains)
	if err != nil {
		return []Domain{}, res, err
	}

	domains := []Domain{}

	if returnedDomains.Status.Code != "1" {
		return domains, nil, fmt.Errorf("Could not get domains: %s", returnedDomains.Status.Message)
	}

	for _, domain := range returnedDomains.Domains {
		domains = append(domains, domain)
	}

	return domains, res, nil
}

// Create a new domain.
//
// dnspod API docs: https://www.dnspod.cn/docs/domains.html#domain-create
func (s *DomainsService) Create(domainAttributes Domain) (Domain, *Response, error) {
	path := domainAction("Create")
	returnedDomain := domainWrapper{}

	payload := newPayLoad(s.client.CommonParams)
	payload.Set("domain", domainAttributes.Name)
	payload.Set("group_id", domainAttributes.GroupID)
	payload.Set("is_mark", domainAttributes.IsMark)

	res, err := s.client.post(path, payload, &returnedDomain)
	if err != nil {
		return Domain{}, res, err
	}

	return returnedDomain.Domain, res, nil
}

// Get fetches a domain.
//
// dnspod API docs: https://www.dnspod.cn/docs/domains.html#domain-info
func (s *DomainsService) Get(ID int) (Domain, *Response, error) {
	path := domainAction("Info")
	returnedDomain := domainWrapper{}

	payload := newPayLoad(s.client.CommonParams)
	payload.Set("domain_id", strconv.FormatInt(int64(ID), 10))

	res, err := s.client.post(path, payload, &returnedDomain)
	if err != nil {
		return Domain{}, res, err
	}

	return returnedDomain.Domain, res, nil
}

// Delete a domain.
//
// dnspod API docs: https://dnsapi.cn/Domain.Remove
func (s *DomainsService) Delete(ID int) (*Response, error) {
	path := domainAction("Remove")
	returnedDomain := domainWrapper{}

	payload := newPayLoad(s.client.CommonParams)
	payload.Set("domain_id", strconv.FormatInt(int64(ID), 10))

	res, err := s.client.post(path, payload, &returnedDomain)
	if err != nil {
		return res, err
	}

	return res, nil
}
