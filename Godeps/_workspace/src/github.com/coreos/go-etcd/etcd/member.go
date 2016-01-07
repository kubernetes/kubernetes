package etcd

import "encoding/json"

type Member struct {
	ID         string   `json:"id"`
	Name       string   `json:"name"`
	PeerURLs   []string `json:"peerURLs"`
	ClientURLs []string `json:"clientURLs"`
}

type memberCollection []Member

func (c *memberCollection) UnmarshalJSON(data []byte) error {
	d := struct {
		Members []Member
	}{}

	if err := json.Unmarshal(data, &d); err != nil {
		return err
	}

	if d.Members == nil {
		*c = make([]Member, 0)
		return nil
	}

	*c = d.Members
	return nil
}
