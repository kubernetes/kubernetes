package etcd

import "fmt"

func (c *Client) CompareAndSwap(key string, value string, ttl uint64,
	prevValue string, prevIndex uint64) (*Response, error) {
	raw, err := c.RawCompareAndSwap(key, value, ttl, prevValue, prevIndex)
	if err != nil {
		return nil, err
	}

	return raw.Unmarshal()
}

func (c *Client) RawCompareAndSwap(key string, value string, ttl uint64,
	prevValue string, prevIndex uint64) (*RawResponse, error) {
	if prevValue == "" && prevIndex == 0 {
		return nil, fmt.Errorf("You must give either prevValue or prevIndex.")
	}

	options := Options{}
	if prevValue != "" {
		options["prevValue"] = prevValue
	}
	if prevIndex != 0 {
		options["prevIndex"] = prevIndex
	}

	raw, err := c.put(key, value, ttl, options)

	if err != nil {
		return nil, err
	}

	return raw, err
}
