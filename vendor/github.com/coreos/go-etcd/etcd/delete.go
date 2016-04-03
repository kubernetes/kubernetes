package etcd

// Delete deletes the given key.
//
// When recursive set to false, if the key points to a
// directory the method will fail.
//
// When recursive set to true, if the key points to a file,
// the file will be deleted; if the key points to a directory,
// then everything under the directory (including all child directories)
// will be deleted.
func (c *Client) Delete(key string, recursive bool) (*Response, error) {
	raw, err := c.RawDelete(key, recursive, false)

	if err != nil {
		return nil, err
	}

	return raw.Unmarshal()
}

// DeleteDir deletes an empty directory or a key value pair
func (c *Client) DeleteDir(key string) (*Response, error) {
	raw, err := c.RawDelete(key, false, true)

	if err != nil {
		return nil, err
	}

	return raw.Unmarshal()
}

func (c *Client) RawDelete(key string, recursive bool, dir bool) (*RawResponse, error) {
	ops := Options{
		"recursive": recursive,
		"dir":       dir,
	}

	return c.delete(key, ops)
}
