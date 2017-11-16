package libtrust

import (
	"path/filepath"
)

// FilterByHosts filters the list of PublicKeys to only those which contain a
// 'hosts' pattern which matches the given host. If *includeEmpty* is true,
// then keys which do not specify any hosts are also returned.
func FilterByHosts(keys []PublicKey, host string, includeEmpty bool) ([]PublicKey, error) {
	filtered := make([]PublicKey, 0, len(keys))

	for _, pubKey := range keys {
		var hosts []string
		switch v := pubKey.GetExtendedField("hosts").(type) {
		case []string:
			hosts = v
		case []interface{}:
			for _, value := range v {
				h, ok := value.(string)
				if !ok {
					continue
				}
				hosts = append(hosts, h)
			}
		}

		if len(hosts) == 0 {
			if includeEmpty {
				filtered = append(filtered, pubKey)
			}
			continue
		}

		// Check if any hosts match pattern
		for _, hostPattern := range hosts {
			match, err := filepath.Match(hostPattern, host)
			if err != nil {
				return nil, err
			}

			if match {
				filtered = append(filtered, pubKey)
				continue
			}
		}
	}

	return filtered, nil
}
