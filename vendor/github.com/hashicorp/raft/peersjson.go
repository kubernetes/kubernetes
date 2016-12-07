package raft

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
)

// ReadPeersJSON consumes a legacy peers.json file in the format of the old JSON
// peer store and creates a new-style configuration structure. This can be used
// to migrate this data or perform manual recovery when running protocol versions
// that can interoperate with older, unversioned Raft servers. This should not be
// used once server IDs are in use, because the old peers.json file didn't have
// support for these, nor non-voter suffrage types.
func ReadPeersJSON(path string) (Configuration, error) {
	// Read in the file.
	buf, err := ioutil.ReadFile(path)
	if err != nil {
		return Configuration{}, err
	}

	// Parse it as JSON.
	var peers []string
	dec := json.NewDecoder(bytes.NewReader(buf))
	if err := dec.Decode(&peers); err != nil {
		return Configuration{}, err
	}

	// Map it into the new-style configuration structure. We can only specify
	// voter roles here, and the ID has to be the same as the address.
	var configuration Configuration
	for _, peer := range peers {
		server := Server{
			Suffrage: Voter,
			ID:       ServerID(peer),
			Address:  ServerAddress(peer),
		}
		configuration.Servers = append(configuration.Servers, server)
	}

	// We should only ingest valid configurations.
	if err := checkConfiguration(configuration); err != nil {
		return Configuration{}, err
	}
	return configuration, nil
}
