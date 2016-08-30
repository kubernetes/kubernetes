package consul

// Status endpoint is used to check on server status
type Status struct {
	server *Server
}

// Ping is used to just check for connectivity
func (s *Status) Ping(args struct{}, reply *struct{}) error {
	return nil
}

// Leader is used to get the address of the leader
func (s *Status) Leader(args struct{}, reply *string) error {
	leader := s.server.raft.Leader()
	if leader != "" {
		*reply = leader
	} else {
		*reply = ""
	}
	return nil
}

// Peers is used to get all the Raft peers
func (s *Status) Peers(args struct{}, reply *[]string) error {
	peers, err := s.server.raftPeers.Peers()
	if err != nil {
		return err
	}

	*reply = peers
	return nil
}
