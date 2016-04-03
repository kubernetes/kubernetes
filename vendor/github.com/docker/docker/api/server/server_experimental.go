// +build experimental

package server

func (s *Server) registerSubRouter() {
	httpHandler := s.daemon.NetworkApiRouter()

	subrouter := s.router.PathPrefix("/v{version:[0-9.]+}/networks").Subrouter()
	subrouter.Methods("GET", "POST", "PUT", "DELETE").HandlerFunc(httpHandler)
	subrouter = s.router.PathPrefix("/networks").Subrouter()
	subrouter.Methods("GET", "POST", "PUT", "DELETE").HandlerFunc(httpHandler)

	subrouter = s.router.PathPrefix("/v{version:[0-9.]+}/services").Subrouter()
	subrouter.Methods("GET", "POST", "PUT", "DELETE").HandlerFunc(httpHandler)
	subrouter = s.router.PathPrefix("/services").Subrouter()
	subrouter.Methods("GET", "POST", "PUT", "DELETE").HandlerFunc(httpHandler)
}
