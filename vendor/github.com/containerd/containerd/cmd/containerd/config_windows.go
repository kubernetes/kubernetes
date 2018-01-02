package main

import "github.com/containerd/containerd/server"

func defaultConfig() *server.Config {
	return &server.Config{
		Root:  server.DefaultRootDir,
		State: server.DefaultStateDir,
		GRPC: server.GRPCConfig{
			Address: server.DefaultAddress,
		},
		Debug: server.Debug{
			Level:   "info",
			Address: server.DefaultDebugAddress,
		},
	}
}
