# Package: v1beta1

Protobuf-generated gRPC API for example plugins (version v1beta1).

## Generated Files

- **api.pb.go**: Protocol buffer message types
- **api_grpc.pb.go**: gRPC service client and server interfaces

## Purpose

This package provides a versioned gRPC API for testing the plugin watcher and registration mechanism. It demonstrates how plugins communicate with kubelet via gRPC over Unix domain sockets.

## Proto Source

Generated from: `pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta1/api.proto`

## Usage

Used by example_plugin.go in the parent pluginwatcher package for integration testing of the plugin registration workflow.
