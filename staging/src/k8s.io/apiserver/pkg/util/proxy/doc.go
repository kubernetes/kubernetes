/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Among other files, this directory contains functionality for two
// stream proxies: streamtranslator.go and streamtunnel.go. Both of
// these proxies allow the inter-connection of WebSocket and SPDY
// streaming connections.
//
// The stream translator proxy is used for the RemoteCommand
// subprotocol (e.g. kubectl exec, cp, and attach), and it connects
// the output streams of a WebSocket connection (e.g. STDIN, STDOUT,
// STDERR, TTY resize, and error streams) to the input streams of a
// SPDY connection.
//
// The stream tunnel proxy tunnels SPDY frames through a WebSocket
// connection, and it is used for the PortForward subprotocol (e.g.
// kubectl port-forward). This proxy implements tunneling by transparently
// encoding and decoding SPDY framed data into and out of the payload of a
// WebSocket data frame. The primary structure for this tunneling is
// the TunnelingConnection. A lot of the other code in streamtunnel.go
// is for properly upgrading both the upstream SPDY connection and the
// downstream WebSocket connection before streaming begins.
package proxy
