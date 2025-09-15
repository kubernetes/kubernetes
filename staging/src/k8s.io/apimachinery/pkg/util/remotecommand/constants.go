/*
Copyright 2016 The Kubernetes Authors.

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

package remotecommand

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	DefaultStreamCreationTimeout = 30 * time.Second

	// The SPDY subprotocol "channel.k8s.io" is used for remote command
	// attachment/execution. This represents the initial unversioned subprotocol,
	// which has the known bugs https://issues.k8s.io/13394 and
	// https://issues.k8s.io/13395.
	StreamProtocolV1Name = "channel.k8s.io"

	// The SPDY subprotocol "v2.channel.k8s.io" is used for remote command
	// attachment/execution. It is the second version of the subprotocol and
	// resolves the issues present in the first version.
	StreamProtocolV2Name = "v2.channel.k8s.io"

	// The SPDY subprotocol "v3.channel.k8s.io" is used for remote command
	// attachment/execution. It is the third version of the subprotocol and
	// adds support for resizing container terminals.
	StreamProtocolV3Name = "v3.channel.k8s.io"

	// The SPDY subprotocol "v4.channel.k8s.io" is used for remote command
	// attachment/execution. It is the 4th version of the subprotocol and
	// adds support for exit codes.
	StreamProtocolV4Name = "v4.channel.k8s.io"

	// The subprotocol "v5.channel.k8s.io" is used for remote command
	// attachment/execution. It is the 5th version of the subprotocol and
	// adds support for a CLOSE signal.
	StreamProtocolV5Name = "v5.channel.k8s.io"

	NonZeroExitCodeReason = metav1.StatusReason("NonZeroExitCode")
	ExitCodeCauseType     = metav1.CauseType("ExitCode")

	// RemoteCommand stream identifiers. The first three identifiers (for STDIN,
	// STDOUT, STDERR) are the same as their file descriptors.
	StreamStdIn  = 0
	StreamStdOut = 1
	StreamStdErr = 2
	StreamErr    = 3
	StreamResize = 4
	StreamClose  = 255
)

var SupportedStreamingProtocols = []string{StreamProtocolV4Name, StreamProtocolV3Name, StreamProtocolV2Name, StreamProtocolV1Name}
