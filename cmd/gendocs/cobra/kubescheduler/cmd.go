/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubescheduler

import (
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/plugin/cmd/kube-scheduler/app"
)

func Cmd() *cobra.Command {
	cmd := &cobra.Command{
		Use: "kube-scheduler",
		Long: `The kubernetes scheduler is a policy-rich, topology-aware,
workload-specific function that significantly impacts availability, performance,
and capacity. The scheduler needs to take into account individual and collective
resource requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality, inter-workload
interference, deadlines, and so on. Workload-specific requirements will be exposed
through the API as necessary.
`,
	}
	(&app.SchedulerServer{}).AddFlags(cmd.Flags())
	return cmd
}
