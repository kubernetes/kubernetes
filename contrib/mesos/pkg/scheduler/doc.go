/*
Copyright 2015 The Kubernetes Authors.

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

// Package scheduler implements the Kubernetes Mesos scheduler.
package scheduler // import "k8s.io/kubernetes/contrib/mesos/pkg/scheduler"

// Created from contrib/mesos/docs/scheduler.monopic:
//
//                     ┌───────────────────────────────────────────────────────────────────────┐
//                     │                ┌───────────────────────────────────────┐            ┌─┴──────────────────────┐             ┌───────────────┐
//            ┌────────▼─────────┐      │Queuer                                 │  Await()   │       podUpdates       │             │               │
//            │ podUpdatesBypass │      │- Yield() *api.Pod                     ├──pod CRUD ─▶ (queue.HistoricalFIFO) ◀──reflector──▶pods ListWatch ├──apiserver──▶
//            └────────▲─────────┘      │- Requeue(pod)/Dequeue(id)/Reoffer(pod)│   events   │                        │             │               │
//                     │                └───────────────────▲───────────────────┘            └───────────┬────────────┘             └───────────────┘
//                     │                                    │                                            │
//                     │                                    │                                            │
//                     └───────────────┐┌───────────────────▲────────────────────▲─────────────────────┐ └───────────────────────┐
//                                     ││                                        │                     │    ┌────────────────────┼─────────────────┐
//                 ┌───────────────────┼┼──────────────────────────────────────┐ │ ┌───────────────────┼────┼───────────┐        │                 │
//     ┌───────────▼──────────┐┌───────┴┴───────┐   ┌───────────────────┐   ┌──┴─┴─┴──────┐   ┌────────┴────┴───┐  ┌────▼────────▼─────────────┐   │
//     │Binder (task launcher)││Deleter         │   │PodReconciler      │   │Controller   │   │  ErrorHandler   │  │SchedulerAlgorithm         │   │
//     │- Bind(binding)       ││- DeleteOne(pod)│   │- Reconcile(pod)   │   │- Run()      │   │- Error(pod, err)│  │- Schedule(pod) -> NodeName│   │
//     │                      ││                │◀──│                   │   │             │──▶│                 │  │                           │   │
//     │               ┌─────┐││    ┌─────┐     │   │      ┌─────┐      │   │   ┌─────┐   │   │    ┌─────┐      │  │┌─────┐                    │   │
//     └───────────────┤sched├┘└────┤sched├─────┘   └──────┤sched├───▲──┘   └───┤sched├───┘   └────┤sched├──────┘  └┤sched├──────────────┬─────┘   │
//                     ├-│││-┴──────┴--││-┴────────────────┴--│--┴───┼──────────┴--│--┴────────────┴-│---┴──────────┴-│││-┤ ┌────────────▼─────────▼─────────┐
//                     │ │││           ││                     │      │             │                 │                │││ │ │          podScheduler          │
//                     │ ││└───────────▼┼─────────────────────▼──────┼─────────────▼─────────────────▼────────────────┘││ │ │    (e.g. fcfsPodScheduler)     │
//                     │ │└─────────────┼────────────────────────────┼─────────────┼──────────────────▼────────────────┘│ │ │                                │
//                     │ │              │                            │             │                  │                 │ │ │  scheduleOne(pod, offers ...)  │
//                     │ │              │                            │             │                  │                 │ │ │     ┌──────────────────────────┤
//                     │ │              │         ╲   │   │   │   ╱  │             │                  │                 ▼ │ │     │    allocationStrategy    │
//                     │ │              │          ╲  └┐  │  ┌┘  ╱   │             │                  │                   │ │     │      - FitPredicate      │
//                     │ │              │           ╲  │  │  │  ╱    │             │                  │                   │ │     │      - Procurement       │
//                     │ │              │            ╲ └┐ │ ┌┘ ╱     │             │                  │                   │ └─────┴──────────────────────────┘
//                     │┌▼────────────┐┌▼──────────┐┌─▼─▼─▼─▼─▼─┐┌───┴────────┐┌───▼───┐         ┌────▼───┐               │
//                     ││LaunchTask(t)││KillTask(t)││sync.Mutex ││reconcile(t)││Tasks()│         │Offers()│               │
//                     │└──────┬──────┘└─────┬─────┘└───────────┘└────────▲───┘└───┬───┘         └────┬───┘               │
//                     │       │             │                            │        │                  │                   │
//                     │       │             └──────────────────┐         │    ┌───▼────────────┐     │                   │
//                     │       └──────────────────────────────┐ │         │    │podtask.Registry│     │                   │
//                     │                                      │ │         │    └────────────────┘     │                   │           ┌──────────────────────┐
//                     │                                      │ │         │                           │                   │           │                      │
//                     │Scheduler                             │ └──────┐  │                           │                   │           │   A ──────────▶ B    │
//                     └──────────────────────────────────────┼────────┼─┬│----┬──────────────────────┼───────────────────┘           │                      │
//                     ┌──────────────────────────────────────┼────────┼─┤sched├──────────────────────┼─────────────────────────┐     │  A has a reference   │
//                     │Framework                             │        │ └─────┘                 ┌────▼───┐                     │     │   on B and calls B   │
//                     │                               ┌──────▼──────┐┌▼──────────┐              │Offers()│                     │     │                      │
//                     │                               │LaunchTask(t)││KillTask(t)│              └────┬───┘                     │     └──────────────────────┘
//                     │                               └─────────┬───┘└──────┬────┘          ┌────────▼───────┐                 │
//                     │implements: mesos-go/scheduler.Scheduler └───────────▼               │offers.Registry │                 │
//                     │                                                     │               └────────────────┘                 │
//                     │                        ┌─────────────────┐       ┌──▼─────────────┐                                    │
//                     └────────────────────────┤                 ├───────┤     Mesos      ├────────────────────────────────────┘
//                                              │ TasksReconciler │       │   Scheduler    │
//                                              │                 ├───────▶     Driver     │
//                                              └─────────────────┘       └────────┬───────┘
//                                                                                 │
//                                                                                 │
//                                                                                 ▼
