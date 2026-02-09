/*
Copyright 2018 The Kubernetes Authors.

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

// Package podlogs enables live capturing of all events and log
// messages for some or all pods in a namespace as they get generated.
// This helps debugging both a running test (what is currently going
// on?) and the output of a CI run (events appear in chronological
// order and output that normally isn't available like the command
// stdout messages are available).
package podlogs

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
)

// LogOutput determines where output from CopyAllLogs goes.
//
// Error messages about receiving log output is kept
// separate from the log output and optionally goes to StatusWriter
//
// The log output can go to one or more possible destinations.
type LogOutput struct {
	// If not nil, errors encountered will be logged here.
	StatusWriter io.Writer

	// If not nil, all container output goes to this writer with "<pod>/<container>:" as prefix.
	LogWriter io.Writer

	// Base directory for one log file per container.
	// The full path of each log file will be <log path prefix><pod>-<container>.log,
	// if not empty.
	LogPathPrefix string

	// LogOpen, if not nil, gets called whenever log output watching starts for
	// a certain container. Returning nil means that the output can be discarded
	// unless it gets written elsewhere.
	//
	// The container's stdout and stderr output get written to the returned writer.
	// The writer gets closed once all output is processed if the writer implements io.Closer.
	// Each write is a single line, including a newline.
	// Write errors are ignored.
	LogOpen func(podName, containerName string) io.Writer
}

// Matches harmless errors from pkg/kubelet/kubelet_pods.go.
var expectedErrors = regexp.MustCompile(`container .* in pod .* is (terminated|waiting to start|not available)|the server could not find the requested resource|context canceled`)

// CopyPodLogs is basically CopyPodLogs for all current or future pods in the given namespace ns
func CopyAllLogs(ctx context.Context, cs clientset.Interface, ns string, to LogOutput) error {
	return CopyPodLogs(ctx, cs, ns, "", to)
}

// CopyPodLogs follows the logs of all containers in pod with the given podName,
// including those that get created in the future, and writes each log
// line as configured in the output options. It does that until the
// context is done or until an error occurs.
//
// If podName is empty, it will follow all pods in the given namespace ns.
//
// Beware that there is currently no way to force log collection
// before removing pods, which means that there is a known race
// between "stop pod" and "collecting log entries". The alternative
// would be a blocking function with collects logs from all currently
// running pods, but that then would have the disadvantage that
// already deleted pods aren't covered.
//
// Another race occurs is when a pod shuts down. Logging stops, but if
// then the pod is not removed from the apiserver quickly enough, logging
// resumes and dumps the old log again. Previously, this was allowed based
// on the assumption that it is better to log twice than miss log messages
// of pods that started and immediately terminated or when logging temporarily
// stopped.
//
// But it turned out to be rather confusing, so now a heuristic is used: if
// log output of a container was already captured, then capturing does not
// resume if the pod is marked for deletion.
func CopyPodLogs(ctx context.Context, cs clientset.Interface, ns, podName string, to LogOutput) error {
	options := meta.ListOptions{}
	if podName != "" {
		options = meta.ListOptions{
			FieldSelector: fmt.Sprintf("metadata.name=%s", podName),
		}
	}
	watcher, err := cs.CoreV1().Pods(ns).Watch(ctx, options)

	if err != nil {
		return fmt.Errorf("cannot create Pod event watcher: %w", err)
	}

	go func() {
		var m sync.Mutex
		// Key is pod/container name, true if currently logging it.
		active := map[string]bool{}
		// Key is pod/container/container-id, true if we have ever started to capture its output.
		started := map[string]bool{}
		// Key is pod/container/container-id, value the time stamp of the last log line that has been seen.
		latest := map[string]*meta.Time{}

		check := func() {
			m.Lock()
			defer m.Unlock()

			pods, err := cs.CoreV1().Pods(ns).List(ctx, options)
			if err != nil {
				if ctx.Err() == nil && to.StatusWriter != nil {
					_, _ = fmt.Fprintf(to.StatusWriter, "ERROR: get pod list in %s: %s\n", ns, err)
				}
				return
			}

			for _, pod := range pods.Items {
				for i, c := range pod.Spec.Containers {
					// sanity check, array should have entry for each container
					if len(pod.Status.ContainerStatuses) <= i {
						continue
					}
					name := pod.ObjectMeta.Name + "/" + c.Name
					id := name + "/" + pod.Status.ContainerStatuses[i].ContainerID
					if active[name] ||
						// If we have worked on a container before and it has now terminated, then
						// there cannot be any new output and we can ignore it.
						(pod.Status.ContainerStatuses[i].State.Terminated != nil &&
							started[id]) ||
						// State.Terminated might not have been updated although the container already
						// stopped running. Also check whether the pod is deleted.
						(pod.DeletionTimestamp != nil && started[id]) ||
						// Don't attempt to get logs for a container unless it is running or has terminated.
						// Trying to get a log would just end up with an error that we would have to suppress.
						(pod.Status.ContainerStatuses[i].State.Running == nil &&
							pod.Status.ContainerStatuses[i].State.Terminated == nil) {
						continue
					}

					// Determine where we write. If this fails, we intentionally return without clearing
					// the active[name] flag, which prevents trying over and over again to
					// create the output file.
					var logWithPrefix, logWithoutPrefix, output io.Writer
					var prefix string
					podHandled := false
					if to.LogWriter != nil {
						podHandled = true
						logWithPrefix = to.LogWriter
						nodeName := pod.Spec.NodeName
						if len(nodeName) > 10 {
							nodeName = nodeName[0:4] + ".." + nodeName[len(nodeName)-4:]
						}
						prefix = name + "@" + nodeName + ": "
					}
					if to.LogPathPrefix != "" {
						podHandled = true
						filename := to.LogPathPrefix + pod.ObjectMeta.Name + "-" + c.Name + ".log"
						if err := os.MkdirAll(path.Dir(filename), 0755); err != nil {
							if to.StatusWriter != nil {
								_, _ = fmt.Fprintf(to.StatusWriter, "ERROR: pod log: create directory for %s: %s\n", filename, err)
							}
							return
						}
						// The test suite might run the same test multiple times,
						// so we have to append here.
						file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
						if err != nil {
							if to.StatusWriter != nil {
								_, _ = fmt.Fprintf(to.StatusWriter, "ERROR: pod log: create file %s: %s\n", filename, err)
							}
							return
						}
						logWithoutPrefix = file
					}
					if to.LogOpen != nil {
						if writer := to.LogOpen(pod.Name, c.Name); writer != nil {
							podHandled = true
							output = writer
						}
					}

					if !podHandled {
						// No-one is interested in this pod, don't bother...
						continue
					}

					closeOutput := func() {
						// Execute all of these, even if one one of them panics.
						defer maybeClose(logWithPrefix)
						defer maybeClose(logWithoutPrefix)
						defer maybeClose(output)
					}

					sinceTime := latest[id]
					readCloser, err := logsForPod(ctx, cs, ns, pod.ObjectMeta.Name,
						&v1.PodLogOptions{
							Container:  c.Name,
							Follow:     true,
							Timestamps: true,
							SinceTime:  sinceTime,
						})
					if err != nil {
						closeOutput()

						// We do get "normal" errors here, like trying to read too early.
						// We can ignore those.
						if to.StatusWriter != nil &&
							expectedErrors.FindStringIndex(err.Error()) == nil {
							_, _ = fmt.Fprintf(to.StatusWriter, "WARNING: pod log: %s: %s\n", name, err)
						}
						continue
					}

					active[name] = true
					started[id] = true

					go func() {
						defer closeOutput()
						first := true
						defer func() {
							m.Lock()
							// If we never printed anything, then also skip the final message.
							if !first {
								if logWithPrefix != nil {
									_, _ = fmt.Fprintf(logWithPrefix, "%s==== end of pod log at %s ====\n", prefix, latest[id])
								}
								if logWithoutPrefix != nil {
									_, _ = fmt.Fprintf(logWithoutPrefix, "==== end of pod log for container %s at %s ====\n", name, latest[id])
								}
							}
							active[name] = false
							m.Unlock()
							readCloser.Close()
						}()
						scanner := bufio.NewScanner(readCloser)
						var latestTS time.Time
						for scanner.Scan() {
							line := scanner.Text()
							// Filter out the expected "end of stream" error message,
							// it would just confuse developers who don't know about it.
							// Same for attempts to read logs from a container that
							// isn't ready (yet?!).
							if !strings.HasPrefix(line, "rpc error: code = Unknown desc = Error: No such container:") &&
								!strings.HasPrefix(line, "unable to retrieve container logs for ") &&
								!strings.HasPrefix(line, "Unable to retrieve container logs for ") {
								if first {
									// Because the same log might be written to multiple times
									// in different test instances, log an extra line to separate them.
									// Also provides some useful extra information.
									since := "(initial part)"
									if sinceTime != nil {
										since = fmt.Sprintf("since %s", sinceTime)
									}
									if logWithPrefix != nil {
										_, _ = fmt.Fprintf(logWithPrefix, "%s==== start of pod log %s ====\n", prefix, since)
									}
									if logWithoutPrefix != nil {
										_, _ = fmt.Fprintf(logWithoutPrefix, "==== start of pod log for container %s %s ====\n", name, since)
									}
									first = false
								}
								index := strings.Index(line, " ")
								if index > 0 {
									ts, err := time.Parse(time.RFC3339, line[:index])
									if err == nil {
										latestTS = ts
										// Log output typically has it's own log header with a time stamp,
										// so let's strip the PodLogOptions time stamp.
										line = line[index+1:]
									}
								}
								if logWithPrefix != nil {
									_, _ = fmt.Fprintf(logWithPrefix, "%s%s\n", prefix, line)
								}
								if logWithoutPrefix != nil {
									_, _ = fmt.Fprintln(logWithoutPrefix, line)
								}
								if output != nil {
									_, _ = output.Write([]byte(line + "\n"))
								}
							}
						}

						if !latestTS.IsZero() {
							m.Lock()
							defer m.Unlock()
							latest[id] = &meta.Time{Time: latestTS}
						}
					}()
				}
			}
		}

		// Watch events to see whether we can start logging
		// and log interesting ones. Also check periodically,
		// in case of failures which are not followed by
		// some pod change.
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		check()
		for {
			select {
			case <-watcher.ResultChan():
				check()
			case <-ticker.C:
			case <-ctx.Done():
				return
			}
		}
	}()

	return nil
}

func maybeClose(writer io.Writer) {
	if closer, ok := writer.(io.Closer); ok {
		_ = closer.Close()
	}
}

// logsForPod starts reading the logs for a certain pod. If the pod has more than one
// container, opts.Container must be set. Reading stops when the context is done.
// The stream includes formatted error messages and ends with
//
//	rpc error: code = Unknown desc = Error: No such container: 41a...
//
// when the pod gets deleted while streaming.
func logsForPod(ctx context.Context, cs clientset.Interface, ns, pod string, opts *v1.PodLogOptions) (io.ReadCloser, error) {
	return cs.CoreV1().Pods(ns).GetLogs(pod, opts).Stream(ctx)
}

// WatchPods prints pod status events for a certain namespace or all namespaces
// when namespace name is empty. The closer can be nil if the caller doesn't want
// the file to be closed when watching stops.
func WatchPods(ctx context.Context, cs clientset.Interface, ns string, to io.Writer, toCloser io.Closer) (finalErr error) {
	defer func() {
		if finalErr != nil && toCloser != nil {
			toCloser.Close()
		}
	}()

	pods, err := cs.CoreV1().Pods(ns).Watch(ctx, meta.ListOptions{})
	if err != nil {
		return fmt.Errorf("cannot create Pod watcher: %w", err)
	}
	defer func() {
		if finalErr != nil {
			pods.Stop()
		}
	}()

	events, err := cs.CoreV1().Events(ns).Watch(ctx, meta.ListOptions{})
	if err != nil {
		return fmt.Errorf("cannot create Event watcher: %w", err)
	}

	go func() {
		defer func() {
			pods.Stop()
			events.Stop()
			if toCloser != nil {
				toCloser.Close()
			}
		}()
		timeFormat := "15:04:05.000"
		for {
			select {
			case e := <-pods.ResultChan():
				if e.Object == nil {
					continue
				}

				pod, ok := e.Object.(*v1.Pod)
				if !ok {
					continue
				}
				buffer := new(bytes.Buffer)
				fmt.Fprintf(buffer,
					"%s pod: %s: %s/%s %s: %s %v\n",
					time.Now().Format(timeFormat),
					e.Type,
					pod.Namespace,
					pod.Name,
					pod.Status.Phase,
					pod.Status.Reason,
					pod.Status.Conditions,
				)
				for _, cst := range pod.Status.ContainerStatuses {
					fmt.Fprintf(buffer, "   %s: ", cst.Name)
					if cst.State.Waiting != nil {
						fmt.Fprintf(buffer, "WAITING: %s - %s",
							cst.State.Waiting.Reason,
							cst.State.Waiting.Message,
						)
					} else if cst.State.Running != nil {
						fmt.Fprintf(buffer, "RUNNING")
					} else if cst.State.Terminated != nil {
						fmt.Fprintf(buffer, "TERMINATED: %s - %s",
							cst.State.Terminated.Reason,
							cst.State.Terminated.Message,
						)
					}
					fmt.Fprintf(buffer, "\n")
				}
				to.Write(buffer.Bytes())
			case e := <-events.ResultChan():
				if e.Object == nil {
					continue
				}

				event, ok := e.Object.(*v1.Event)
				if !ok {
					continue
				}
				to.Write([]byte(fmt.Sprintf("%s event: %s/%s %s: %s %s: %s (%v - %v)\n",
					time.Now().Format(timeFormat),
					event.InvolvedObject.APIVersion,
					event.InvolvedObject.Kind,
					event.InvolvedObject.Name,
					event.Source.Component,
					event.Type,
					event.Message,
					event.FirstTimestamp,
					event.LastTimestamp,
				)))
			case <-ctx.Done():
				to.Write([]byte(fmt.Sprintf("%s ==== stopping pod watch ====\n",
					time.Now().Format(timeFormat))))
				return
			}
		}
	}()

	return nil
}
