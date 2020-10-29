package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"gopkg.in/natefinch/lumberjack.v2"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
)

func main() {
	os.Exit(run())
}

func run() int {
	terminationLog := flag.String("termination-log-file", "", "Write logs after SIGTERM to this file (in addition to stderr)")
	terminationLock := flag.String("termination-touch-file", "", "Touch this file on SIGTERM and delete on termination")
	processOverlapDetectionFile := flag.String("process-overlap-detection-file", "", "This file is present when the kube-apiserver initialization timed out while waiting for kubelet to terminate old process")
	kubeconfigPath := flag.String("kubeconfig", "", "Optional kubeconfig used to create events")
	gracefulTerminatioPeriod := flag.Duration("graceful-termination-duration", 105*time.Second, "The duration of the graceful termination period, e.g. 105s")

	klog.InitFlags(nil)
	flag.Set("v", "9")

	// never log to stderr, only through our termination log writer (which sends it also to stderr)
	flag.Set("logtostderr", "false")
	flag.Set("stderrthreshold", "99")

	flag.Parse()
	args := flag.CommandLine.Args()

	if len(args) == 0 {
		fmt.Println("Missing command line")
		return 1
	}

	// use special tee-like writer when termination log is set
	termCh := make(chan struct{})
	var stderr io.Writer = os.Stderr
	var terminationLogger *terminationFileWriter
	if len(*terminationLog) > 0 {
		terminationLogger = &terminationFileWriter{
			Writer:             os.Stderr,
			fn:                 *terminationLog,
			startFileLoggingCh: termCh,
		}
		stderr = terminationLogger

		// do the klog file writer dance: klog writes to all outputs of lower
		// severity. No idea why. So we discard for anything other than info.
		// Otherwise, we would see errors multiple times.
		klog.SetOutput(ioutil.Discard)
		klog.SetOutputBySeverity("INFO", stderr)
	}

	var client kubernetes.Interface
	if len(*kubeconfigPath) > 0 {
		loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(&clientcmd.ClientConfigLoadingRules{ExplicitPath: *kubeconfigPath}, &clientcmd.ConfigOverrides{})
		if cfg, err := loader.ClientConfig(); err != nil {
			klog.Errorf("failed to load kubeconfig %q: %v", *kubeconfigPath, err)
			return 1
		} else {
			client = kubernetes.NewForConfigOrDie(cfg)
		}
	}

	if processOverlapDetectionFile != nil && len(*processOverlapDetectionFile) > 0 {
		var deleteDetectionFileOnce sync.Once

		if _, err := os.Stat(*processOverlapDetectionFile); err != nil && !os.IsNotExist(err) {
			klog.Errorf("failed to read process overlap detection file %q: %v", *processOverlapDetectionFile, err)
			return 1
		} else if err == nil {
			ref, err := eventReference()
			if err != nil {
				klog.Errorf("failed to get event target: %v", err)
				return 1
			}
			go func() {
				defer deleteDetectionFileOnce.Do(func() {
					if err := os.Remove(*processOverlapDetectionFile); err != nil {
						klog.Warningf("Failed to remove process overlap termination file %q: %v", *processOverlapDetectionFile, err)
					}
				})
				if err := retry.OnError(retry.DefaultBackoff, func(err error) bool {
					select {
					case <-termCh:
						// stop retry on termination
						return false
					default:
					}
					// every error is retriable
					return true
				}, func() error {
					return eventf(client.CoreV1().Events(ref.Namespace), *ref, corev1.EventTypeWarning, "TerminationProcessOverlapDetected", "The kube-apiserver initialization timed out while waiting for kubelet to terminate old process")
				}); err != nil {
					klog.Warning(err)
				}
			}()
		}
	}

	// touch file early. If the file is not removed on termination, we are not
	// terminating cleanly via SIGTERM.
	if len(*terminationLock) > 0 {
		ref, err := eventReference()
		if err != nil {
			klog.Errorf("failed to get event target: %v", err)
			return 1
		}

		if st, err := os.Stat(*terminationLock); err == nil {
			podName := "unknown"
			if v := os.Getenv("POD_NAME"); len(v) > 0 {
				podName = v // pod name is always the same for static pods
			}
			msg := fmt.Sprintf("Previous pod %s started at %s did not terminate gracefully", podName, st.ModTime().String())

			klog.Warning(msg)
			_, _ = terminationLogger.WriteToTerminationLog([]byte(msg + "\n"))

			go retry.OnError(retry.DefaultBackoff, func(err error) bool {
				select {
				case <-termCh:
					// stop retry on termination
					return false
				default:
				}
				// every error is retriable
				return true
			}, func() error {
				return eventf(client.CoreV1().Events(ref.Namespace), *ref, corev1.EventTypeWarning, "NonGracefulTermination", msg)
			})

			klog.Infof("Deleting old termination lock file %q", *terminationLock)
			if err := os.Remove(*terminationLock); err != nil {
				klog.Errorf("Old termination lock file deletion failed: %v", err)
			}
		}

		// separation to see where the new one is starting
		_, _ = terminationLogger.WriteToTerminationLog([]byte("---\n"))

		klog.Infof("Touching termination lock file %q", *terminationLock)
		if err := touch(*terminationLock); err != nil {
			klog.Infof("Error touching %s: %v", *terminationLock, err)
			// keep going
		}

		var deleteLockOnce sync.Once

		if *gracefulTerminatioPeriod > 2*time.Second {
			go func() {
				<-termCh
				<-time.After(*gracefulTerminatioPeriod - 2*time.Second)

				deleteLockOnce.Do(func() {
					klog.Infof("Graceful termination time nearly passed and kube-apiserver has still not terminated. Deleting termination lock file %q to avoid a false positive.", *terminationLock)
					if err := os.Remove(*terminationLock); err != nil {
						klog.Errorf("Termination lock file deletion failed: %v", err)
					}

					if err := eventf(client.CoreV1().Events(ref.Namespace), *ref, corev1.EventTypeWarning, "GracefulTerminationTimeout", "kube-apiserver did not terminate within %s", *gracefulTerminatioPeriod); err != nil {
						klog.Error(err)
					}
				})
			}()
		}

		defer deleteLockOnce.Do(func() {
			klog.Infof("Deleting termination lock file %q", *terminationLock)
			if err := os.Remove(*terminationLock); err != nil {
				klog.Errorf("Termination lock file deletion failed: %v", err)
			}
		})
	}

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = stderr

	// forward SIGTERM and SIGINT to child
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for s := range sigCh {
			select {
			case <-termCh:
			default:
				close(termCh)
			}

			klog.Infof("Received signal %s. Forwarding to sub-process %q.", s, args[0])

			cmd.Process.Signal(s)
		}
	}()

	klog.Infof("Launching sub-process %q", cmd)
	rc := 0
	if err := cmd.Run(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			rc = exitError.ExitCode()
		} else {
			klog.Infof("Failed to launch %s: %v", args[0], err)
			return 255
		}
	}

	// remove signal handling
	signal.Stop(sigCh)
	close(sigCh)
	wg.Wait()

	klog.Infof("Termination finished with exit code %d", rc)
	return rc
}

// terminationFileWriter forwards everything to the embedded writer. When
// startFileLoggingCh is closed, everything is appended to the given file name
// in addition.
type terminationFileWriter struct {
	io.Writer
	fn                 string
	startFileLoggingCh <-chan struct{}

	logger io.Writer
}

func (w *terminationFileWriter) WriteToTerminationLog(bs []byte) (int, error) {
	if w == nil {
		return len(bs), nil
	}

	if w.logger == nil {
		l := &lumberjack.Logger{
			Filename:   w.fn,
			MaxSize:    100,
			MaxBackups: 3,
			MaxAge:     28,
			Compress:   false,
		}
		w.logger = l
		fmt.Fprintf(os.Stderr, "Copying termination logs to %q\n", w.fn)
	}
	if n, err := w.logger.Write(bs); err != nil {
		return n, err
	} else if n != len(bs) {
		return n, io.ErrShortWrite
	}
	return len(bs), nil
}

func (w *terminationFileWriter) Write(bs []byte) (int, error) {
	// temporary hack to avoid logging sensitive tokens.
	// TODO: drop when we moved to a non-sensitive storage format
	if strings.Contains(string(bs), "URI=\"/apis/oauth.openshift.io/v1/oauthaccesstokens/") || strings.Contains(string(bs), "URI=\"/apis/oauth.openshift.io/v1/oauthauthorizetokens/") {
		return len(bs), nil
	}

	select {
	case <-w.startFileLoggingCh:
		if n, err := w.WriteToTerminationLog(bs); err != nil {
			return n, err
		}
	default:
	}

	return w.Writer.Write(bs)
}

func touch(fn string) error {
	_, err := os.Stat(fn)
	if os.IsNotExist(err) {
		file, err := os.Create(fn)
		if err != nil {
			return err
		}
		defer file.Close()
		return nil
	}

	currentTime := time.Now().Local()
	return os.Chtimes(fn, currentTime, currentTime)
}

func eventf(client corev1client.EventInterface, ref corev1.ObjectReference, eventType, reason, messageFmt string, args ...interface{}) error {
	t := metav1.Time{Time: time.Now()}
	host, _ := os.Hostname() // expicitly ignore error. Empty host is fine

	e := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: ref.Namespace,
		},
		InvolvedObject: ref,
		Reason:         reason,
		Message:        fmt.Sprintf(messageFmt, args...),
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
		Type:           eventType,
		Source:         corev1.EventSource{Component: "apiserver", Host: host},
	}

	_, err := client.Create(context.TODO(), e, metav1.CreateOptions{})

	if err == nil {
		klog.V(2).Infof("Event(%#v): type: '%v' reason: '%v' %v", e.InvolvedObject, e.Type, e.Reason, e.Message)
	}

	return err
}

func eventReference() (*corev1.ObjectReference, error) {
	ns := os.Getenv("POD_NAMESPACE")
	pod := os.Getenv("POD_NAME")
	if len(ns) == 0 && len(pod) > 0 {
		serviceAccountNamespaceFile := "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
		if _, err := os.Stat(serviceAccountNamespaceFile); err == nil {
			bs, err := ioutil.ReadFile(serviceAccountNamespaceFile)
			if err != nil {
				return nil, err
			}
			ns = string(bs)
		}
	}
	if len(ns) == 0 {
		pod = ""
		ns = "kube-system"
	}
	if len(pod) == 0 {
		return &corev1.ObjectReference{
			Kind:       "Namespace",
			Name:       ns,
			APIVersion: "v1",
		}, nil
	}

	return &corev1.ObjectReference{
		Kind:       "Pod",
		Namespace:  ns,
		Name:       pod,
		APIVersion: "v1",
	}, nil
}
