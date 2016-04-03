/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/auth"
	"github.com/mesos/mesos-go/auth/sasl"
	"github.com/mesos/mesos-go/auth/sasl/mech"
	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	sched "github.com/mesos/mesos-go/scheduler"
	"golang.org/x/net/context"
)

const (
	CPUS_PER_TASK       = 1
	MEM_PER_TASK        = 128
	DISK_PER_TASK       = 20
	defaultArtifactPort = 12345
)

var (
	address      = flag.String("address", "127.0.0.1", "Binding address for artifact server")
	artifactPort = flag.Int("artifactPort", defaultArtifactPort, "Binding port for artifact server")
	authProvider = flag.String("mesos_authentication_provider", sasl.ProviderName,
		fmt.Sprintf("Authentication provider to use, default is SASL that supports mechanisms: %+v", mech.ListSupported()))
	master              = flag.String("master", "127.0.0.1:5050", "Master address <ip:port>")
	executorPath        = flag.String("executor", "./executor", "Path to test executor")
	taskCount           = flag.String("task-count", "5", "Total task count to run.")
	role                = flag.String("role", "golang", "Role for framework and tasks.")
	mesosAuthPrincipal  = flag.String("mesos_authentication_principal", "", "Mesos authentication principal.")
	mesosAuthSecretFile = flag.String("mesos_authentication_secret_file", "", "Mesos authentication secret file.")
	execUri             *string
	execCmd             string
)

type ExampleScheduler struct {
	tasks []*ExampleTask
}

type ExampleTaskState int

const (
	InitState ExampleTaskState = iota
	RequestedReservationState
	LaunchedState
	FinishedState
	UnreservedState
)

type ExampleTask struct {
	executor      *mesos.ExecutorInfo
	name          string
	persistenceId string
	containerPath string
	state         ExampleTaskState
}

func newExampleScheduler() *ExampleScheduler {
	total, err := strconv.Atoi(*taskCount)
	if err != nil {
		total = 5
	}

	tasks := make([]*ExampleTask, total)
	for i := 0; i < total; i++ {
		tasks[i] = &ExampleTask{
			executor:      prepareExecutorInfo(fmt.Sprintf("persistent-task-executor-%d", i+1)),
			name:          fmt.Sprintf("persistent-task-%d", i+1),
			persistenceId: fmt.Sprintf("persistence-%d", i+1),
			containerPath: "data",
			state:         InitState,
		}
	}

	return &ExampleScheduler{
		tasks: tasks,
	}
}

func getResource(name string, resources []*mesos.Resource, withReservation bool) float64 {
	filtered := util.FilterResources(resources, func(res *mesos.Resource) bool {
		if withReservation {
			return res.GetName() == name && res.Reservation != nil
		}
		return res.GetName() == name && res.Reservation == nil
	})
	val := 0.0
	for _, res := range filtered {
		val += res.GetScalar().GetValue()
	}
	return val
}

func resourcesHaveVolume(resources []*mesos.Resource, persistenceId string) bool {
	filtered := util.FilterResources(resources, func(res *mesos.Resource) bool {
		return res.GetName() == "disk" &&
			res.Reservation != nil &&
			res.Disk != nil &&
			res.Disk.Persistence.GetId() == persistenceId
	})
	return len(filtered) > 0
}

func getUnreservedResources(resources []*mesos.Resource) (float64, float64, float64) {
	return getResource("cpus", resources, false), getResource("mem", resources, false), getResource("disk", resources, false)
}

func getReservedResources(resources []*mesos.Resource) (float64, float64, float64) {
	return getResource("cpus", resources, true), getResource("mem", resources, true), getResource("disk", resources, true)
}

func (sched *ExampleScheduler) Registered(driver sched.SchedulerDriver, frameworkId *mesos.FrameworkID, masterInfo *mesos.MasterInfo) {
	log.Infoln("Framework Registered with Master ", masterInfo)
}

func (sched *ExampleScheduler) Reregistered(driver sched.SchedulerDriver, masterInfo *mesos.MasterInfo) {
	log.Infoln("Framework Re-Registered with Master ", masterInfo)
}

func (sched *ExampleScheduler) Disconnected(sched.SchedulerDriver) {
	log.Fatalf("disconnected from master, aborting")
}

func (sched *ExampleScheduler) ResourceOffers(driver sched.SchedulerDriver, offers []*mesos.Offer) {
	for _, offer := range offers {
		operations := []*mesos.Offer_Operation{}
		resourcesToCreate := []*mesos.Resource{}
		resourcesToDestroy := []*mesos.Resource{}
		resourcesToReserve := []*mesos.Resource{}
		resourcesToUnreserve := []*mesos.Resource{}
		taskInfosToLaunch := []*mesos.TaskInfo{}
		totalUnreserved := 0

		unreservedCpus, unreservedMem, unreservedDisk := getUnreservedResources(offer.Resources)
		reservedCpus, reservedMem, reservedDisk := getReservedResources(offer.Resources)

		log.Infoln("Received Offer <", offer.Id.GetValue(), "> with unreserved cpus=", unreservedCpus, " mem=", unreservedMem, " disk=", unreservedDisk)
		log.Infoln("Received Offer <", offer.Id.GetValue(), "> with reserved cpus=", reservedCpus, " mem=", reservedMem, " disk=", reservedDisk)

		for _, task := range sched.tasks {
			switch task.state {
			case InitState:
				if CPUS_PER_TASK <= unreservedCpus &&
					MEM_PER_TASK <= unreservedMem &&
					DISK_PER_TASK <= unreservedDisk {

					resourcesToReserve = append(resourcesToReserve, []*mesos.Resource{
						util.NewScalarResourceWithReservation("cpus", CPUS_PER_TASK, *mesosAuthPrincipal, *role),
						util.NewScalarResourceWithReservation("mem", MEM_PER_TASK, *mesosAuthPrincipal, *role),
						util.NewScalarResourceWithReservation("disk", DISK_PER_TASK, *mesosAuthPrincipal, *role),
					}...)
					resourcesToCreate = append(resourcesToCreate,
						util.NewVolumeResourceWithReservation(DISK_PER_TASK, task.containerPath, task.persistenceId, mesos.Volume_RW.Enum(), *mesosAuthPrincipal, *role))
					task.state = RequestedReservationState
					unreservedCpus = unreservedCpus - CPUS_PER_TASK
					unreservedMem = unreservedMem - MEM_PER_TASK
					unreservedDisk = unreservedDisk - DISK_PER_TASK
				}
			case RequestedReservationState:
				if CPUS_PER_TASK <= reservedCpus &&
					MEM_PER_TASK <= reservedMem &&
					DISK_PER_TASK <= reservedDisk &&
					resourcesHaveVolume(offer.Resources, task.persistenceId) {

					taskId := &mesos.TaskID{
						Value: proto.String(task.name),
					}

					taskInfo := &mesos.TaskInfo{
						Name:     proto.String("go-task-" + taskId.GetValue()),
						TaskId:   taskId,
						SlaveId:  offer.SlaveId,
						Executor: task.executor,
						Resources: []*mesos.Resource{
							util.NewScalarResourceWithReservation("cpus", CPUS_PER_TASK, *mesosAuthPrincipal, *role),
							util.NewScalarResourceWithReservation("mem", MEM_PER_TASK, *mesosAuthPrincipal, *role),
							util.NewVolumeResourceWithReservation(DISK_PER_TASK, task.containerPath, task.persistenceId, mesos.Volume_RW.Enum(), *mesosAuthPrincipal, *role),
						},
					}

					taskInfosToLaunch = append(taskInfosToLaunch, taskInfo)
					task.state = LaunchedState
					reservedCpus = reservedCpus - CPUS_PER_TASK
					reservedMem = reservedMem - MEM_PER_TASK
					reservedDisk = reservedDisk - DISK_PER_TASK

					log.Infof("Prepared task: %s with offer %s for launch\n", taskInfo.GetName(), offer.Id.GetValue())
				}
			case FinishedState:
				resourcesToDestroy = append(resourcesToDestroy,
					util.NewVolumeResourceWithReservation(DISK_PER_TASK, task.containerPath, task.persistenceId, mesos.Volume_RW.Enum(), *mesosAuthPrincipal, *role))
				resourcesToUnreserve = append(resourcesToUnreserve, []*mesos.Resource{
					util.NewScalarResourceWithReservation("cpus", CPUS_PER_TASK, *mesosAuthPrincipal, *role),
					util.NewScalarResourceWithReservation("mem", MEM_PER_TASK, *mesosAuthPrincipal, *role),
					util.NewScalarResourceWithReservation("disk", DISK_PER_TASK, *mesosAuthPrincipal, *role),
				}...)
				task.state = UnreservedState
			case UnreservedState:
				totalUnreserved = totalUnreserved + 1
			}
		}

		// Clean up reservations we no longer need
		if len(resourcesToReserve) == 0 && len(resourcesToCreate) == 0 && len(taskInfosToLaunch) == 0 {
			if reservedCpus >= 0.0 {
				resourcesToUnreserve = append(resourcesToUnreserve, util.NewScalarResourceWithReservation("cpus", reservedCpus, *mesosAuthPrincipal, *role))
			}
			if reservedMem >= 0.0 {
				resourcesToUnreserve = append(resourcesToUnreserve, util.NewScalarResourceWithReservation("mem", reservedCpus, *mesosAuthPrincipal, *role))
			}
			if reservedDisk >= 0.0 {
				filtered := util.FilterResources(offer.Resources, func(res *mesos.Resource) bool {
					return res.GetName() == "disk" &&
						res.Reservation != nil &&
						res.Disk != nil
				})
				for _, volume := range filtered {
					resourcesToDestroy = append(resourcesToDestroy,
						util.NewVolumeResourceWithReservation(
							volume.GetScalar().GetValue(),
							volume.Disk.Volume.GetContainerPath(),
							volume.Disk.Persistence.GetId(),
							volume.Disk.Volume.Mode,
							*mesosAuthPrincipal,
							*role))
				}
				resourcesToUnreserve = append(resourcesToUnreserve, util.NewScalarResourceWithReservation("mem", reservedDisk, *mesosAuthPrincipal, *role))
			}
		}

		// Make a single operation per type
		if len(resourcesToReserve) > 0 {
			operations = append(operations, util.NewReserveOperation(resourcesToReserve))
		}
		if len(resourcesToCreate) > 0 {
			operations = append(operations, util.NewCreateOperation(resourcesToCreate))
		}
		if len(resourcesToUnreserve) > 0 {
			operations = append(operations, util.NewUnreserveOperation(resourcesToUnreserve))
		}
		if len(resourcesToDestroy) > 0 {
			operations = append(operations, util.NewDestroyOperation(resourcesToDestroy))
		}
		if len(taskInfosToLaunch) > 0 {
			operations = append(operations, util.NewLaunchOperation(taskInfosToLaunch))
		}

		log.Infoln("Accepting offers with ", len(operations), "operations for offer", offer.Id.GetValue())
		refuseSeconds := 5.0
		if len(operations) == 0 {
			refuseSeconds = 5.0
		}
		driver.AcceptOffers([]*mesos.OfferID{offer.Id}, operations, &mesos.Filters{RefuseSeconds: proto.Float64(refuseSeconds)})

		if totalUnreserved >= len(sched.tasks) {
			log.Infoln("Total tasks completed and unreserved, stopping framework.")
			driver.Stop(false)
		}
	}
}

func (sched *ExampleScheduler) StatusUpdate(driver sched.SchedulerDriver, status *mesos.TaskStatus) {
	log.Infoln("Status update: task", status.TaskId.GetValue(), " is in state ", status.State.Enum().String())
	for _, task := range sched.tasks {
		if task.name == status.TaskId.GetValue() &&
			(status.GetState() == mesos.TaskState_TASK_FINISHED ||
				status.GetState() == mesos.TaskState_TASK_LOST ||
				status.GetState() == mesos.TaskState_TASK_KILLED ||
				status.GetState() == mesos.TaskState_TASK_FAILED ||
				status.GetState() == mesos.TaskState_TASK_ERROR) {

			// No matter what the outcome was, move to finished state so that we can unreserve resources
			task.state = FinishedState
		}
	}

	if status.GetState() == mesos.TaskState_TASK_LOST ||
		status.GetState() == mesos.TaskState_TASK_KILLED ||
		status.GetState() == mesos.TaskState_TASK_FAILED ||
		status.GetState() == mesos.TaskState_TASK_ERROR {
		log.Infoln(
			"Task", status.TaskId.GetValue(),
			"is in unexpected state", status.State.String(),
			"with message", status.GetMessage(),
			". Unreserving resources",
		)
	}
}

func (sched *ExampleScheduler) OfferRescinded(_ sched.SchedulerDriver, oid *mesos.OfferID) {
	log.Errorf("offer rescinded: %v", oid)
}
func (sched *ExampleScheduler) FrameworkMessage(_ sched.SchedulerDriver, eid *mesos.ExecutorID, sid *mesos.SlaveID, msg string) {
	log.Errorf("framework message from executor %q slave %q: %q", eid, sid, msg)
}
func (sched *ExampleScheduler) SlaveLost(_ sched.SchedulerDriver, sid *mesos.SlaveID) {
	log.Errorf("slave lost: %v", sid)
}
func (sched *ExampleScheduler) ExecutorLost(_ sched.SchedulerDriver, eid *mesos.ExecutorID, sid *mesos.SlaveID, code int) {
	log.Errorf("executor %q lost on slave %q code %d", eid, sid, code)
}
func (sched *ExampleScheduler) Error(_ sched.SchedulerDriver, err string) {
	log.Errorf("Scheduler received error: %v", err)
}

// ----------------------- func init() ------------------------- //

func init() {
	flag.Parse()
	log.Infoln("Initializing the Example Scheduler...")
}

// returns (downloadURI, basename(path))
func serveExecutorArtifact(path string) (*string, string) {
	serveFile := func(pattern string, filename string) {
		http.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
			http.ServeFile(w, r, filename)
		})
	}

	// Create base path (http://foobar:5000/<base>)
	pathSplit := strings.Split(path, "/")
	var base string
	if len(pathSplit) > 0 {
		base = pathSplit[len(pathSplit)-1]
	} else {
		base = path
	}
	serveFile("/"+base, path)

	hostURI := fmt.Sprintf("http://%s:%d/%s", *address, *artifactPort, base)
	log.V(2).Infof("Hosting artifact '%s' at '%s'", path, hostURI)

	return &hostURI, base
}

func prepareExecutorInfo(id string) *mesos.ExecutorInfo {
	executorUris := []*mesos.CommandInfo_URI{}
	executorUris = append(executorUris, &mesos.CommandInfo_URI{Value: execUri, Executable: proto.Bool(true)})

	// forward the value of the scheduler's -v flag to the executor
	v := 0
	if f := flag.Lookup("v"); f != nil && f.Value != nil {
		if vstr := f.Value.String(); vstr != "" {
			if vi, err := strconv.ParseInt(vstr, 10, 32); err == nil {
				v = int(vi)
			}
		}
	}
	executorCommand := fmt.Sprintf("./%s -logtostderr=true -v=%d", execCmd, v)

	go http.ListenAndServe(fmt.Sprintf("%s:%d", *address, *artifactPort), nil)
	log.V(2).Info("Serving executor artifacts...")

	// Create mesos scheduler driver.
	return &mesos.ExecutorInfo{
		ExecutorId: util.NewExecutorID(id),
		Name:       proto.String("Test Executor (Go)"),
		Source:     proto.String("go_test"),
		Command: &mesos.CommandInfo{
			Value: proto.String(executorCommand),
			Uris:  executorUris,
		},
	}
}

func parseIP(address string) net.IP {
	addr, err := net.LookupIP(address)
	if err != nil {
		log.Fatal(err)
	}
	if len(addr) < 1 {
		log.Fatalf("failed to parse IP from address '%v'", address)
	}
	return addr[0]
}

// ----------------------- func main() ------------------------- //

func main() {
	execUri, execCmd = serveExecutorArtifact(*executorPath)

	// the framework
	fwinfo := &mesos.FrameworkInfo{
		User: proto.String(""), // Mesos-go will fill in user.
		Name: proto.String("Test Framework (Go)"),
		Role: proto.String(*role),
	}

	cred := (*mesos.Credential)(nil)
	if *mesosAuthPrincipal != "" {
		fwinfo.Principal = proto.String(*mesosAuthPrincipal)
		cred = &mesos.Credential{
			Principal: proto.String(*mesosAuthPrincipal),
		}
		if *mesosAuthSecretFile != "" {
			_, err := os.Stat(*mesosAuthSecretFile)
			if err != nil {
				log.Fatal("missing secret file: ", err.Error())
			}
			secret, err := ioutil.ReadFile(*mesosAuthSecretFile)
			if err != nil {
				log.Fatal("failed to read secret file: ", err.Error())
			}
			cred.Secret = proto.String(string(secret))
		}
	}
	bindingAddress := parseIP(*address)
	config := sched.DriverConfig{
		Scheduler:      newExampleScheduler(),
		Framework:      fwinfo,
		Master:         *master,
		Credential:     cred,
		BindingAddress: bindingAddress,
		WithAuthContext: func(ctx context.Context) context.Context {
			ctx = auth.WithLoginProvider(ctx, *authProvider)
			ctx = sasl.WithBindingAddress(ctx, bindingAddress)
			return ctx
		},
	}
	driver, err := sched.NewMesosSchedulerDriver(config)

	if err != nil {
		log.Errorln("Unable to create a SchedulerDriver ", err.Error())
	}

	if stat, err := driver.Run(); err != nil {
		log.Infof("Framework stopped with status %s and error: %s\n", stat.String(), err.Error())
	}
	log.Infof("framework terminating")
}
