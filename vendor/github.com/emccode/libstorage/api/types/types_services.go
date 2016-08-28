package types

// Service is the base type for services.
type Service interface {
	Driver
}

// Services is a service's container.
type Services interface {
	// Storage gets the storage service.
	Storage() StorageService

	// Tasks gets the task service.
	Tasks() TaskTrackingService
}

// TaskRunFunc is a function responsible for a task's execution.
type TaskRunFunc func(ctx Context) (interface{}, error)

// StorageTaskRunFunc is a function responsible for a storage-service task's
// execution.
type StorageTaskRunFunc func(
	ctx Context,
	service StorageService) (interface{}, error)

// StorageService is a service that provides the interaction with
// StorageDrivers.
type StorageService interface {
	Service

	// Driver returns the service's StorageDriver.
	Driver() StorageDriver

	// TaskExecute enqueues a task for execution.
	TaskExecute(
		ctx Context,
		run StorageTaskRunFunc,
		schema []byte) *Task
}

// TaskTrackingService a service for tracking tasks.
type TaskTrackingService interface {
	Service

	// Tasks returns a channel on which all tasks tracked via TrackTasks are
	// received.
	Tasks() <-chan *Task

	// TaskTrack creates a new, trackable task.
	TaskTrack(ctx Context) *Task

	// TaskExecute enqueues a task for execution.
	TaskExecute(
		ctx Context,
		run TaskRunFunc,
		schema []byte) *Task

	// TaskInspect returns the task with the specified ID.
	TaskInspect(taskID int) *Task

	// TaskWait blocks until the specified task completes.
	TaskWait(taskID int) <-chan int

	// TaskWaitAll blocks until all the specified tasks complete.
	TaskWaitAll(taskIDs ...int) <-chan int

	// TaskWaitC returns a channel that is closed when the specified task
	// completes.
	TaskWaitC(taskID int) <-chan int

	// TaskWaitAll returns a channel that is closed when the specified task
	// completes.
	TaskWaitAllC(taskIDs ...int) <-chan int
}

// TaskExecutionService is a service for executing tasks.
type TaskExecutionService interface {
	Service

	// TaskExecute enqueues a task for execution.
	TaskExecute(
		ctx Context,
		run TaskRunFunc,
		schema []byte) *Task
}
