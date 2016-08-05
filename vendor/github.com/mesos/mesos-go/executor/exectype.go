package executor

import (
	"github.com/mesos/mesos-go/mesosproto"
)

/**
 * Executor callback interface to be implemented by frameworks' executors. Note
 * that only one callback will be invoked at a time, so it is not
 * recommended that you block within a callback because it may cause a
 * deadlock.
 *
 * Each callback includes an instance to the executor driver that was
 * used to run this executor. The driver will not change for the
 * duration of an executor (i.e., from the point you do
 * ExecutorDriver.Start() to the point that ExecutorDriver.Join()
 * returns). This is intended for convenience so that an executor
 * doesn't need to store a pointer to the driver itself.
 */
type Executor interface {
	/**
	 * Invoked once the executor driver has been able to successfully
	 * connect with Mesos. In particular, a scheduler can pass some
	 * data to its executors through the FrameworkInfo.ExecutorInfo's
	 * data field.
	 */
	Registered(ExecutorDriver, *mesosproto.ExecutorInfo, *mesosproto.FrameworkInfo, *mesosproto.SlaveInfo)

	/**
	 * Invoked when the executor re-registers with a restarted slave.
	 */
	Reregistered(ExecutorDriver, *mesosproto.SlaveInfo)

	/**
	 * Invoked when the executor becomes "disconnected" from the slave
	 * (e.g., the slave is being restarted due to an upgrade).
	 */
	Disconnected(ExecutorDriver)

	/**
	 * Invoked when a task has been launched on this executor (initiated
	 * via SchedulerDriver.LaunchTasks). Note that this task can be realized
	 * with a goroutine, an external process, or some simple computation, however,
	 * no other callbacks will be invoked on this executor until this
	 * callback has returned.
	 */
	LaunchTask(ExecutorDriver, *mesosproto.TaskInfo)

	/**
	 * Invoked when a task running within this executor has been killed
	 * (via SchedulerDriver.KillTask). Note that no status update will
	 * be sent on behalf of the executor, the executor is responsible
	 * for creating a new TaskStatus (i.e., with TASK_KILLED) and
	 * invoking ExecutorDriver.SendStatusUpdate.
	 */
	KillTask(ExecutorDriver, *mesosproto.TaskID)

	/**
	 * Invoked when a framework message has arrived for this
	 * executor. These messages are best effort; do not expect a
	 * framework message to be retransmitted in any reliable fashion.
	 */
	FrameworkMessage(ExecutorDriver, string)

	/**
	 * Invoked when the executor should terminate all of its currently
	 * running tasks. Note that after Mesos has determined that an
	 * executor has terminated, any tasks that the executor did not send
	 * terminal status updates for (e.g., TASK_KILLED, TASK_FINISHED,
	 * TASK_FAILED, etc) a TASK_LOST status update will be created.
	 */
	Shutdown(ExecutorDriver)

	/**
	 * Invoked when a fatal error has occured with the executor and/or
	 * executor driver. The driver will be aborted BEFORE invoking this
	 * callback.
	 */
	Error(ExecutorDriver, string)
}

/**
 * ExecutorDriver interface for connecting an executor to Mesos. This
 * interface is used both to manage the executor's lifecycle (start
 * it, stop it, or wait for it to finish) and to interact with Mesos
 * (e.g., send status updates, send framework messages, etc.).
 * A driver method is expected to fail-fast and return an error when possible.
 * Other internal errors (or remote error) that occur asynchronously are handled
 * using the the Executor.Error() callback.
 */
type ExecutorDriver interface {
	/**
	 * Starts the executor driver. This needs to be called before any
	 * other driver calls are made.
	 */
	Start() (mesosproto.Status, error)

	/**
	 * Stops the executor driver.
	 */
	Stop() (mesosproto.Status, error)

	/**
	 * Aborts the driver so that no more callbacks can be made to the
	 * executor. The semantics of abort and stop have deliberately been
	 * separated so that code can detect an aborted driver (i.e., via
	 * the return status of ExecutorDriver.Join, see below), and
	 * instantiate and start another driver if desired (from within the
	 * same process ... although this functionality is currently not
	 * supported for executors).
	 */
	Abort() (mesosproto.Status, error)

	/**
	 * Waits for the driver to be stopped or aborted, possibly
	 * blocking the calling goroutine indefinitely. The return status of
	 * this function can be used to determine if the driver was aborted
	 * (see package mesosproto for a description of Status).
	 */
	Join() (mesosproto.Status, error)

	/**
	 * Starts and immediately joins (i.e., blocks on) the driver.
	 */
	Run() (mesosproto.Status, error)

	/**
	 * Sends a status update to the framework scheduler, retrying as
	 * necessary until an acknowledgement has been received or the
	 * executor is terminated (in which case, a TASK_LOST status update
	 * will be sent). See Scheduler.StatusUpdate for more information
	 * about status update acknowledgements.
	 */
	SendStatusUpdate(*mesosproto.TaskStatus) (mesosproto.Status, error)

	/**
	 * Sends a message to the framework scheduler. These messages are
	 * best effort; do not expect a framework message to be
	 * retransmitted in any reliable fashion.
	 */
	SendFrameworkMessage(string) (mesosproto.Status, error)
}
