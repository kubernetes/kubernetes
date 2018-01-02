package runtime

const (
	// TaskCreateEventTopic for task create
	TaskCreateEventTopic = "/tasks/create"
	// TaskStartEventTopic for task start
	TaskStartEventTopic = "/tasks/start"
	// TaskOOMEventTopic for task oom
	TaskOOMEventTopic = "/tasks/oom"
	// TaskExitEventTopic for task exit
	TaskExitEventTopic = "/tasks/exit"
	// TaskDeleteEventTopic for task delete
	TaskDeleteEventTopic = "/tasks/delete"
	// TaskExecAddedEventTopic for task exec create
	TaskExecAddedEventTopic = "/tasks/exec-added"
	// TaskExecStartedEventTopic for task exec start
	TaskExecStartedEventTopic = "/tasks/exec-started"
	// TaskPausedEventTopic for task pause
	TaskPausedEventTopic = "/tasks/paused"
	// TaskResumedEventTopic for task resume
	TaskResumedEventTopic = "/tasks/resumed"
	// TaskCheckpointedEventTopic for task checkpoint
	TaskCheckpointedEventTopic = "/tasks/checkpointed"
	// TaskUnknownTopic for unknown task events
	TaskUnknownTopic = "/tasks/?"
)
