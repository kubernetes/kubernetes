package specerror

import (
	"fmt"

	rfc2119 "github.com/opencontainers/runtime-tools/error"
)

// define error codes
const (
	// EntityOperSameContainer represents "The entity using a runtime to create a container MUST be able to use the operations defined in this specification against that same container."
	EntityOperSameContainer Code = 0xe001 + iota
	// StateIDUniq represents "`id` (string, REQUIRED) is the container's ID. This MUST be unique across all containers on this host."
	StateIDUniq
	// StateNewStatus represents "Additional values MAY be defined by the runtime, however, they MUST be used to represent new runtime states not defined above."
	StateNewStatus
	// DefaultStateJSONPattern represents "When serialized in JSON, the format MUST adhere to the default pattern."
	DefaultStateJSONPattern
	// EnvCreateImplement represents "The container's runtime environment MUST be created according to the configuration in `config.json`."
	EnvCreateImplement
	// EnvCreateError represents "If the runtime is unable to create the environment specified in the `config.json`, it MUST generate an error."
	EnvCreateError
	// ProcNotRunAtResRequest represents "While the resources requested in the `config.json` MUST be created, the user-specified program (from `process`) MUST NOT be run at this time."
	ProcNotRunAtResRequest
	// ConfigUpdatesWithoutAffect represents "Any updates to `config.json` after this step MUST NOT affect the container."
	ConfigUpdatesWithoutAffect
	// PrestartHooksInvoke represents "The prestart hooks MUST be invoked by the runtime."
	PrestartHooksInvoke
	// PrestartHookFailGenError represents "If any prestart hook fails, the runtime MUST generate an error, stop the container, and continue the lifecycle at step 9."
	PrestartHookFailGenError
	// ProcImplement represents "The runtime MUST run the user-specified program, as specified by `process`."
	ProcImplement
	// PoststartHooksInvoke represents "The poststart hooks MUST be invoked by the runtime."
	PoststartHooksInvoke
	// PoststartHookFailGenWarn represents "If any poststart hook fails, the runtime MUST log a warning, but the remaining hooks and lifecycle continue as if the hook had succeeded."
	PoststartHookFailGenWarn
	// UndoCreateSteps represents "The container MUST be destroyed by undoing the steps performed during create phase (step 2)."
	UndoCreateSteps
	// PoststopHooksInvoke represents "The poststop hooks MUST be invoked by the runtime."
	PoststopHooksInvoke
	// PoststopHookFailGenWarn represents "If any poststop hook fails, the runtime MUST log a warning, but the remaining hooks and lifecycle continue as if the hook had succeeded."
	PoststopHookFailGenWarn
	// ErrorsLeaveStateUnchange represents "Unless otherwise stated, generating an error MUST leave the state of the environment as if the operation were never attempted - modulo any possible trivial ancillary changes such as logging."
	ErrorsLeaveStateUnchange
	// WarnsLeaveFlowUnchange represents "Unless otherwise stated, logging a warning does not change the flow of the operation; it MUST continue as if the warning had not been logged."
	WarnsLeaveFlowUnchange
	// DefaultOperations represents "Unless otherwise stated, runtimes MUST support the default operations."
	DefaultOperations
	// QueryWithoutIDGenError represents "This operation MUST generate an error if it is not provided the ID of a container."
	QueryWithoutIDGenError
	// QueryNonExistGenError represents "Attempting to query a container that does not exist MUST generate an error."
	QueryNonExistGenError
	// QueryStateImplement represents "This operation MUST return the state of a container as specified in the State section."
	QueryStateImplement
	// CreateWithBundlePathAndID represents "This operation MUST generate an error if it is not provided a path to the bundle and the container ID to associate with the container."
	CreateWithBundlePathAndID
	// CreateWithUniqueID represents "If the ID provided is not unique across all containers within the scope of the runtime, or is not valid in any other way, the implementation MUST generate an error and a new container MUST NOT be created."
	CreateWithUniqueID
	// CreateNewContainer represents "This operation MUST create a new container."
	CreateNewContainer
	// PropsApplyExceptProcOnCreate represents "All of the properties configured in `config.json` except for `process` MUST be applied."
	PropsApplyExceptProcOnCreate
	// ProcArgsApplyUntilStart represents `process.args` MUST NOT be applied until triggered by the `start` operation."
	ProcArgsApplyUntilStart
	// PropApplyFailGenError represents "If the runtime cannot apply a property as specified in the configuration, it MUST generate an error."
	PropApplyFailGenError
	// PropApplyFailNotCreate represents "If the runtime cannot apply a property as specified in the configuration, a new container MUST NOT be created."
	PropApplyFailNotCreate
	// StartWithoutIDGenError represents "`start` operation MUST generate an error if it is not provided the container ID."
	StartWithoutIDGenError
	// StartNotCreatedHaveNoEffect represents "Attempting to `start` a container that is not `created` MUST have no effect on the container."
	StartNotCreatedHaveNoEffect
	// StartNotCreatedGenError represents "Attempting to `start` a container that is not `created` MUST generate an error."
	StartNotCreatedGenError
	// StartProcImplement represents "`start` operation MUST run the user-specified program as specified by `process`."
	StartProcImplement
	// StartWithProcUnsetGenError represents "`start` operation MUST generate an error if `process` was not set."
	StartWithProcUnsetGenError
	// KillWithoutIDGenError represents "`kill` operation MUST generate an error if it is not provided the container ID."
	KillWithoutIDGenError
	// KillNonCreateRunHaveNoEffect represents "Attempting to send a signal to a container that is neither `created` nor `running` MUST have no effect on the container."
	KillNonCreateRunHaveNoEffect
	// KillNonCreateRunGenError represents "Attempting to send a signal to a container that is neither `created` nor `running` MUST generate an error."
	KillNonCreateRunGenError
	// KillSignalImplement represents "`kill` operation MUST send the specified signal to the container process."
	KillSignalImplement
	// DeleteWithoutIDGenError represents "`delete` operation MUST generate an error if it is not provided the container ID."
	DeleteWithoutIDGenError
	// DeleteNonStopHaveNoEffect represents "Attempting to `delete` a container that is not `stopped` MUST have no effect on the container."
	DeleteNonStopHaveNoEffect
	// DeleteNonStopGenError represents "Attempting to `delete` a container that is not `stopped` MUST generate an error."
	DeleteNonStopGenError
	// DeleteResImplement represents "Deleting a container MUST delete the resources that were created during the `create` step."
	DeleteResImplement
	// DeleteOnlyCreatedRes represents "Note that resources associated with the container, but not created by this container, MUST NOT be deleted."
	DeleteOnlyCreatedRes
)

var (
	scopeOfAContainerRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#scope-of-a-container"), nil
	}
	stateRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#state"), nil
	}
	lifecycleRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#lifecycle"), nil
	}
	errorsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#errors"), nil
	}
	warningsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#warnings"), nil
	}
	operationsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#operations"), nil
	}
	queryStateRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#query-state"), nil
	}
	createRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#create"), nil
	}
	startRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#start"), nil
	}
	killRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#kill"), nil
	}
	deleteRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "runtime.md#delete"), nil
	}
)

func init() {
	register(EntityOperSameContainer, rfc2119.Must, scopeOfAContainerRef)
	register(StateIDUniq, rfc2119.Must, stateRef)
	register(StateNewStatus, rfc2119.Must, stateRef)
	register(DefaultStateJSONPattern, rfc2119.Must, stateRef)
	register(EnvCreateImplement, rfc2119.Must, lifecycleRef)
	register(EnvCreateError, rfc2119.Must, lifecycleRef)
	register(ProcNotRunAtResRequest, rfc2119.Must, lifecycleRef)
	register(ConfigUpdatesWithoutAffect, rfc2119.Must, lifecycleRef)
	register(PrestartHooksInvoke, rfc2119.Must, lifecycleRef)
	register(PrestartHookFailGenError, rfc2119.Must, lifecycleRef)
	register(ProcImplement, rfc2119.Must, lifecycleRef)
	register(PoststartHooksInvoke, rfc2119.Must, lifecycleRef)
	register(PoststartHookFailGenWarn, rfc2119.Must, lifecycleRef)
	register(UndoCreateSteps, rfc2119.Must, lifecycleRef)
	register(PoststopHooksInvoke, rfc2119.Must, lifecycleRef)
	register(PoststopHookFailGenWarn, rfc2119.Must, lifecycleRef)
	register(ErrorsLeaveStateUnchange, rfc2119.Must, errorsRef)
	register(WarnsLeaveFlowUnchange, rfc2119.Must, warningsRef)
	register(DefaultOperations, rfc2119.Must, operationsRef)
	register(QueryWithoutIDGenError, rfc2119.Must, queryStateRef)
	register(QueryNonExistGenError, rfc2119.Must, queryStateRef)
	register(QueryStateImplement, rfc2119.Must, queryStateRef)
	register(CreateWithBundlePathAndID, rfc2119.Must, createRef)
	register(CreateWithUniqueID, rfc2119.Must, createRef)
	register(CreateNewContainer, rfc2119.Must, createRef)
	register(PropsApplyExceptProcOnCreate, rfc2119.Must, createRef)
	register(ProcArgsApplyUntilStart, rfc2119.Must, createRef)
	register(PropApplyFailGenError, rfc2119.Must, createRef)
	register(PropApplyFailNotCreate, rfc2119.Must, createRef)
	register(StartWithoutIDGenError, rfc2119.Must, startRef)
	register(StartNotCreatedHaveNoEffect, rfc2119.Must, startRef)
	register(StartNotCreatedGenError, rfc2119.Must, startRef)
	register(StartProcImplement, rfc2119.Must, startRef)
	register(StartWithProcUnsetGenError, rfc2119.Must, startRef)
	register(KillWithoutIDGenError, rfc2119.Must, killRef)
	register(KillNonCreateRunHaveNoEffect, rfc2119.Must, killRef)
	register(KillNonCreateRunGenError, rfc2119.Must, killRef)
	register(KillSignalImplement, rfc2119.Must, killRef)
	register(DeleteWithoutIDGenError, rfc2119.Must, deleteRef)
	register(DeleteNonStopHaveNoEffect, rfc2119.Must, deleteRef)
	register(DeleteNonStopGenError, rfc2119.Must, deleteRef)
	register(DeleteResImplement, rfc2119.Must, deleteRef)
	register(DeleteOnlyCreatedRes, rfc2119.Must, deleteRef)
}
