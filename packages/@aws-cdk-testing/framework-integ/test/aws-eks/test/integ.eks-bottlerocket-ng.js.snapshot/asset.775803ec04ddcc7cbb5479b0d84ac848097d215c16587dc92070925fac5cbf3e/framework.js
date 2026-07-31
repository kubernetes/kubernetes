"use strict";
/* eslint-disable @cdklabs/no-throw-default-error */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
/* eslint-disable max-len */
const cfnResponse = __importStar(require("./cfn-response"));
const consts = __importStar(require("./consts"));
const outbound_1 = require("./outbound");
const util_1 = require("./util");
/**
 * The main runtime entrypoint of the async custom resource lambda function.
 *
 * Any lifecycle event changes to the custom resources will invoke this handler, which will, in turn,
 * interact with the user-defined `onEvent` and `isComplete` handlers.
 *
 * This function will always succeed. If an error occurs, it is logged but an error is not thrown.
 *
 * @param cfnRequest The cloudformation custom resource event.
 */
async function onEvent(cfnRequest) {
    const sanitizedRequest = { ...cfnRequest, ResponseURL: '...' };
    (0, util_1.log)('onEventHandler', sanitizedRequest);
    cfnRequest.ResourceProperties = cfnRequest.ResourceProperties || {};
    const onEventResult = await invokeUserFunction(consts.USER_ON_EVENT_FUNCTION_ARN_ENV, sanitizedRequest, cfnRequest.ResponseURL);
    if (onEventResult?.NoEcho) {
        (0, util_1.log)('redacted onEvent returned:', cfnResponse.redactDataFromPayload(onEventResult));
    }
    else {
        (0, util_1.log)('onEvent returned:', onEventResult);
    }
    // merge the request and the result from onEvent to form the complete resource event
    // this also performs validation.
    const resourceEvent = createResponseEvent(cfnRequest, onEventResult);
    const sanitizedEvent = { ...resourceEvent, ResponseURL: '...' };
    if (onEventResult?.NoEcho) {
        (0, util_1.log)('readacted event:', cfnResponse.redactDataFromPayload(sanitizedEvent));
    }
    else {
        (0, util_1.log)('event:', sanitizedEvent);
    }
    // determine if this is an async provider based on whether we have an isComplete handler defined.
    // if it is not defined, then we are basically ready to return a positive response.
    if (!process.env[consts.USER_IS_COMPLETE_FUNCTION_ARN_ENV]) {
        return cfnResponse.submitResponse('SUCCESS', resourceEvent, { noEcho: resourceEvent.NoEcho });
    }
    // ok, we are not complete, so kick off the waiter workflow
    const waiter = {
        stateMachineArn: (0, util_1.getEnv)(consts.WAITER_STATE_MACHINE_ARN_ENV),
        input: JSON.stringify(resourceEvent),
    };
    (0, util_1.log)('starting waiter', {
        stateMachineArn: (0, util_1.getEnv)(consts.WAITER_STATE_MACHINE_ARN_ENV),
    });
    // kick off waiter state machine
    await (0, outbound_1.startExecution)(waiter);
}
// invoked a few times until `complete` is true or until it times out.
async function isComplete(event) {
    const sanitizedRequest = { ...event, ResponseURL: '...' };
    if (event?.NoEcho) {
        (0, util_1.log)('redacted isComplete request', cfnResponse.redactDataFromPayload(sanitizedRequest));
    }
    else {
        (0, util_1.log)('isComplete', sanitizedRequest);
    }
    const isCompleteResult = await invokeUserFunction(consts.USER_IS_COMPLETE_FUNCTION_ARN_ENV, sanitizedRequest, event.ResponseURL);
    if (event?.NoEcho) {
        (0, util_1.log)('redacted user isComplete returned:', cfnResponse.redactDataFromPayload(isCompleteResult));
    }
    else {
        (0, util_1.log)('user isComplete returned:', isCompleteResult);
    }
    // if we are not complete, return false, and don't send a response back.
    if (!isCompleteResult.IsComplete) {
        if (isCompleteResult.Data && Object.keys(isCompleteResult.Data).length > 0) {
            throw new Error('"Data" is not allowed if "IsComplete" is "False"');
        }
        // This must be the full event, it will be deserialized in `onTimeout` to send the response to CloudFormation
        throw new cfnResponse.Retry(JSON.stringify(event));
    }
    const response = {
        ...event,
        ...isCompleteResult,
        Data: {
            ...event.Data,
            ...isCompleteResult.Data,
        },
    };
    await cfnResponse.submitResponse('SUCCESS', response, { noEcho: event.NoEcho });
}
// invoked when completion retries are exhaused.
async function onTimeout(timeoutEvent) {
    (0, util_1.log)('timeoutHandler', timeoutEvent);
    const isCompleteRequest = JSON.parse(JSON.parse(timeoutEvent.Cause).errorMessage);
    await cfnResponse.submitResponse('FAILED', isCompleteRequest, {
        reason: 'Operation timed out',
    });
}
async function invokeUserFunction(functionArnEnv, sanitizedPayload, responseUrl) {
    const functionArn = (0, util_1.getEnv)(functionArnEnv);
    (0, util_1.log)(`executing user function ${functionArn} with payload`, sanitizedPayload);
    // transient errors such as timeouts, throttling errors (429), and other
    // errors that aren't caused by a bad request (500 series) are retried
    // automatically by the JavaScript SDK.
    const resp = await (0, outbound_1.invokeFunction)({
        FunctionName: functionArn,
        // Cannot strip 'ResponseURL' here as this would be a breaking change even though the downstream CR doesn't need it
        Payload: JSON.stringify({ ...sanitizedPayload, ResponseURL: responseUrl }),
    });
    (0, util_1.log)('user function response:', resp, typeof (resp));
    // ParseJsonPayload is very defensive. It should not be possible for `Payload`
    // to be anything other than a JSON encoded string (or intarray). Something weird is
    // going on if that happens. Still, we should do our best to survive it.
    const jsonPayload = (0, util_1.parseJsonPayload)(resp.Payload);
    if (resp.FunctionError) {
        (0, util_1.log)('user function threw an error:', resp.FunctionError);
        const errorMessage = jsonPayload.errorMessage || 'error';
        // parse function name from arn
        // arn:${Partition}:lambda:${Region}:${Account}:function:${FunctionName}
        const arn = functionArn.split(':');
        const functionName = arn[arn.length - 1];
        // append a reference to the log group.
        const message = [
            errorMessage,
            '',
            `Logs: /aws/lambda/${functionName}`, // cloudwatch log group
            '',
        ].join('\n');
        const e = new Error(message);
        // the output that goes to CFN is what's in `stack`, not the error message.
        // if we have a remote trace, construct a nice message with log group information
        if (jsonPayload.trace) {
            // skip first trace line because it's the message
            e.stack = [message, ...jsonPayload.trace.slice(1)].join('\n');
        }
        throw e;
    }
    return jsonPayload;
}
function createResponseEvent(cfnRequest, onEventResult) {
    //
    // validate that onEventResult always includes a PhysicalResourceId
    onEventResult = onEventResult || {};
    // if physical ID is not returned, we have some defaults for you based
    // on the request type.
    const physicalResourceId = onEventResult.PhysicalResourceId || defaultPhysicalResourceId(cfnRequest);
    // if we are in DELETE and physical ID was changed, it's an error.
    if (cfnRequest.RequestType === 'Delete' && physicalResourceId !== cfnRequest.PhysicalResourceId) {
        throw new Error(`DELETE: cannot change the physical resource ID from "${cfnRequest.PhysicalResourceId}" to "${onEventResult.PhysicalResourceId}" during deletion`);
    }
    // if we are in UPDATE and physical ID was changed, it's a replacement (just log)
    if (cfnRequest.RequestType === 'Update' && physicalResourceId !== cfnRequest.PhysicalResourceId) {
        (0, util_1.log)(`UPDATE: changing physical resource ID from "${cfnRequest.PhysicalResourceId}" to "${onEventResult.PhysicalResourceId}"`);
    }
    // merge request event and result event (result prevails).
    return {
        ...cfnRequest,
        ...onEventResult,
        PhysicalResourceId: physicalResourceId,
    };
}
/**
 * Calculates the default physical resource ID based in case user handler did
 * not return a PhysicalResourceId.
 *
 * For "CREATE", it uses the RequestId.
 * For "UPDATE" and "DELETE" and returns the current PhysicalResourceId (the one provided in `event`).
 */
function defaultPhysicalResourceId(req) {
    switch (req.RequestType) {
        case 'Create':
            return req.RequestId;
        case 'Update':
        case 'Delete':
            return req.PhysicalResourceId;
        default:
            throw new Error(`Invalid "RequestType" in request "${JSON.stringify(req)}"`);
    }
}
module.exports = {
    [consts.FRAMEWORK_ON_EVENT_HANDLER_NAME]: cfnResponse.safeHandler(onEvent),
    [consts.FRAMEWORK_IS_COMPLETE_HANDLER_NAME]: cfnResponse.safeHandler(isComplete),
    [consts.FRAMEWORK_ON_TIMEOUT_HANDLER_NAME]: onTimeout,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZnJhbWV3b3JrLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiZnJhbWV3b3JrLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQSxvREFBb0Q7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFFcEQsNEJBQTRCO0FBRTVCLDREQUE4QztBQUM5QyxpREFBbUM7QUFDbkMseUNBQTREO0FBQzVELGlDQUF1RDtBQVV2RDs7Ozs7Ozs7O0dBU0c7QUFDSCxLQUFLLFVBQVUsT0FBTyxDQUFDLFVBQXVEO0lBQzVFLE1BQU0sZ0JBQWdCLEdBQUcsRUFBRSxHQUFHLFVBQVUsRUFBRSxXQUFXLEVBQUUsS0FBSyxFQUFXLENBQUM7SUFDeEUsSUFBQSxVQUFHLEVBQUMsZ0JBQWdCLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztJQUV4QyxVQUFVLENBQUMsa0JBQWtCLEdBQUcsVUFBVSxDQUFDLGtCQUFrQixJQUFJLEVBQUcsQ0FBQztJQUVyRSxNQUFNLGFBQWEsR0FBRyxNQUFNLGtCQUFrQixDQUFDLE1BQU0sQ0FBQyw4QkFBOEIsRUFBRSxnQkFBZ0IsRUFBRSxVQUFVLENBQUMsV0FBVyxDQUFvQixDQUFDO0lBQ25KLElBQUksYUFBYSxFQUFFLE1BQU0sRUFBRSxDQUFDO1FBQzFCLElBQUEsVUFBRyxFQUFDLDRCQUE0QixFQUFFLFdBQVcsQ0FBQyxxQkFBcUIsQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDO0lBQ3RGLENBQUM7U0FBTSxDQUFDO1FBQ04sSUFBQSxVQUFHLEVBQUMsbUJBQW1CLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFDMUMsQ0FBQztJQUVELG9GQUFvRjtJQUNwRixpQ0FBaUM7SUFDakMsTUFBTSxhQUFhLEdBQUcsbUJBQW1CLENBQUMsVUFBVSxFQUFFLGFBQWEsQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sY0FBYyxHQUFHLEVBQUUsR0FBRyxhQUFhLEVBQUUsV0FBVyxFQUFFLEtBQUssRUFBRSxDQUFDO0lBQ2hFLElBQUksYUFBYSxFQUFFLE1B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LENBQUMsRUFBRSxXQUFXLENBQUMsV0FBVyxDQUFDLFVBQVUsQ0FBQztJQUNoRixDQUFDLE1BQU0sQ0FBQyxpQ0FBaUMsQ0FBQyxFQUFFLFNBQVM7Q0FDdEQsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qIGVzbGludC1kaXNhYmxlIEBjZGtsYWJzL25vLXRocm93LWRlZmF1bHQtZXJyb3IgKi9cblxuLyogZXNsaW50LWRpc2FibGUgbWF4LWxlbiAqL1xuXG5pbXBvcnQgKiBhcyBjZm5SZXNwb25zZSBmcm9tICcuL2Nmbi1yZXNwb25zZSc7XG5pbXBvcnQgKiBhcyBjb25zdHMgZnJvbSAnLi9jb25zdHMnO1xuaW1wb3J0IHsgaW52b2tlRnVuY3Rpb24sIHN0YXJ0RXhlY3V0aW9uIH0gZnJvbSAnLi9vdXRib3VuZCc7XG5pbXBvcnQgeyBnZXRFbnYsIGxvZywgcGFyc2VKc29uUGF5bG9hZCB9IGZyb20gJy4vdXRpbCc7XG5pbXBvcnQgdHlwZSB7IElzQ29tcGxldGVSZXNwb25zZSwgT25FdmVudFJlc3BvbnNlIH0gZnJvbSAnLi4vdHlwZXMnO1xuXG4vLyB1c2UgY29uc3RzIGZvciBoYW5kbGVyIG5hbWVzIHRvIGNvbXBpbGVyLWVuZm9yY2UgdGhlIGNvdXBsaW5nIHdpdGggY29uc3RydWN0aW9uIGNvZGUuXG5leHBvcnQgPSB7XG4gIFtjb25zdHMuRlJBTUVXT1JLX09OX0VWRU5UX0hBTkRMRVJfTkFNRV06IGNmblJlc3BvbnNlLnNhZmVIYW5kbGVyKG9uRXZlbnQpLFxuICBbY29uc3RzLkZSQU1FV09SS19JU19DT01QTEVURV9IQU5ETEVSX05BTUVdOiBjZm5SZXNwb25zZS5zYWZlSGFuZGxlcihpc0NvbXBsZXRlKSxcbiAgW2NvbnN0cy5GUkFNRVdPUktfT05fVElNRU9VVF9IQU5ETEVSX05BTUVdOiBvblRpbWVvdXQsXG59O1xuXG4vKipcbiAqIFRoZSBtYWluIHJ1bnRpbWUgZW50cnlwb2ludCBvZiB0aGUgYXN5bmMgY3VzdG9tIHJlc291cmNlIGxhbWJkYSBmdW5jdGlvbi5cbiAqXG4gKiBBbnkgbGlmZWN5Y2xlIGV2ZW50IGNoYW5nZXMgdG8gdGhlIGN1c3RvbSByZXNvdXJjZXMgd2lsbCBpbnZva2UgdGhpcyBoYW5kbGVyLCB3aGljaCB3aWxsLCBpbiB0dXJuLFxuICogaW50ZXJhY3Qgd2l0aCB0aGUgdXNlci1kZWZpbmVkIGBvbkV2ZW50YCBhbmQgYGlzQ29tcGxldGVgIGhhbmRsZXJzLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gd2lsbCBhbHdheXMgc3VjY2VlZC4gSWYgYW4gZXJyb3Igb2NjdXJzLCBpdCBpcyBsb2dnZWQgYnV0IGFuIGVycm9yIGlzIG5vdCB0aHJvd24uXG4gKlxuICogQHBhcmFtIGNmblJlcXVlc3QgVGhlIGNsb3VkZm9ybWF0aW9uIGN1c3RvbSByZXNvdXJjZSBldmVudC5cbiAqL1xuYXN5bmMgZnVuY3Rpb24gb25FdmVudChjZm5SZXF1ZXN0OiBBV1NMYW1iZGEuQ2xvdWRGb3JtYXRpb25DdXN0b21SZXNvdXJjZUV2ZW50KSB7XG4gIGNvbnN0IHNhbml0aXplZFJlcXVlc3QgPSB7IC4uLmNmblJlcXVlc3QsIFJlc3BvbnNlVVJMOiAnLi4uJyB9IGFzIGNvbnN0O1xuICBsb2coJ29uRXZlbnRIYW5kbGVyJywgc2FuaXRpemVkUmVxdWVzdCk7XG5cbiAgY2ZuUmVxdWVzdC5SZXNvdXJjZVByb3BlcnRpZXMgPSBjZm5SZXF1ZXN0LlJlc291cmNlUHJvcGVydGllcyB8fCB7IH07XG5cbiAgY29uc3Qgb25FdmVudFJlc3VsdCA9IGF3YWl0IGludm9rZVVzZXJGdW5jdGlvbihjb25zdHMuVVNFUl9PTl9FVkVOVF9GVU5DVElPTl9BUk5fRU5WLCBzYW5pdGl6ZWRSZXF1ZXN0LCBjZm5SZXF1ZXN0LlJlc3BvbnNlVVJMKSBhcyBPbkV2ZW50UmVzcG9uc2U7XG4gIGlmIChvbkV2ZW50UmVzdWx0Py5Ob0VjaG8pIHtcbiAgICBsb2coJ3JlZGFjdGVkIG9uRXZlbnQgcmV0dXJuZWQ6JywgY2ZuUmVzcG9uc2UucmVkYWN0RGF0YUZyb21QYXlsb2FkKG9uRXZlbnRSZXN1bHQpKTtcbiAgfSBlbHNlIHtcbiAgICBsb2coJ29uRXZlbnQgcmV0dXJuZWQ6Jywgb25FdmVudFJlc3VsdCk7XG4gIH1cblxuICAvLyBtZXJnZSB0aGUgcmVxdWVzdCBhbmQgdGhlIHJlc3VsdCBmcm9tIG9uRXZlbnQgdG8gZm9ybSB0aGUgY29tcGxldGUgcmVzb3VyY2UgZXZlbnRcbiAgLy8gdGhpcyBhbHNvIHBlcmZvcm1zIHZhbGlkYXRpb24uXG4gIGNvbnN0IHJlc291cmNlRXZlbnQgPSBjcmVhdGVSZXNwb25zZUV2ZW50KGNmblJlcXVlc3QsIG9uRXZlbnRSZXN1bHQpO1xuICBjb25zdCBzYW5pdGl6ZWRFdmVudCA9IHsgLi4ucmVzb3VyY2VFdmVudCwgUmVzcG9uc2VVUkw6ICcuLi4nIH07XG4gIGlmIChvbkV2ZW50UmVzdWx0Py5Ob0VjaG8pIHtcbiAgICBsb2coJ3JlYWRhY3RlZCBldmVudDonLCBjZm5SZXNwb25zZS5yZWRhY3REYXRhRnJvbVBheWxvYWQoc2FuaXRpemVkRXZlbnQpKTtcbiAgfSBlbHNlIHtcbiAgICBsb2coJ2V2ZW50OicsIHNhbml0aXplZEV2ZW50KTtcbiAgfVxuXG4gIC8vIGRldGVybWluZSBpZiB0aGlzIGlzIGFuIGFzeW5jIHByb3ZpZGVyIGJhc2VkIG9uIHdoZXRoZXIgd2UgaGF2ZSBhbiBpc0NvbXBsZXRlIGhhbmRsZXIgZGVmaW5lZC5cbiAgLy8gaWYgaXQgaXMgbm90IGRlZmluZWQsIHRoZW4gd2UgYXJlIGJhc2ljYWxseSByZWFkeSB0byByZXR1cm4gYSBwb3NpdGl2ZSByZXNwb25zZS5cbiAgaWYgKCFwcm9jZXNzLmVudltjb25zdHMuVVNFUl9JU19DT01QTEVURV9GVU5DVElPTl9BUk5fRU5WXSkge1xuICAgIHJldHVybiBjZm5SZXNwb25zZS5zdWJtaXRSZXNwb25zZSgnU1VDQ0VTUycsIHJlc291cmNlRXZlbnQsIHsgbm9FY2hvOiByZXNvdXJjZUV2ZW50Lk5vRWNobyB9KTtcbiAgfVxuXG4gIC8vIG9rLCB3ZSBhcmUgbm90IGNvbXBsZXRlLCBzbyBraWNrIG9mZiB0aGUgd2FpdGVyIHdvcmtmbG93XG4gIGNvbnN0IHdhaXRlciA9IHtcbiAgICBzdGF0ZU1hY2hpbmVBcm46IGdldEVudihjb25zdHMuV0FJVEVSX1NUQVRFX01BQ0hJTkVfQVJOX0VOViksXG4gICAgaW5wdXQ6IEpTT04uc3RyaW5naWZ5KHJlc291cmNlRXZlbnQpLFxuICB9O1xuXG4gIGxvZygnc3RhcnRpbmcgd2FpdGVyJywge1xuICAgIHN0YXRlTWFjaGluZUFybjogZ2V0RW52KGNvbnN0cy5XQUlURVJfU1RBVEVfTUFDSElORV9BUk5fRU5WKSxcbiAgfSk7XG5cbiAgLy8ga2ljayBvZmYgd2FpdGVyIHN0YXRlIG1hY2hpbmVcbiAgYXdhaXQgc3RhcnRFeGVjdXRpb24od2FpdGVyKTtcbn1cblxuLy8gaW52b2tlZCBhIGZldyB0aW1lcyB1bnRpbCBgY29tcGxldGVgIGlzIHRydWUgb3IgdW50aWwgaXQgdGltZXMgb3V0LlxuYXN5bmMgZnVuY3Rpb24gaXNDb21wbGV0ZShldmVudDogQVdTQ0RLQXN5bmNDdXN0b21SZXNvdXJjZS5Jc0NvbXBsZXRlUmVxdWVzdCkge1xuICBjb25zdCBzYW5pdGl6ZWRSZXF1ZXN0ID0geyAuLi5ldmVudCwgUmVzcG9uc2VVUkw6ICcuLi4nIH0gYXMgY29uc3Q7XG4gIGlmIChldmVudD8uTm9FY2hvKSB7XG4gICAgbG9nKCdyZWRhY3RlZCBpc0NvbXBsZXRlIHJlcXVlc3QnLCBjZm5SZXNwb25zZS5yZWRhY3REYXRhRnJvbVBheWxvYWQoc2FuaXRpemVkUmVxdWVzdCkpO1xuICB9IGVsc2Uge1xuICAgIGxvZygnaXNDb21wbGV0ZScsIHNhbml0aXplZFJlcXVlc3QpO1xuICB9XG5cbiAgY29uc3QgaXNDb21wbGV0ZVJlc3VsdCA9IGF3YWl0IGludm9rZVVzZXJGdW5jdGlvbihjb25zdHMuVVNFUl9JU19DT01QTEVURV9GVU5DVElPTl9BUk5fRU5WLCBzYW5pdGl6ZWRSZXF1ZXN0LCBldmVudC5SZXNwb25zZVVSTCkgYXMgSXNDb21wbGV0ZVJlc3BvbnNlO1xuICBpZiAoZXZlbnQ/Lk5vRWNobykge1xuICAgIGxvZygncmVkYWN0ZWQgdXNlciBpc0NvbXBsZXRlIHJldHVybmVkOicsIGNmblJlc3BvbnNlLnJlZGFjdERhdGFGcm9tUGF5bG9hZChpc0NvbXBsZXRlUmVzdWx0KSk7XG4gIH0gZWxzZSB7XG4gICAgbG9nKCd1c2VyIGlzQ29tcGxldGUgcmV0dXJuZWQ6JywgaXNDb21wbGV0ZVJlc3VsdCk7XG4gIH1cblxuICAvLyBpZiB3ZSBhcmUgbm90IGNvbXBsZXRlLCByZXR1cm4gZmFsc2UsIGFuZCBkb24ndCBzZW5kIGEgcmVzcG9uc2UgYmFjay5cbiAgaWYgKCFpc0NvbXBsZXRlUmVzdWx0LklzQ29tcGxldGUpIHtcbiAgICBpZiAoaXNDb21wbGV0ZVJlc3VsdC5EYXRhICYmIE9iamVjdC5rZXlzKGlzQ29tcGxldGVSZXN1bHQuRGF0YSkubGVuZ3RoID4gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdcIkRhdGFcIiBpcyBub3QgYWxsb3dlZCBpZiBcIklzQ29tcGxldGVcIiBpcyBcIkZhbHNlXCInKTtcbiAgICB9XG5cbiAgICAvLyBUaGlzIG11c3QgYmUgdGhlIGZ1bGwgZXZlbnQsIGl0IHdpbGwgYmUgZGVzZXJpYWxpemVkIGluIGBvblRpbWVvdXRgIHRvIHNlbmQgdGhlIHJlc3BvbnNlIHRvIENsb3VkRm9ybWF0aW9uXG4gICAgdGhyb3cgbmV3IGNmblJlc3BvbnNlLlJldHJ5KEpTT04uc3RyaW5naWZ5KGV2ZW50KSk7XG4gIH1cblxuICBjb25zdCByZXNwb25zZSA9IHtcbiAgICAuLi5ldmVudCxcbiAgICAuLi5pc0NvbXBsZXRlUmVzdWx0LFxuICAgIERhdGE6IHtcbiAgICAgIC4uLmV2ZW50LkRhdGEsXG4gICAgICAuLi5pc0NvbXBsZXRlUmVzdWx0LkRhdGEsXG4gICAgfSxcbiAgfTtcblxuICBhd2FpdCBjZm5SZXNwb25zZS5zdWJtaXRSZXNwb25zZSgnU1VDQ0VTUycsIHJlc3BvbnNlLCB7IG5vRWNobzogZXZlbnQuTm9FY2hvIH0pO1xufVxuXG4vLyBpbnZva2VkIHdoZW4gY29tcGxldGlvbiByZXRyaWVzIGFyZSBleGhhdXNlZC5cbmFzeW5jIGZ1bmN0aW9uIG9uVGltZW91dCh0aW1lb3V0RXZlbnQ6IGFueSkge1xuICBsb2coJ3RpbWVvdXRIYW5kbGVyJywgdGltZW91dEV2ZW50KTtcblxuICBjb25zdCBpc0NvbXBsZXRlUmVxdWVzdCA9IEpTT04ucGFyc2UoSlNPTi5wYXJzZSh0aW1lb3V0RXZlbnQuQ2F1c2UpLmVycm9yTWVzc2FnZSkgYXMgQVdTQ0RLQXN5bmNDdXN0b21SZXNvdXJjZS5Jc0NvbXBsZXRlUmVxdWVzdDtcbiAgYXdhaXQgY2ZuUmVzcG9uc2Uuc3VibWl0UmVzcG9uc2UoJ0ZBSUxFRCcsIGlzQ29tcGxldGVSZXF1ZXN0LCB7XG4gICAgcmVhc29uOiAnT3BlcmF0aW9uIHRpbWVkIG91dCcsXG4gIH0pO1xufVxuXG5hc3luYyBmdW5jdGlvbiBpbnZva2VVc2VyRnVuY3Rpb248QSBleHRlbmRzIHsgUmVzcG9uc2VVUkw6ICcuLi4nIH0+KGZ1bmN0aW9uQXJuRW52OiBzdHJpbmcsIHNhbml0aXplZFBheWxvYWQ6IEEsIHJlc3BvbnNlVXJsOiBzdHJpbmcpIHtcbiAgY29uc3QgZnVuY3Rpb25Bcm4gPSBnZXRFbnYoZnVuY3Rpb25Bcm5FbnYpO1xuICBsb2coYGV4ZWN1dGluZyB1c2VyIGZ1bmN0aW9uICR7ZnVuY3Rpb25Bcm59IHdpdGggcGF5bG9hZGAsIHNhbml0aXplZFBheWxvYWQpO1xuXG4gIC8vIHRyYW5zaWVudCBlcnJvcnMgc3VjaCBhcyB0aW1lb3V0cywgdGhyb3R0bGluZyBlcnJvcnMgKDQyOSksIGFuZCBvdGhlclxuICAvLyBlcnJvcnMgdGhhdCBhcmVuJ3QgY2F1c2VkIGJ5IGEgYmFkIHJlcXVlc3QgKDUwMCBzZXJpZXMpIGFyZSByZXRyaWVkXG4gIC8vIGF1dG9tYXRpY2FsbHkgYnkgdGhlIEphdmFTY3JpcHQgU0RLLlxuICBjb25zdCByZXNwID0gYXdhaXQgaW52b2tlRnVuY3Rpb24oe1xuICAgIEZ1bmN0aW9uTmFtZTogZnVuY3Rpb25Bcm4sXG5cbiAgICAvLyBDYW5ub3Qgc3RyaXAgJ1Jlc3BvbnNlVVJMJyBoZXJlIGFzIHRoaXMgd291bGQgYmUgYSBicmVha2luZyBjaGFuZ2UgZXZlbiB0aG91Z2ggdGhlIGRvd25zdHJlYW0gQ1IgZG9lc24ndCBuZWVkIGl0XG4gICAgUGF5bG9hZDogSlNPTi5zdHJpbmdpZnkoeyAuLi5zYW5pdGl6ZWRQYXlsb2FkLCBSZXNwb25zZVVSTDogcmVzcG9uc2VVcmwgfSksXG4gIH0pO1xuXG4gIGxvZygndXNlciBmdW5jdGlvbiByZXNwb25zZTonLCByZXNwLCB0eXBlb2YocmVzcCkpO1xuXG4gIC8vIFBhcnNlSnNvblBheWxvYWQgaXMgdmVyeSBkZWZlbnNpdmUuIEl0IHNob3VsZCBub3QgYmUgcG9zc2libGUgZm9yIGBQYXlsb2FkYFxuICAvLyB0byBiZSBhbnl0aGluZyBvdGhlciB0aGFuIGEgSlNPTiBlbmNvZGVkIHN0cmluZyAob3IgaW50YXJyYXkpLiBTb21ldGhpbmcgd2VpcmQgaXNcbiAgLy8gZ29pbmcgb24gaWYgdGhhdCBoYXBwZW5zLiBTdGlsbCwgd2Ugc2hvdWxkIGRvIG91ciBiZXN0IHRvIHN1cnZpdmUgaXQuXG4gIGNvbnN0IGpzb25QYXlsb2FkID0gcGFyc2VKc29uUGF5bG9hZChyZXNwLlBheWxvYWQpO1xuICBpZiAocmVzcC5GdW5jdGlvbkVycm9yKSB7XG4gICAgbG9nKCd1c2VyIGZ1bmN0aW9uIHRocmV3IGFuIGVycm9yOicsIHJlc3AuRnVuY3Rpb25FcnJvcik7XG5cbiAgICBjb25zdCBlcnJvck1lc3NhZ2UgPSBqc29uUGF5bG9hZC5lcnJvck1lc3NhZ2UgfHwgJ2Vycm9yJztcblxuICAgIC8vIHBhcnNlIGZ1bmN0aW9uIG5hbWUgZnJvbSBhcm5cbiAgICAvLyBhcm46JHtQYXJ0aXRpb259OmxhbWJkYToke1JlZ2lvbn06JHtBY2NvdW50fTpmdW5jdGlvbjoke0Z1bmN0aW9uTmFtZX1cbiAgICBjb25zdCBhcm4gPSBmdW5jdGlvbkFybi5zcGxpdCgnOicpO1xuICAgIGNvbnN0IGZ1bmN0aW9uTmFtZSA9IGFyblthcm4ubGVuZ3RoIC0gMV07XG5cbiAgICAvLyBhcHBlbmQgYSByZWZlcmVuY2UgdG8gdGhlIGxvZyBncm91cC5cbiAgICBjb25zdCBtZXNzYWdlID0gW1xuICAgICAgZXJyb3JNZXNzYWdlLFxuICAgICAgJycsXG4gICAgICBgTG9nczogL2F3cy9sYW1iZGEvJHtmdW5jdGlvbk5hbWV9YCwgLy8gY2xvdWR3YXRjaCBsb2cgZ3JvdXBcbiAgICAgICcnLFxuICAgIF0uam9pbignXFxuJyk7XG5cbiAgICBjb25zdCBlID0gbmV3IEVycm9yKG1lc3NhZ2UpO1xuXG4gICAgLy8gdGhlIG91dHB1dCB0aGF0IGdvZXMgdG8gQ0ZOIGlzIHdoYXQncyBpbiBgc3RhY2tgLCBub3QgdGhlIGVycm9yIG1lc3NhZ2UuXG4gICAgLy8gaWYgd2UgaGF2ZSBhIHJlbW90ZSB0cmFjZSwgY29uc3RydWN0IGEgbmljZSBtZXNzYWdlIHdpdGggbG9nIGdyb3VwIGluZm9ybWF0aW9uXG4gICAgaWYgKGpzb25QYXlsb2FkLnRyYWNlKSB7XG4gICAgICAvLyBza2lwIGZpcnN0IHRyYWNlIGxpbmUgYmVjYXVzZSBpdCdzIHRoZSBtZXNzYWdlXG4gICAgICBlLnN0YWNrID0gW21lc3NhZ2UsIC4uLmpzb25QYXlsb2FkLnRyYWNlLnNsaWNlKDEpXS5qb2luKCdcXG4nKTtcbiAgICB9XG5cbiAgICB0aHJvdyBlO1xuICB9XG5cbiAgcmV0dXJuIGpzb25QYXlsb2FkO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVSZXNwb25zZUV2ZW50KGNmblJlcXVlc3Q6IEFXU0xhbWJkYS5DbG91ZEZvcm1hdGlvbkN1c3RvbVJlc291cmNlRXZlbnQsIG9uRXZlbnRSZXN1bHQ6IE9uRXZlbnRSZXNwb25zZSk6IEFXU0NES0FzeW5jQ3VzdG9tUmVzb3VyY2UuSXNDb21wbGV0ZVJlcXVlc3Qge1xuICAvL1xuICAvLyB2YWxpZGF0ZSB0aGF0IG9uRXZlbnRSZXN1bHQgYWx3YXlzIGluY2x1ZGVzIGEgUGh5c2ljYWxSZXNvdXJjZUlkXG5cbiAgb25FdmVudFJlc3VsdCA9IG9uRXZlbnRSZXN1bHQgfHwgeyB9O1xuXG4gIC8vIGlmIHBoeXNpY2FsIElEIGlzIG5vdCByZXR1cm5lZCwgd2UgaGF2ZSBzb21lIGRlZmF1bHRzIGZvciB5b3UgYmFzZWRcbiAgLy8gb24gdGhlIHJlcXVlc3QgdHlwZS5cbiAgY29uc3QgcGh5c2ljYWxSZXNvdXJjZUlkID0gb25FdmVudFJlc3VsdC5QaHlzaWNhbFJlc291cmNlSWQgfHwgZGVmYXVsdFBoeXNpY2FsUmVzb3VyY2VJZChjZm5SZXF1ZXN0KTtcblxuICAvLyBpZiB3ZSBhcmUgaW4gREVMRVRFIGFuZCBwaHlzaWNhbCBJRCB3YXMgY2hhbmdlZCwgaXQncyBhbiBlcnJvci5cbiAgaWYgKGNmblJlcXVlc3QuUmVxdWVzdFR5cGUgPT09ICdEZWxldGUnICYmIHBoeXNpY2FsUmVzb3VyY2VJZCAhPT0gY2ZuUmVxdWVzdC5QaHlzaWNhbFJlc291cmNlSWQpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYERFTEVURTogY2Fubm90IGNoYW5nZSB0aGUgcGh5c2ljYWwgcmVzb3VyY2UgSUQgZnJvbSBcIiR7Y2ZuUmVxdWVzdC5QaHlzaWNhbFJlc291cmNlSWR9XCIgdG8gXCIke29uRXZlbnRSZXN1bHQuUGh5c2ljYWxSZXNvdXJjZUlkfVwiIGR1cmluZyBkZWxldGlvbmApO1xuICB9XG5cbiAgLy8gaWYgd2UgYXJlIGluIFVQREFURSBhbmQgcGh5c2ljYWwgSUQgd2FzIGNoYW5nZWQsIGl0J3MgYSByZXBsYWNlbWVudCAoanVzdCBsb2cpXG4gIGlmIChjZm5SZXF1ZXN0LlJlcXVlc3RUeXBlID09PSAnVXBkYXRlJyAmJiBwaHlzaWNhbFJlc291cmNlSWQgIT09IGNmblJlcXVlc3QuUGh5c2ljYWxSZXNvdXJjZUlkKSB7XG4gICAgbG9nKGBVUERBVEU6IGNoYW5naW5nIHBoeXNpY2FsIHJlc291cmNlIElEIGZyb20gXCIke2NmblJlcXVlc3QuUGh5c2ljYWxSZXNvdXJjZUlkfVwiIHRvIFwiJHtvbkV2ZW50UmVzdWx0LlBoeXNpY2FsUmVzb3VyY2VJZH1cImApO1xuICB9XG5cbiAgLy8gbWVyZ2UgcmVxdWVzdCBldmVudCBhbmQgcmVzdWx0IGV2ZW50IChyZXN1bHQgcHJldmFpbHMpLlxuICByZXR1cm4ge1xuICAgIC4uLmNmblJlcXVlc3QsXG4gICAgLi4ub25FdmVudFJlc3VsdCxcbiAgICBQaHlzaWNhbFJlc291cmNlSWQ6IHBoeXNpY2FsUmVzb3VyY2VJZCxcbiAgfTtcbn1cblxuLyoqXG4gKiBDYWxjdWxhdGVzIHRoZSBkZWZhdWx0IHBoeXNpY2FsIHJlc291cmNlIElEIGJhc2VkIGluIGNhc2UgdXNlciBoYW5kbGVyIGRpZFxuICogbm90IHJldHVybiBhIFBoeXNpY2FsUmVzb3VyY2VJZC5cbiAqXG4gKiBGb3IgXCJDUkVBVEVcIiwgaXQgdXNlcyB0aGUgUmVxdWVzdElkLlxuICogRm9yIFwiVVBEQVRFXCIgYW5kIFwiREVMRVRFXCIgYW5kIHJldHVybnMgdGhlIGN1cnJlbnQgUGh5c2ljYWxSZXNvdXJjZUlkICh0aGUgb25lIHByb3ZpZGVkIGluIGBldmVudGApLlxuICovXG5mdW5jdGlvbiBkZWZhdWx0UGh5c2ljYWxSZXNvdXJjZUlkKHJlcTogQVdTTGFtYmRhLkNsb3VkRm9ybWF0aW9uQ3VzdG9tUmVzb3VyY2VFdmVudCk6IHN0cmluZyB7XG4gIHN3aXRjaCAocmVxLlJlcXVlc3RUeXBlKSB7XG4gICAgY2FzZSAnQ3JlYXRlJzpcbiAgICAgIHJldHVybiByZXEuUmVxdWVzdElkO1xuXG4gICAgY2FzZSAnVXBkYXRlJzpcbiAgICBjYXNlICdEZWxldGUnOlxuICAgICAgcmV0dXJuIHJlcS5QaHlzaWNhbFJlc291cmNlSWQ7XG5cbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIFwiUmVxdWVzdFR5cGVcIiBpbiByZXF1ZXN0IFwiJHtKU09OLnN0cmluZ2lmeShyZXEpfVwiYCk7XG4gIH1cbn1cbiJdfQ==