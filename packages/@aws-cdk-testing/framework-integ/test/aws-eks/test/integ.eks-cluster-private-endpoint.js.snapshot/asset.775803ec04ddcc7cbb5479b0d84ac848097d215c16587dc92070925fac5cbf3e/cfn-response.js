"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.Retry = exports.includeStackTraces = exports.MISSING_PHYSICAL_ID_MARKER = exports.CREATE_FAILED_PHYSICAL_ID_MARKER = void 0;
exports.submitResponse = submitResponse;
exports.safeHandler = safeHandler;
exports.redactDataFromPayload = redactDataFromPayload;
const url = __importStar(require("url"));
const outbound_1 = require("./outbound");
const util_1 = require("./util");
exports.CREATE_FAILED_PHYSICAL_ID_MARKER = 'AWSCDK::CustomResourceProviderFramework::CREATE_FAILED';
exports.MISSING_PHYSICAL_ID_MARKER = 'AWSCDK::CustomResourceProviderFramework::MISSING_PHYSICAL_ID';
async function submitResponse(status, event, options = {}) {
    const json = {
        Status: status,
        Reason: options.reason || status,
        StackId: event.StackId,
        RequestId: event.RequestId,
        PhysicalResourceId: event.PhysicalResourceId || exports.MISSING_PHYSICAL_ID_MARKER,
        LogicalResourceId: event.LogicalResourceId,
        NoEcho: options.noEcho,
        Data: event.Data,
    };
    const responseBody = JSON.stringify(json);
    const parsedUrl = url.parse(event.ResponseURL);
    const loggingSafeUrl = `${parsedUrl.protocol}//${parsedUrl.hostname}/${parsedUrl.pathname}?***`;
    if (options?.noEcho) {
        (0, util_1.log)('submit redacted response to cloudformation', loggingSafeUrl, redactDataFromPayload(json));
    }
    else {
        (0, util_1.log)('submit response to cloudformation', loggingSafeUrl, json);
    }
    const retryOptions = {
        attempts: 5,
        sleep: 1000,
    };
    await (0, util_1.withRetries)(retryOptions, outbound_1.httpRequest)({
        hostname: parsedUrl.hostname,
        path: parsedUrl.path,
        method: 'PUT',
        headers: {
            'content-type': '',
            'content-length': Buffer.byteLength(responseBody, 'utf8'),
        },
    }, responseBody);
}
exports.includeStackTraces = true; // for unit tests
function safeHandler(block) {
    return async (event) => {
        // ignore DELETE event when the physical resource ID is the marker that
        // indicates that this DELETE is a subsequent DELETE to a failed CREATE
        // operation.
        if (event.RequestType === 'Delete' && event.PhysicalResourceId === exports.CREATE_FAILED_PHYSICAL_ID_MARKER) {
            (0, util_1.log)('ignoring DELETE event caused by a failed CREATE event');
            await submitResponse('SUCCESS', event);
            return;
        }
        try {
            await block(event);
        }
        catch (e) {
            // tell waiter state machine to retry
            if (e instanceof Retry) {
                (0, util_1.log)('retry requested by handler');
                throw e;
            }
            if (!event.PhysicalResourceId) {
                // special case: if CREATE fails, which usually implies, we usually don't
                // have a physical resource id. in this case, the subsequent DELETE
                // operation does not have any meaning, and will likely fail as well. to
                // address this, we use a marker so the provider framework can simply
                // ignore the subsequent DELETE.
                if (event.RequestType === 'Create') {
                    (0, util_1.log)('CREATE failed, responding with a marker physical resource id so that the subsequent DELETE will be ignored');
                    event.PhysicalResourceId = exports.CREATE_FAILED_PHYSICAL_ID_MARKER;
                }
                else {
                    // otherwise, if PhysicalResourceId is not specified, something is
                    // terribly wrong because all other events should have an ID.
                    (0, util_1.log)(`ERROR: Malformed event. "PhysicalResourceId" is required: ${JSON.stringify({ ...event, ResponseURL: '...' })}`);
                }
            }
            // this is an actual error, fail the activity altogether and exist.
            await submitResponse('FAILED', event, {
                reason: exports.includeStackTraces ? e.stack : e.message,
            });
        }
    };
}
function redactDataFromPayload(payload) {
    // Create a deep copy of the payload object
    const redactedPayload = JSON.parse(JSON.stringify(payload));
    // Redact the data in the copied payload object
    if (redactedPayload.Data) {
        const keys = Object.keys(redactedPayload.Data);
        for (const key of keys) {
            redactedPayload.Data[key] = '*****';
        }
    }
    return redactedPayload;
}
class Retry extends Error {
}
exports.Retry = Retry;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY2ZuLXJlc3BvbnNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiY2ZuLXJlc3BvbnNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQXVCQSx3Q0FtQ0M7QUFJRCxrQ0EwQ0M7QUFFRCxzREFZQztBQXJIRCx5Q0FBMkI7QUFDM0IseUNBQXlDO0FBQ3pDLGlDQUEwQztBQUc3QixRQUFBLGdDQUFnQyxHQUFHLHdEQUF3RCxDQUFDO0FBQzVGLFFBQUEsMEJBQTBCLEdBQUcsOERBQThELENBQUM7QUFnQmxHLEtBQUssVUFBVSxjQUFjLENBQUMsTUFBNEIsRUFBRSxLQUFpQyxFQUFFLFVBQXlDLEVBQUc7SUFDaEosTUFBTSxJQUFJLEdBQW1EO1FBQzNELE1BQU0sRUFBRSxNQUFNO1FBQ2QsTUFBTSxFQUFFLE9BQU8sQ0FBQyxNQUFNLElBQUksTUFBTTtRQUNoQyxPQUFPLEVBQUUsS0FBSyxDQUFDLE9BQU87UUFDdEIsU0FBUyxFQUFFLEtBQUssQ0FBQyxTQUFTO1FBQzFCLGtCQUFrQixFQUFFLEtBQUssQ0FBQyxrQkFBa0IsSUFBSSxrQ0FBMEI7UUFDMUUsaUJBQWlCLEVBQUUsS0FBSyxDQUFDLGlCQUFpQjtRQUMxQyxNQUFNLEVBQUUsT0FBTyxDQUFDLE1BQU07UUFDdEIsSUFBSSxFQUFFLEtBQUssQ0FBQyxJQUFJO0tBQ2pCLENBQUM7SUFFRixNQUFNLFlBQVksR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBRTFDLE1BQU0sU0FBUyxHQUFHLEdBQUcsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxDQUFDO0lBQy9DLE1BQU0sY0FBYyxHQUFHLEdBQUcsU0FBUyxDQUFDLFFBQVEsS0FBSyxTQUFTLENBQUMsUUFBUSxJQUFJLFNBQVMsQ0FBQyxRQUFRLE1BQU0sQ0FBQztJQUNoRyxJQUFJLE9BQU8sRUFBRSxNQUFNLEVBQUUsQ0FBQztRQUNw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QyxJQUFJLEVBQUUsQ0FBQztRQUN6QixNQUFNLElBQUksR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUMvQyxLQUFLLE1BQU0sR0FBRyxJQUFJLElBQUksRUFBRSxDQUFDO1lBQ3ZCLGVBQWUsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsT0FBTyxDQUFDO1FBQ3RDLENBQUM7SUFDSCxDQUFDO0lBQ0QsT0FBTyxlQUFlLENBQUM7QUFDekIsQ0FBQztBQUVELE1BQWEsS0FBTSxTQUFRLEtBQUs7Q0FBSTtBQUFwQyxzQkFBb0MiLCJzb3VyY2VzQ29udGVudCI6WyJcbmltcG9ydCAqIGFzIHVybCBmcm9tICd1cmwnO1xuaW1wb3J0IHsgaHR0cFJlcXVlc3QgfSBmcm9tICcuL291dGJvdW5kJztcbmltcG9ydCB7IGxvZywgd2l0aFJldHJpZXMgfSBmcm9tICcuL3V0aWwnO1xuaW1wb3J0IHR5cGUgeyBPbkV2ZW50UmVzcG9uc2UgfSBmcm9tICcuLi90eXBlcyc7XG5cbmV4cG9ydCBjb25zdCBDUkVBVEVfRkFJTEVEX1BIWVNJQ0FMX0lEX01BUktFUiA9ICdBV1NDREs6OkN1c3RvbVJlc291cmNlUHJvdmlkZXJGcmFtZXdvcms6OkNSRUFURV9GQUlMRUQnO1xuZXhwb3J0IGNvbnN0IE1JU1NJTkdfUEhZU0lDQUxfSURfTUFSS0VSID0gJ0FXU0NESzo6Q3VzdG9tUmVzb3VyY2VQcm92aWRlckZyYW1ld29yazo6TUlTU0lOR19QSFlTSUNBTF9JRCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgQ2xvdWRGb3JtYXRpb25SZXNwb25zZU9wdGlvbnMge1xuICByZWFkb25seSByZWFzb24/OiBzdHJpbmc7XG4gIHJlYWRvbmx5IG5vRWNobz86IGJvb2xlYW47XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgQ2xvdWRGb3JtYXRpb25FdmVudENvbnRleHQge1xuICBTdGFja0lkOiBzdHJpbmc7XG4gIFJlcXVlc3RJZDogc3RyaW5nO1xuICBQaHlzaWNhbFJlc291cmNlSWQ/OiBzdHJpbmc7XG4gIExvZ2ljYWxSZXNvdXJjZUlkOiBzdHJpbmc7XG4gIFJlc3BvbnNlVVJMOiBzdHJpbmc7XG4gIERhdGE/OiBhbnk7XG59XG5cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBzdWJtaXRSZXNwb25zZShzdGF0dXM6ICdTVUNDRVNTJyB8ICdGQUlMRUQnLCBldmVudDogQ2xvdWRGb3JtYXRpb25FdmVudENvbnRleHQsIG9wdGlvbnM6IENsb3VkRm9ybWF0aW9uUmVzcG9uc2VPcHRpb25zID0geyB9KSB7XG4gIGNvbnN0IGpzb246IEFXU0xhbWJkYS5DbG91ZEZvcm1hdGlvbkN1c3RvbVJlc291cmNlUmVzcG9uc2UgPSB7XG4gICAgU3RhdHVzOiBzdGF0dXMsXG4gICAgUmVhc29uOiBvcHRpb25zLnJlYXNvbiB8fCBzdGF0dXMsXG4gICAgU3RhY2tJZDogZXZlbnQuU3RhY2tJZCxcbiAgICBSZXF1ZXN0SWQ6IGV2ZW50LlJlcXVlc3RJZCxcbiAgICBQaHlzaWNhbFJlc291cmNlSWQ6IGV2ZW50LlBoeXNpY2FsUmVzb3VyY2VJZCB8fCBNSVNTSU5HX1BIWVNJQ0FMX0lEX01BUktFUixcbiAgICBMb2dpY2FsUmVzb3VyY2VJZDogZXZlbnQuTG9naWNhbFJlc291cmNlSWQsXG4gICAgTm9FY2hvOiBvcHRpb25zLm5vRWNobyxcbiAgICBEYXRhOiBldmVudC5EYXRhLFxuICB9O1xuXG4gIGNvbnN0IHJlc3BvbnNlQm9keSA9IEpTT04uc3RyaW5naWZ5KGpzb24pO1xuXG4gIGNvbnN0IHBhcnNlZFVybCA9IHVybC5wYXJzZShldmVudC5SZXNwb25zZVVSTCk7XG4gIGNvbnN0IGxvZ2dpbmdTYWZlVXJsID0gYCR7cGFyc2VkVXJsLnByb3RvY29sfS8vJHtwYXJzZWRVcmwuaG9zdG5hbWV9LyR7cGFyc2VkVXJsLnBhdGhuYW1lfT8qKipgO1xuICBpZiAob3B0aW9ucz8ubm9FY2hvKSB7XG4gICAgbG9nKCdzdWJtaXQgcmVkYWN0ZWQgcmVzcG9uc2UgdG8gY2xvdWRmb3JtYXRpb24nLCBsb2dnaW5nU2FmZVVybCwgcmVkYWN0RGF0YUZyb21QYXlsb2FkKGpzb24pKTtcbiAgfSBlbHNlIHtcbiAgICBsb2coJ3N1Ym1pdCByZXNwb25zZSB0byBjbG91ZGZvcm1hdGlvbicsIGxvZ2dpbmdTYWZlVXJsLCBqc29uKTtcbiAgfVxuXG4gIGNvbnN0IHJldHJ5T3B0aW9ucyA9IHtcbiAgICBhdHRlbXB0czogNSxcbiAgICBzbGVlcDogMTAwMCxcbiAgfTtcbiAgYXdhaXQgd2l0aFJldHJpZXMocmV0cnlPcHRpb25zLCBodHRwUmVxdWVzdCkoe1xuICAgIGhvc3RuYW1lOiBwYXJzZWRVcmwuaG9zdG5hbWUsXG4gICAgcGF0aDogcGFyc2VkVXJsLnBhdGgsXG4gICAgbWV0aG9kOiAnUFVUJyxcbiAgICBoZWFkZXJzOiB7XG4gICAgICAnY29udGVudC10eXBlJzogJycsXG4gICAgICAnY29udGVudC1sZW5ndGgnOiBCdWZmZXIuYnl0ZUxlbmd0aChyZXNwb25zZUJvZHksICd1dGY4JyksXG4gICAgfSxcbiAgfSwgcmVzcG9uc2VCb2R5KTtcbn1cblxuZXhwb3J0IGxldCBpbmNsdWRlU3RhY2tUcmFjZXMgPSB0cnVlOyAvLyBmb3IgdW5pdCB0ZXN0c1xuXG5leHBvcnQgZnVuY3Rpb24gc2FmZUhhbmRsZXIoYmxvY2s6IChldmVudDogYW55KSA9PiBQcm9taXNlPHZvaWQ+KSB7XG4gIHJldHVybiBhc3luYyAoZXZlbnQ6IGFueSkgPT4ge1xuICAgIC8vIGlnbm9yZSBERUxFVEUgZXZlbnQgd2hlbiB0aGUgcGh5c2ljYWwgcmVzb3VyY2UgSUQgaXMgdGhlIG1hcmtlciB0aGF0XG4gICAgLy8gaW5kaWNhdGVzIHRoYXQgdGhpcyBERUxFVEUgaXMgYSBzdWJzZXF1ZW50IERFTEVURSB0byBhIGZhaWxlZCBDUkVBVEVcbiAgICAvLyBvcGVyYXRpb24uXG4gICAgaWYgKGV2ZW50LlJlcXVlc3RUeXBlID09PSAnRGVsZXRlJyAmJiBldmVudC5QaHlzaWNhbFJlc291cmNlSWQgPT09IENSRUFURV9GQUlMRURfUEhZU0lDQUxfSURfTUFSS0VSKSB7XG4gICAgICBsb2coJ2lnbm9yaW5nIERFTEVURSBldmVudCBjYXVzZWQgYnkgYSBmYWlsZWQgQ1JFQVRFIGV2ZW50Jyk7XG4gICAgICBhd2FpdCBzdWJtaXRSZXNwb25zZSgnU1VDQ0VTUycsIGV2ZW50KTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0cnkge1xuICAgICAgYXdhaXQgYmxvY2soZXZlbnQpO1xuICAgIH0gY2F0Y2ggKGU6IGFueSkge1xuICAgICAgLy8gdGVsbCB3YWl0ZXIgc3RhdGUgbWFjaGluZSB0byByZXRyeVxuICAgICAgaWYgKGUgaW5zdGFuY2VvZiBSZXRyeSkge1xuICAgICAgICBsb2coJ3JldHJ5IHJlcXVlc3RlZCBieSBoYW5kbGVyJyk7XG4gICAgICAgIHRocm93IGU7XG4gICAgICB9XG5cbiAgICAgIGlmICghZXZlbnQuUGh5c2ljYWxSZXNvdXJjZUlkKSB7XG4gICAgICAgIC8vIHNwZWNpYWwgY2FzZTogaWYgQ1JFQVRFIGZhaWxzLCB3aGljaCB1c3VhbGx5IGltcGxpZXMsIHdlIHVzdWFsbHkgZG9uJ3RcbiAgICAgICAgLy8gaGF2ZSBhIHBoeXNpY2FsIHJlc291cmNlIGlkLiBpbiB0aGlzIGNhc2UsIHRoZSBzdWJzZXF1ZW50IERFTEVURVxuICAgICAgICAvLyBvcGVyYXRpb24gZG9lcyBub3QgaGF2ZSBhbnkgbWVhbmluZywgYW5kIHdpbGwgbGlrZWx5IGZhaWwgYXMgd2VsbC4gdG9cbiAgICAgICAgLy8gYWRkcmVzcyB0aGlzLCB3ZSB1c2UgYSBtYXJrZXIgc28gdGhlIHByb3ZpZGVyIGZyYW1ld29yayBjYW4gc2ltcGx5XG4gICAgICAgIC8vIGlnbm9yZSB0aGUgc3Vic2VxdWVudCBERUxFVEUuXG4gICAgICAgIGlmIChldmVudC5SZXF1ZXN0VHlwZSA9PT0gJ0NyZWF0ZScpIHtcbiAgICAgICAgICBsb2coJ0NSRUFURSBmYWlsZWQsIHJlc3BvbmRpbmcgd2l0aCBhIG1hcmtlciBwaHlzaWNhbCByZXNvdXJjZSBpZCBzbyB0aGF0IHRoZSBzdWJzZXF1ZW50IERFTEVURSB3aWxsIGJlIGlnbm9yZWQnKTtcbiAgICAgICAgICBldmVudC5QaHlzaWNhbFJlc291cmNlSWQgPSBDUkVBVEVfRkFJTEVEX1BIWVNJQ0FMX0lEX01BUktFUjtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAvLyBvdGhlcndpc2UsIGlmIFBoeXNpY2FsUmVzb3VyY2VJZCBpcyBub3Qgc3BlY2lmaWVkLCBzb21ldGhpbmcgaXNcbiAgICAgICAgICAvLyB0ZXJyaWJseSB3cm9uZyBiZWNhdXNlIGFsbCBvdGhlciBldmVudHMgc2hvdWxkIGhhdmUgYW4gSUQuXG4gICAgICAgICAgbG9nKGBFUlJPUjogTWFsZm9ybWVkIGV2ZW50LiBcIlBoeXNpY2FsUmVzb3VyY2VJZFwiIGlzIHJlcXVpcmVkOiAke0pTT04uc3RyaW5naWZ5KHsgLi4uZXZlbnQsIFJlc3BvbnNlVVJMOiAnLi4uJyB9KX1gKTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICAvLyB0aGlzIGlzIGFuIGFjdHVhbCBlcnJvciwgZmFpbCB0aGUgYWN0aXZpdHkgYWx0b2dldGhlciBhbmQgZXhpc3QuXG4gICAgICBhd2FpdCBzdWJtaXRSZXNwb25zZSgnRkFJTEVEJywgZXZlbnQsIHtcbiAgICAgICAgcmVhc29uOiBpbmNsdWRlU3RhY2tUcmFjZXMgPyBlLnN0YWNrIDogZS5tZXNzYWdlLFxuICAgICAgfSk7XG4gICAgfVxuICB9O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcmVkYWN0RGF0YUZyb21QYXlsb2FkKHBheWxvYWQ6IE9uRXZlbnRSZXNwb25zZSkge1xuICAvLyBDcmVhdGUgYSBkZWVwIGNvcHkgb2YgdGhlIHBheWxvYWQgb2JqZWN0XG4gIGNvbnN0IHJlZGFjdGVkUGF5bG9hZDogT25FdmVudFJlc3BvbnNlID0gSlNPTi5wYXJzZShKU09OLnN0cmluZ2lmeShwYXlsb2FkKSk7XG5cbiAgLy8gUmVkYWN0IHRoZSBkYXRhIGluIHRoZSBjb3BpZWQgcGF5bG9hZCBvYmplY3RcbiAgaWYgKHJlZGFjdGVkUGF5bG9hZC5EYXRhKSB7XG4gICAgY29uc3Qga2V5cyA9IE9iamVjdC5rZXlzKHJlZGFjdGVkUGF5bG9hZC5EYXRhKTtcbiAgICBmb3IgKGNvbnN0IGtleSBvZiBrZXlzKSB7XG4gICAgICByZWRhY3RlZFBheWxvYWQuRGF0YVtrZXldID0gJyoqKioqJztcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHJlZGFjdGVkUGF5bG9hZDtcbn1cblxuZXhwb3J0IGNsYXNzIFJldHJ5IGV4dGVuZHMgRXJyb3IgeyB9XG4iXX0=