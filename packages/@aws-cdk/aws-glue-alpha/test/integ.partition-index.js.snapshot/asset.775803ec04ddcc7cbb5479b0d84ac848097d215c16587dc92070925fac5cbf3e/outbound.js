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
exports.httpRequest = exports.invokeFunction = exports.startExecution = void 0;
/* istanbul ignore file */
const https = __importStar(require("https"));
// eslint-disable-next-line import/no-extraneous-dependencies
const client_lambda_1 = require("@aws-sdk/client-lambda");
// eslint-disable-next-line import/no-extraneous-dependencies
const client_sfn_1 = require("@aws-sdk/client-sfn");
const FRAMEWORK_HANDLER_TIMEOUT = 900000; // 15 minutes
// In order to honor the overall maximum timeout set for the target process,
// the default 2 minutes from AWS SDK has to be overriden:
// https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/Config.html#httpOptions-property
const awsSdkConfig = {
    httpOptions: { timeout: FRAMEWORK_HANDLER_TIMEOUT },
};
async function defaultHttpRequest(options, requestBody) {
    return new Promise((resolve, reject) => {
        try {
            const request = https.request(options, (response) => {
                response.resume(); // Consume the response but don't care about it
                if (!response.statusCode || response.statusCode >= 400) {
                    reject(new Error(`Unsuccessful HTTP response: ${response.statusCode}`));
                }
                else {
                    resolve();
                }
            });
            request.on('error', reject);
            request.write(requestBody);
            request.end();
        }
        catch (e) {
            reject(e);
        }
    });
}
let sfn;
let lambda;
async function defaultStartExecution(req) {
    if (!sfn) {
        sfn = new client_sfn_1.SFN(awsSdkConfig);
    }
    return sfn.startExecution(req);
}
async function defaultInvokeFunction(req) {
    if (!lambda) {
        lambda = new client_lambda_1.Lambda(awsSdkConfig);
    }
    try {
        /**
         * Try an initial invoke.
         *
         * When you try to invoke a function that is inactive, the invocation fails and Lambda sets
         * the function to pending state until the function resources are recreated.
         * If Lambda fails to recreate the resources, the function is set to the inactive state.
         *
         * We're using invoke first because `waitFor` doesn't trigger an inactive function to do anything,
         * it just runs `getFunction` and checks the state.
         */
        return await lambda.invoke(req);
    }
    catch {
        /**
         * The status of the Lambda function is checked every second for up to 300 seconds.
         * Exits the loop on 'Active' state and throws an error on 'Inactive' or 'Failed'.
         *
         * And now we wait.
         */
        await (0, client_lambda_1.waitUntilFunctionActiveV2)({
            client: lambda,
            maxWaitTime: 300,
        }, {
            FunctionName: req.FunctionName,
        });
        return lambda.invoke(req);
    }
}
exports.startExecution = defaultStartExecution;
exports.invokeFunction = defaultInvokeFunction;
exports.httpRequest = defaultHttpRequest;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib3V0Ym91bmQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyJvdXRib3VuZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQSwwQkFBMEI7QUFDMUIsNkNBQStCO0FBRS9CLDZEQUE2RDtBQUM3RCwwREFBMkU7QUFFM0UsNkRBQTZEO0FBQzdELG9EQUEwQztBQUUxQyxNQUFNLHlCQUF5QixHQUFHLE1BQU0sQ0FBQyxDQUFDLGFBQWE7QUFFdkQsNEVBQTRFO0FBQzVFLDBEQUEwRDtBQUMxRCwyRkFBMkY7QUFDM0YsTUFBTSxZQUFZLEdBQUc7SUFDbkIsV0FBVyxFQUFFLEVBQUUsT0FBTyxFQUFFLHlCQUF5QixFQUFFO0NBQ3BELENBQUM7QUFFRixLQUFLLFVBQVUsa0JBQWtCLENBQUMsT0FBNkIsRUFBRSxXQUFtQjtJQUNsRixPQUFPLElBQUksT0FBTyxDQUFPLENBQUMsT0FBTyxFQUFFLE1BQU0sRUFBRSxFQUFFO1FBQzNDLElBQUksQ0FBQztZQUNILE1BQU0sT0FBTyxHQUFHLEtBQUssQ0FBQyxPQUFPLENBQUMsT0FBTyxFQUFFLENBQUMsUUFBUSxFQUFFLEVBQUU7Z0JBQ2xELFFBQVEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLCtDQUErQztnQkFDbEUsSUFBSSxDQUFDLFFBQVEsQ0FBQyxVQUFVLElBQUksUUFBUSxDQUFDLFVBQVUsSUFBSSxHQUFHLEVBQUUsQ0FBQztvQkFDdkQsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLCtCQUErQixRQUFRLENBQUMsVUFBVSxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUMxRSxDQUFDO3FCQUFNLENBQUM7b0JBQ04sT0FBTyxFQUFFLENBQUM7Z0JBQ1osQ0FBQztZQUNILENBQUMsQ0FBQyxDQUFDO1lBQ0gsT0FBTyxDQUFDLEVBQUUsQ0FBQyxPQUFPLEVBQUUsTUFBTSxDQUFDLENBQUM7WUFDNUIsT0FBTyxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsQ0FBQztZQUMzQixPQUFPLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDaEIsQ0FBQztRQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUM7WUFDWCxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDWixDQUFDO0lBQ0gsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDO0FBRUQsSUFBSSxHQUFRLENBQUM7QUFDYixJQUFJLE1BQWMsQ0FBQztBQUVuQixLQUFLLFVBQVUscUJBQXFCLENBQUMsR0FBd0I7SUFDM0QsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ1QsR0FBRyxHQUFHLElBQUksZ0JBQUcsQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUM5QixDQUFDO0lBRUQsT0FBTyxHQUFHLENBQUMsY0FBYyxDQUFDLEdBQUcsQ0FBQyxDQUFDO0FBQ2pDLENBQUM7QUFFRCxLQUFLLFVBQVUscUJBQXFCLENBQUMsR0FBdUI7SUFDMUQsSUFBSSxDQUFDLE1BQU0sRUFBRSxDQUFDO1FBQ1osTUFBTSxHQUFHLElBQUksc0JBQU0sQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUNwQyxDQUFDO0lBRUQsSUFBSSxDQUFDO1FBQ0g7Ozs7Ozs7OztXQVNHO1FBQ0gsT0FBTyxNQUFNLE1BQU0sQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDbEMsQ0FBQztJQUFDLE1BQU0sQ0FBQztRQUNQOzs7OztXQUtHO1FBQ0gsTUFBTSxJQUFBLHlDQUF5QixFQUFDO1lBQzlCLE1BQU0sRUFBRSxNQUFNO1lBQ2QsV0FBVyxFQUFFLEdBQUc7U0FDakIsRUFBRTtZQUNELFlBQVksRUFBRSxHQUFHLENBQUMsWUFBWTtTQUMvQixDQUFDLENBQUM7UUFDSCxPQUFPLE1BQU0sQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDNUIsQ0FBQztBQUNILENBQUM7QUFFVSxRQUFBLGNBQWMsR0FBRyxxQkFBcUIsQ0FBQztBQUN2QyxRQUFBLGNBQWMsR0FBRyxxQkFBcUIsQ0FBQztBQUN2QyxRQUFBLFdBQVcsR0FBRyxrQkFBa0IsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qIGlzdGFuYnVsIGlnbm9yZSBmaWxlICovXG5pbXBvcnQgKiBhcyBodHRwcyBmcm9tICdodHRwcyc7XG5pbXBvcnQgdHlwZSB7IEludm9jYXRpb25SZXNwb25zZSwgSW52b2tlQ29tbWFuZElucHV0IH0gZnJvbSAnQGF3cy1zZGsvY2xpZW50LWxhbWJkYSc7XG4vLyBlc2xpbnQtZGlzYWJsZS1uZXh0LWxpbmUgaW1wb3J0L25vLWV4dHJhbmVvdXMtZGVwZW5kZW5jaWVzXG5pbXBvcnQgeyBMYW1iZGEsIHdhaXRVbnRpbEZ1bmN0aW9uQWN0aXZlVjIgfSBmcm9tICdAYXdzLXNkay9jbGllbnQtbGFtYmRhJztcbmltcG9ydCB0eXBlIHsgU3RhcnRFeGVjdXRpb25JbnB1dCwgU3RhcnRFeGVjdXRpb25PdXRwdXQgfSBmcm9tICdAYXdzLXNkay9jbGllbnQtc2ZuJztcbi8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBpbXBvcnQvbm8tZXh0cmFuZW91cy1kZXBlbmRlbmNpZXNcbmltcG9ydCB7IFNGTiB9IGZyb20gJ0Bhd3Mtc2RrL2NsaWVudC1zZm4nO1xuXG5jb25zdCBGUkFNRVdPUktfSEFORExFUl9USU1FT1VUID0gOTAwMDAwOyAvLyAxNSBtaW51dGVzXG5cbi8vIEluIG9yZGVyIHRvIGhvbm9yIHRoZSBvdmVyYWxsIG1heGltdW0gdGltZW91dCBzZXQgZm9yIHRoZSB0YXJnZXQgcHJvY2Vzcyxcbi8vIHRoZSBkZWZhdWx0IDIgbWludXRlcyBmcm9tIEFXUyBTREsgaGFzIHRvIGJlIG92ZXJyaWRlbjpcbi8vIGh0dHBzOi8vZG9jcy5hd3MuYW1hem9uLmNvbS9BV1NKYXZhU2NyaXB0U0RLL2xhdGVzdC9BV1MvQ29uZmlnLmh0bWwjaHR0cE9wdGlvbnMtcHJvcGVydHlcbmNvbnN0IGF3c1Nka0NvbmZpZyA9IHtcbiAgaHR0cE9wdGlvbnM6IHsgdGltZW91dDogRlJBTUVXT1JLX0hBTkRMRVJfVElNRU9VVCB9LFxufTtcblxuYXN5bmMgZnVuY3Rpb24gZGVmYXVsdEh0dHBSZXF1ZXN0KG9wdGlvbnM6IGh0dHBzLlJlcXVlc3RPcHRpb25zLCByZXF1ZXN0Qm9keTogc3RyaW5nKSB7XG4gIHJldHVybiBuZXcgUHJvbWlzZTx2b2lkPigocmVzb2x2ZSwgcmVqZWN0KSA9PiB7XG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHJlcXVlc3QgPSBodHRwcy5yZXF1ZXN0KG9wdGlvbnMsIChyZXNwb25zZSkgPT4ge1xuICAgICAgICByZXNwb25zZS5yZXN1bWUoKTsgLy8gQ29uc3VtZSB0aGUgcmVzcG9uc2UgYnV0IGRvbid0IGNhcmUgYWJvdXQgaXRcbiAgICAgICAgaWYgKCFyZXNwb25zZS5zdGF0dXNDb2RlIHx8IHJlc3BvbnNlLnN0YXR1c0NvZGUgPj0gNDAwKSB7XG4gICAgICAgICAgcmVqZWN0KG5ldyBFcnJvcihgVW5zdWNjZXNzZnVsIEhUVFAgcmVzcG9uc2U6ICR7cmVzcG9uc2Uuc3RhdHVzQ29kZX1gKSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgcmVzb2x2ZSgpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICAgIHJlcXVlc3Qub24oJ2Vycm9yJywgcmVqZWN0KTtcbiAgICAgIHJlcXVlc3Qud3JpdGUocmVxdWVzdEJvZHkpO1xuICAgICAgcmVxdWVzdC5lbmQoKTtcbiAgICB9IGNhdGNoIChlKSB7XG4gICAgICByZWplY3QoZSk7XG4gICAgfVxuICB9KTtcbn1cblxubGV0IHNmbjogU0ZOO1xubGV0IGxhbWJkYTogTGFtYmRhO1xuXG5hc3luYyBmdW5jdGlvbiBkZWZhdWx0U3RhcnRFeGVjdXRpb24ocmVxOiBTdGFydEV4ZWN1dGlvbklucHV0KTogUHJvbWlzZTxTdGFydEV4ZWN1dGlvbk91dHB1dD4ge1xuICBpZiAoIXNmbikge1xuICAgIHNmbiA9IG5ldyBTRk4oYXdzU2RrQ29uZmlnKTtcbiAgfVxuXG4gIHJldHVybiBzZm4uc3RhcnRFeGVjdXRpb24ocmVxKTtcbn1cblxuYXN5bmMgZnVuY3Rpb24gZGVmYXVsdEludm9rZUZ1bmN0aW9uKHJlcTogSW52b2tlQ29tbWFuZElucHV0KTogUHJvbWlzZTxJbnZvY2F0aW9uUmVzcG9uc2U+IHtcbiAgaWYgKCFsYW1iZGEpIHtcbiAgICBsYW1iZGEgPSBuZXcgTGFtYmRhKGF3c1Nka0NvbmZpZyk7XG4gIH1cblxuICB0cnkge1xuICAgIC8qKlxuICAgICAqIFRyeSBhbiBpbml0aWFsIGludm9rZS5cbiAgICAgKlxuICAgICAqIFdoZW4geW91IHRyeSB0byBpbnZva2UgYSBmdW5jdGlvbiB0aGF0IGlzIGluYWN0aXZlLCB0aGUgaW52b2NhdGlvbiBmYWlscyBhbmQgTGFtYmRhIHNldHNcbiAgICAgKiB0aGUgZnVuY3Rpb24gdG8gcGVuZGluZyBzdGF0ZSB1bnRpbCB0aGUgZnVuY3Rpb24gcmVzb3VyY2VzIGFyZSByZWNyZWF0ZWQuXG4gICAgICogSWYgTGFtYmRhIGZhaWxzIHRvIHJlY3JlYXRlIHRoZSByZXNvdXJjZXMsIHRoZSBmdW5jdGlvbiBpcyBzZXQgdG8gdGhlIGluYWN0aXZlIHN0YXRlLlxuICAgICAqXG4gICAgICogV2UncmUgdXNpbmcgaW52b2tlIGZpcnN0IGJlY2F1c2UgYHdhaXRGb3JgIGRvZXNuJ3QgdHJpZ2dlciBhbiBpbmFjdGl2ZSBmdW5jdGlvbiB0byBkbyBhbnl0aGluZyxcbiAgICAgKiBpdCBqdXN0IHJ1bnMgYGdldEZ1bmN0aW9uYCBhbmQgY2hlY2tzIHRoZSBzdGF0ZS5cbiAgICAgKi9cbiAgICByZXR1cm4gYXdhaXQgbGFtYmRhLmludm9rZShyZXEpO1xuICB9IGNhdGNoIHtcbiAgICAvKipcbiAgICAgKiBUaGUgc3RhdHVzIG9mIHRoZSBMYW1iZGEgZnVuY3Rpb24gaXMgY2hlY2tlZCBldmVyeSBzZWNvbmQgZm9yIHVwIHRvIDMwMCBzZWNvbmRzLlxuICAgICAqIEV4aXRzIHRoZSBsb29wIG9uICdBY3RpdmUnIHN0YXRlIGFuZCB0aHJvd3MgYW4gZXJyb3Igb24gJ0luYWN0aXZlJyBvciAnRmFpbGVkJy5cbiAgICAgKlxuICAgICAqIEFuZCBub3cgd2Ugd2FpdC5cbiAgICAgKi9cbiAgICBhd2FpdCB3YWl0VW50aWxGdW5jdGlvbkFjdGl2ZVYyKHtcbiAgICAgIGNsaWVudDogbGFtYmRhLFxuICAgICAgbWF4V2FpdFRpbWU6IDMwMCxcbiAgICB9LCB7XG4gICAgICBGdW5jdGlvbk5hbWU6IHJlcS5GdW5jdGlvbk5hbWUsXG4gICAgfSk7XG4gICAgcmV0dXJuIGxhbWJkYS5pbnZva2UocmVxKTtcbiAgfVxufVxuXG5leHBvcnQgbGV0IHN0YXJ0RXhlY3V0aW9uID0gZGVmYXVsdFN0YXJ0RXhlY3V0aW9uO1xuZXhwb3J0IGxldCBpbnZva2VGdW5jdGlvbiA9IGRlZmF1bHRJbnZva2VGdW5jdGlvbjtcbmV4cG9ydCBsZXQgaHR0cFJlcXVlc3QgPSBkZWZhdWx0SHR0cFJlcXVlc3Q7XG4iXX0=