# Package: cleaner

## Purpose
Implements a garbage collector for Certificate Signing Requests (CSRs) that removes old or expired CSRs to prevent unbounded growth.

## Key Types/Structs
- `CSRCleanerController`: Controller with CSR client and lister for GC operations

## Key Constants
- `pollingInterval`: 1 hour between full CSR list scans
- `approvedExpiration`: 1 hour after approval before cleanup
- `deniedExpiration`: 1 hour after denial before cleanup
- `pendingExpiration`: 24 hours for pending CSRs before cleanup

## Key Functions
- `NewCSRCleanerController(csrClient, csrInformer)`: Creates the cleaner controller
- `Run(ctx, workers)`: Starts the controller with periodic worker loops

## Cleanup Criteria (any one triggers deletion)
- `isIssuedPastDeadline`: Approved with certificate, older than approvedExpiration
- `isDeniedPastDeadline`: Denied, older than deniedExpiration
- `isFailedPastDeadline`: Failed, older than deniedExpiration
- `isPendingPastDeadline`: No conditions, older than pendingExpiration
- `isIssuedExpired`: Has certificate with NotAfter in the past
- `isApprovedUnissuedPastDeadline`: Approved but no certificate, older than pendingExpiration

## Design Notes
- Runs on polling interval rather than watch-based (periodic full list)
- Prevents CSR accumulation in clusters with automated CSR creation/approval
- Essential for clusters with short certificate durations
- Does not use workqueue; directly iterates all CSRs each poll
