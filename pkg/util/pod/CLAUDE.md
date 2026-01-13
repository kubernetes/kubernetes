# Package: pod

## Purpose
Provides utilities for working with Pod objects, particularly around status patching and conditions.

## Key Functions
- `PatchPodStatus()` - Patches pod status using strategic merge patch
- `ReplaceOrAppendPodCondition()` - Replaces existing condition or appends new one

## PatchPodStatus Details
- Creates minimal patch containing only changed fields
- Includes UID as precondition to prevent stale updates
- Returns unchanged=true if patch would be empty
- Uses strategic merge patch for proper list handling

## Design Patterns
- Strategic merge patch for efficient status updates
- UID precondition prevents overwriting concurrent changes
- Condition helpers maintain condition list consistency
- Commonly used by kubelet for pod status reporting
