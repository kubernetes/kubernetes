<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Resource Quota - reservations

## Problem Description

When a request for creating a resource is received,
the associated resource quota is checked by the
server in admission control.  If the request would
not violate the quota, the quota usage is incremented
and the requesting object is allowed to be created.
Quota usage is incremented using compare-and-swap
to allow for optimistic locking.  If the quota usage
was stale, the admission control logic runs again
to validate the request is still valid with the latest
quota document.

A quota controller regularly sweeps the system to
recalculate observed usage.  This is primarily useful
for monitoring deleted resources and replenishing
available quota.  If the quota controller sees a delta
in its recalculated usage relative to the currently
reported status, it will update the usage.  If its update
is stale, it will recalculate usage again.

If the quota controller works has an up to date quota,
but interacts with a latent API server, the quota controller
could under report usage.  This is possible in a single
server environment (though the race window is small), but
the risk is commensurate to the number of masters in an
HA configuration and their relative latency.

In pseudo-code:

**API_SERVER**

```
Admission control (quota)
T1. QUOTA = ...
T2. IF QUOTA.USED + REQUEST >= QUOTA.HARD
T3.   REJECT
T4. UPDATE QUOTA.STATUS(QUOTA.USED + REQUEST)
T5. ADMIT

RESTStorage
T6. Create Object
```

**QUOTA_CONTROLLER**

```
T1. QUOTA := ...
T2. OBSERVED_USAGE := ... [live queries based on latest data]
T3. UPDATE QUOTA.STATUS.USED IFF OBSERVED USAGE != QUOTA.STATUS.USED
```

If the `QUOTA_CONTROLLER.T1` happens after `API_SERVER.T4`,
and `QUOTA_CONTROLLER.T3` happens before `API_SERVER.T6` completes,
then when the quota controller recalculates usage, it could have calculated
usage without observing the outcome of the create.

In practice, this scenario is *extremely rare*, but could happen if
there was a network hiccup of some kind that blocked creation of
the object at time `API_SERVER.T6` in etcd without causing the request
to timeout.  The API server would have also been responding and reading
from that same etcd to support a full quota recalculation and status
update at time `QUOTA_CONTROLLER.T3`.

The next controller pass would recalculate usage correctly.

A solution should be put in place to mitigate this risk and a long-term
plan to handle should be agreed upon.

## Use Cases

* As a `cluster-admin`, I want more reliable quota enforcement.

### ResourceQuota reservations

The recommended solution is for the quota status to track
observed usage and reserved usage as separate values.

The quota admission controller allows a request to be admitted
if the `status.used + status.reserved + request <= status.hard`.
If it chooses to admit the object, it will add a new reservation
to the quota status that captures the `APIVersion`, `Kind`,
and `UID` of the object that made the reservation.  It will also
add a `ExpirationTime` that dictates how long that reservation
is valid.

The quota controller must be updated to become reservation aware
in its synchronization loop.  It would calculate observed usage
as before, but it would compare the set of observed resources
against the set of current reservations.  If an object is observed,
any corresponding reservation made by that object will be removed
since its consumption will properly be tracked in usage.  If a
reservation has expired, it will be removed from the quota document.
A default expirationTime could be applied to sufficiently mitigate
risk based on the anticipated quorum window across masters.

In order to implement this solution, an admission controller must
have access to the UID that will be assigned to an object at creation
time.  This is currently not the case since UID assignment happens
after admission control, and any UID that is assigned by admission
control is overwritten.  The API server code should be updated
to populate ObjectMetaSystemFields prior to invoking admission control.
For this proposal, UID is required since Name would be insufficient
to track against.  For a number of other use cases, code is simplified
if Name is set as well (error-messages come to mind).

## Data Model Impact

```
type ResourceQuotaReservation struct {
  // ReservedBy is a reference to an object that took the
  // reservation.  It must have an APIVersion, Kind, and UID.
  ReservedBy ObjectReference `json:"reservedBy"`
  // Reserved is the incremental set of resources reserved
  Reserved ResourceList `json:"reserved,omitempty"`
  // ExpirationTime defines when the reservation if not realized is expired
  ExpirationTime unversioned.Time `json:"expirationTime,omitempty"`
}

// ResourceQuotaStatus defines the enforced hard limits and observed use
type ResourceQuotaStatus struct {
  // Hard is the set of enforced hard limits for each named resource
  Hard ResourceList `json:"hard,omitempty"`
  // Used is the current observed total usage of the resource in the namespace
  Used ResourceList `json:"used,omitempty"`
  // Reservations is the reserved usage not yet observed in the namespace
  Reservations []ResourceQuotaReservations `json:"reservations,omitempty"`
}
```

## Rest API Impact

None.

## Security Impact

None.

## End User Impact

The `kubectl` commands that render quota should display
reserved usage separate from observed usage.

## Performance Impact

In theory, this feature would allow the quota controller
to run a more frequent loop across its data set to find
reservations and calculate usage specific to a kind
with greater frequency.

It would result in additional writes to the API server as it
expires reservations.

## Developer Impact

None.

## Alternatives

The ability to request a quorum read could be added to the
API server.  The quota controller could request quorum reads
to get a more accurate usage count, but this does not eliminate
the race condition between the quorum read and the quota status
update.

Using a reservation model eliminates the race window without
requring quorum reads.

## Implementation

### Assignee

@derekwaynecarr

### Work Items

* Add support for UID assignment in admission control
* Add support for reservation in admission control
* Add support for expiring reservations in quota controller

## Dependencies

None

## Testing

Appropriate unit and e2e testing will be authored.

## Documentation Impact

Existing resource quota documentation and examples will be updated.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/resource-quota-reservations.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
