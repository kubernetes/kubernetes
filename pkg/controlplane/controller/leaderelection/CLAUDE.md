# Package: leaderelection

## Purpose
This package implements Coordinated Leader Election (CLE) for Kubernetes control plane components. It elects leaders for leases based on LeaseCandidate resources, favoring candidates with the oldest emulation version to support safe version skew during upgrades.

## Key Types

- **Controller**: Observes Lease and LeaseCandidate resources, elects leaders for leases
- **LeaseCandidateGCController**: Garbage collects expired LeaseCandidate resources
- **LeaderElectionTimers**: Configuration for lease duration, renew deadline, and retry period

## Key Functions

- **NewController()**: Creates the main election controller watching Leases and LeaseCandidates
- **reconcileElectionStep()**: Steps through an election - checks if needed, pings candidates, collects acks, elects leader
- **pickBestLeaderOldestEmulationVersion()**: Selects candidate with lowest emulation version, then binary version, then oldest creation time
- **pickBestStrategy()**: Determines election strategy from candidates (currently supports OldestEmulationVersion)
- **RunWithLeaderElection()**: Helper to run a controller function with standard leader election
- **NewLeaseCandidateGC()**: Creates GC controller for expired candidates

## Election Process

1. Check if election needed (lease expired, no holder, better candidate exists)
2. Ping all candidates by setting PingTime
3. Wait for candidates to ack by updating RenewTime
4. After electionDuration (5s), select best candidate from those who acked
5. Create or update Lease with elected holder

## Design Notes

- Uses OldestEmulationVersion strategy: prefers lower emulation version for safe upgrades
- LeaseCandidates expire after 30 minutes without renewal
- Elections re-run every 15 minutes to update all candidates
- Supports PreferredHolder field for graceful leader transitions
- GC controller runs every 30 minutes to clean up expired candidates
