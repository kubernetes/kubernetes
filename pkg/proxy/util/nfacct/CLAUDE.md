# Package: nfacct

This package provides an interface for managing Linux nfacct (netfilter accounting) objects, used for tracking packet/byte counters in nftables rules.

## Key Types

- `Interface` - Main interface for nfacct operations
- `Counter` - Represents an nfacct accounting object with name, packets, and bytes

## Key Functions

- `Ensure()` - Creates an nfacct counter if it doesn't exist
- `Add()` - Creates a new nfacct counter (errors if exists)
- `Get()` - Retrieves counter values by name
- `List()` - Lists all nfacct counters

## Design Notes

- Used by the nftables proxier for traffic accounting
- Counters persist across rule updates, unlike iptables counters
- Enables accurate service traffic metrics
- Works with the nfnetlink_acct kernel module
