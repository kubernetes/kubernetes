package winapi

// Get count from all processor groups.
// https://docs.microsoft.com/en-us/windows/win32/procthread/processor-groups
const ALL_PROCESSOR_GROUPS = 0xFFFF

//sys GetActiveProcessorCount(groupNumber uint16) (amount uint32) = kernel32.GetActiveProcessorCount
