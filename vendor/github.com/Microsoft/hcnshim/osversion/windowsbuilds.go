package osversion

// Windows Client and Server build numbers.
//
// See:
// https://learn.microsoft.com/en-us/windows/release-health/release-information
// https://learn.microsoft.com/en-us/windows/release-health/windows-server-release-info
// https://learn.microsoft.com/en-us/windows/release-health/windows11-release-information
const (
	// RS1 (version 1607, codename "Redstone 1") corresponds to Windows Server
	// 2016 (ltsc2016) and Windows 10 (Anniversary Update).
	RS1 = 14393
	// V1607 (version 1607, codename "Redstone 1") is an alias for [RS1].
	V1607 = RS1
	// LTSC2016 (Windows Server 2016) is an alias for [RS1].
	LTSC2016 = RS1

	// RS2 (version 1703, codename "Redstone 2") was a client-only update, and
	// corresponds to Windows 10 (Creators Update).
	RS2 = 15063
	// V1703 (version 1703, codename "Redstone 2") is an alias for [RS2].
	V1703 = RS2

	// RS3 (version 1709, codename "Redstone 3") corresponds to Windows Server
	// 1709 (Semi-Annual Channel (SAC)), and Windows 10 (Fall Creators Update).
	RS3 = 16299
	// V1709 (version 1709, codename "Redstone 3") is an alias for [RS3].
	V1709 = RS3

	// RS4 (version 1803, codename "Redstone 4") corresponds to Windows Server
	// 1803 (Semi-Annual Channel (SAC)), and Windows 10 (April 2018 Update).
	RS4 = 17134
	// V1803 (version 1803, codename "Redstone 4") is an alias for [RS4].
	V1803 = RS4

	// RS5 (version 1809, codename "Redstone 5") corresponds to Windows Server
	// 2019 (ltsc2019), and Windows 10 (October 2018 Update).
	RS5 = 17763
	// V1809 (version 1809, codename "Redstone 5") is an alias for [RS5].
	V1809 = RS5
	// LTSC2019 (Windows Server 2019) is an alias for [RS5].
	LTSC2019 = RS5

	// V19H1 (version 1903, codename 19H1) corresponds to Windows Server 1903 (semi-annual
	// channel).
	V19H1 = 18362
	// V1903 (version 1903) is an alias for [V19H1].
	V1903 = V19H1

	// V19H2 (version 1909, codename 19H2) corresponds to Windows Server 1909 (semi-annual
	// channel).
	V19H2 = 18363
	// V1909 (version 1909) is an alias for [V19H2].
	V1909 = V19H2

	// V20H1 (version 2004, codename 20H1) corresponds to Windows Server 2004 (semi-annual
	// channel).
	V20H1 = 19041
	// V2004 (version 2004) is an alias for [V20H1].
	V2004 = V20H1

	// V20H2 corresponds to Windows Server 20H2 (semi-annual channel).
	V20H2 = 19042

	// V21H1 corresponds to Windows Server 21H1 (semi-annual channel).
	V21H1 = 19043

	// V21H2Win10 corresponds to Windows 10 (November 2021 Update).
	V21H2Win10 = 19044

	// V21H2Server corresponds to Windows Server 2022 (ltsc2022).
	V21H2Server = 20348
	// LTSC2022 (Windows Server 2022) is an alias for [V21H2Server]
	LTSC2022 = V21H2Server

	// V21H2Win11 corresponds to Windows 11 (original release).
	V21H2Win11 = 22000

	// V22H2Win10 corresponds to Windows 10 (2022 Update).
	V22H2Win10 = 19045

	// V22H2Win11 corresponds to Windows 11 (2022 Update).
	V22H2Win11 = 22621
)
