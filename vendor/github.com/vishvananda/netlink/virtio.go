package netlink

// features for virtio net
const (
	VIRTIO_NET_F_CSUM                = 0  // Host handles pkts w/ partial csum
	VIRTIO_NET_F_GUEST_CSUM          = 1  // Guest handles pkts w/ partial csum
	VIRTIO_NET_F_CTRL_GUEST_OFFLOADS = 2  // Dynamic offload configuration.
	VIRTIO_NET_F_MTU                 = 3  // Initial MTU advice
	VIRTIO_NET_F_MAC                 = 5  // Host has given MAC address.
	VIRTIO_NET_F_GUEST_TSO4          = 7  // Guest can handle TSOv4 in.
	VIRTIO_NET_F_GUEST_TSO6          = 8  // Guest can handle TSOv6 in.
	VIRTIO_NET_F_GUEST_ECN           = 9  // Guest can handle TSO[6] w/ ECN in.
	VIRTIO_NET_F_GUEST_UFO           = 10 // Guest can handle UFO in.
	VIRTIO_NET_F_HOST_TSO4           = 11 // Host can handle TSOv4 in.
	VIRTIO_NET_F_HOST_TSO6           = 12 // Host can handle TSOv6 in.
	VIRTIO_NET_F_HOST_ECN            = 13 // Host can handle TSO[6] w/ ECN in.
	VIRTIO_NET_F_HOST_UFO            = 14 // Host can handle UFO in.
	VIRTIO_NET_F_MRG_RXBUF           = 15 // Host can merge receive buffers.
	VIRTIO_NET_F_STATUS              = 16 // virtio_net_config.status available
	VIRTIO_NET_F_CTRL_VQ             = 17 // Control channel available
	VIRTIO_NET_F_CTRL_RX             = 18 // Control channel RX mode support
	VIRTIO_NET_F_CTRL_VLAN           = 19 // Control channel VLAN filtering
	VIRTIO_NET_F_CTRL_RX_EXTRA       = 20 // Extra RX mode control support
	VIRTIO_NET_F_GUEST_ANNOUNCE      = 21 // Guest can announce device on the* network
	VIRTIO_NET_F_MQ                  = 22 // Device supports Receive Flow Steering
	VIRTIO_NET_F_CTRL_MAC_ADDR       = 23 // Set MAC address
	VIRTIO_NET_F_VQ_NOTF_COAL        = 52 // Device supports virtqueue notification coalescing
	VIRTIO_NET_F_NOTF_COAL           = 53 // Device supports notifications coalescing
	VIRTIO_NET_F_GUEST_USO4          = 54 // Guest can handle USOv4 in.
	VIRTIO_NET_F_GUEST_USO6          = 55 // Guest can handle USOv6 in.
	VIRTIO_NET_F_HOST_USO            = 56 // Host can handle USO in.
	VIRTIO_NET_F_HASH_REPORT         = 57 // Supports hash report
	VIRTIO_NET_F_GUEST_HDRLEN        = 59 // Guest provides the exact hdr_len value.
	VIRTIO_NET_F_RSS                 = 60 // Supports RSS RX steering
	VIRTIO_NET_F_RSC_EXT             = 61 // extended coalescing info
	VIRTIO_NET_F_STANDBY             = 62 // Act as standby for another device with the same MAC.
	VIRTIO_NET_F_SPEED_DUPLEX        = 63 // Device set linkspeed and duplex
	VIRTIO_NET_F_GSO                 = 6  // Host handles pkts any GSO type
)

// virtio net status
const (
	VIRTIO_NET_S_LINK_UP  = 1 // Link is up
	VIRTIO_NET_S_ANNOUNCE = 2 // Announcement is needed
)

// virtio config
const (
	// Do we get callbacks when the ring is completely used, even if we've
	// suppressed them?
	VIRTIO_F_NOTIFY_ON_EMPTY = 24
	// Can the device handle any descriptor layout?
	VIRTIO_F_ANY_LAYOUT = 27
	// v1.0 compliant
	VIRTIO_F_VERSION_1 = 32
	// If clear - device has the platform DMA (e.g. IOMMU) bypass quirk feature.
	// If set - use platform DMA tools to access the memory.
	// Note the reverse polarity (compared to most other features),
	// this is for compatibility with legacy systems.
	VIRTIO_F_ACCESS_PLATFORM = 33
	// Legacy name for VIRTIO_F_ACCESS_PLATFORM (for compatibility with old userspace)
	VIRTIO_F_IOMMU_PLATFORM = VIRTIO_F_ACCESS_PLATFORM
	// This feature indicates support for the packed virtqueue layout.
	VIRTIO_F_RING_PACKED = 34
	// Inorder feature indicates that all buffers are used by the device
	// in the same order in which they have been made available.
	VIRTIO_F_IN_ORDER = 35
	// This feature indicates that memory accesses by the driver and the
	// device are ordered in a way described by the platform.
	VIRTIO_F_ORDER_PLATFORM = 36
	// Does the device support Single Root I/O Virtualization?
	VIRTIO_F_SR_IOV = 37
	// This feature indicates that the driver passes extra data (besides
	// identifying the virtqueue) in its device notifications.
	VIRTIO_F_NOTIFICATION_DATA = 38
	// This feature indicates that the driver uses the data provided by the device
	// as a virtqueue identifier in available buffer notifications.
	VIRTIO_F_NOTIF_CONFIG_DATA = 39
	// This feature indicates that the driver can reset a queue individually.
	VIRTIO_F_RING_RESET = 40
)

// virtio device ids
const (
	VIRTIO_ID_NET            = 1  // virtio net
	VIRTIO_ID_BLOCK          = 2  // virtio block
	VIRTIO_ID_CONSOLE        = 3  // virtio console
	VIRTIO_ID_RNG            = 4  // virtio rng
	VIRTIO_ID_BALLOON        = 5  // virtio balloon
	VIRTIO_ID_IOMEM          = 6  // virtio ioMemory
	VIRTIO_ID_RPMSG          = 7  // virtio remote processor messaging
	VIRTIO_ID_SCSI           = 8  // virtio scsi
	VIRTIO_ID_9P             = 9  // 9p virtio console
	VIRTIO_ID_MAC80211_WLAN  = 10 // virtio WLAN MAC
	VIRTIO_ID_RPROC_SERIAL   = 11 // virtio remoteproc serial link
	VIRTIO_ID_CAIF           = 12 // Virtio caif
	VIRTIO_ID_MEMORY_BALLOON = 13 // virtio memory balloon
	VIRTIO_ID_GPU            = 16 // virtio GPU
	VIRTIO_ID_CLOCK          = 17 // virtio clock/timer
	VIRTIO_ID_INPUT          = 18 // virtio input
	VIRTIO_ID_VSOCK          = 19 // virtio vsock transport
	VIRTIO_ID_CRYPTO         = 20 // virtio crypto
	VIRTIO_ID_SIGNAL_DIST    = 21 // virtio signal distribution device
	VIRTIO_ID_PSTORE         = 22 // virtio pstore device
	VIRTIO_ID_IOMMU          = 23 // virtio IOMMU
	VIRTIO_ID_MEM            = 24 // virtio mem
	VIRTIO_ID_SOUND          = 25 // virtio sound
	VIRTIO_ID_FS             = 26 // virtio filesystem
	VIRTIO_ID_PMEM           = 27 // virtio pmem
	VIRTIO_ID_RPMB           = 28 // virtio rpmb
	VIRTIO_ID_MAC80211_HWSIM = 29 // virtio mac80211-hwsim
	VIRTIO_ID_VIDEO_ENCODER  = 30 // virtio video encoder
	VIRTIO_ID_VIDEO_DECODER  = 31 // virtio video decoder
	VIRTIO_ID_SCMI           = 32 // virtio SCMI
	VIRTIO_ID_NITRO_SEC_MOD  = 33 // virtio nitro secure module
	VIRTIO_ID_I2C_ADAPTER    = 34 // virtio i2c adapter
	VIRTIO_ID_WATCHDOG       = 35 // virtio watchdog
	VIRTIO_ID_CAN            = 36 // virtio can
	VIRTIO_ID_DMABUF         = 37 // virtio dmabuf
	VIRTIO_ID_PARAM_SERV     = 38 // virtio parameter server
	VIRTIO_ID_AUDIO_POLICY   = 39 // virtio audio policy
	VIRTIO_ID_BT             = 40 // virtio bluetooth
	VIRTIO_ID_GPIO           = 41 // virtio gpio
	// Virtio Transitional IDs
	VIRTIO_TRANS_ID_NET     = 0x1000 // transitional virtio net
	VIRTIO_TRANS_ID_BLOCK   = 0x1001 // transitional virtio block
	VIRTIO_TRANS_ID_BALLOON = 0x1002 // transitional virtio balloon
	VIRTIO_TRANS_ID_CONSOLE = 0x1003 // transitional virtio console
	VIRTIO_TRANS_ID_SCSI    = 0x1004 // transitional virtio SCSI
	VIRTIO_TRANS_ID_RNG     = 0x1005 // transitional virtio rng
	VIRTIO_TRANS_ID_9P      = 0x1009 // transitional virtio 9p console
)
